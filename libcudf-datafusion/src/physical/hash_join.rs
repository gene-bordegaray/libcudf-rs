use crate::errors::cudf_to_df;
use crate::physical::cudf_load::{cast_to_target_schema, cudf_schema_compatibility_map};
use arrow::array::{Array, RecordBatch};
use arrow::compute::concat_batches;
use arrow_schema::{Schema, SchemaRef};
use datafusion::common::{JoinType, NullEquality, Statistics};
use datafusion::error::DataFusionError;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion_physical_plan::expressions::Column;
use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    execute_stream, DisplayAs, DisplayFormatType, ExecutionPlan, PhysicalExpr, PlanProperties,
};
use futures::StreamExt;
use futures_util::TryStreamExt;
use libcudf_rs::{full_join, inner_join, left_join, CuDFTable, CuDFTableView};
use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;
use tokio::sync::OnceCell;

/// GPU-accelerated hash join execution node.
///
/// Replaces DataFusion's `HashJoinExec` for equi-joins where all keys are
/// simple column references. Supports `Inner`, `Left`, and `Full` join types.
/// The right side is uploaded from host memory, the left side is expected to
/// already be on GPU (via `CuDFLoadExec`).
pub struct CuDFHashJoinExec {
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    on: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)>,
    join_type: JoinType,
    projection: Option<Vec<usize>>,
    partition_mode: PartitionMode,
    properties: PlanProperties,
    statistics: Statistics,
    shared_table: Arc<OnceCell<Arc<CuDFTable>>>,
}

impl std::fmt::Debug for CuDFHashJoinExec {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.debug_struct("CuDFHashJoinExec")
            .field("join_type", &self.join_type)
            .field("partition_mode", &self.partition_mode)
            .finish()
    }
}

/// Derive plan properties from `node` with the schema normalized for cuDF types.
fn normalized_properties(node: &HashJoinExec) -> PlanProperties {
    PlanProperties::new(
        EquivalenceProperties::new(cudf_schema_compatibility_map(node.schema())),
        node.properties().partitioning.clone(),
        node.properties().emission_type,
        node.properties().boundedness,
    )
}

impl CuDFHashJoinExec {
    pub fn try_new(node: &HashJoinExec) -> Result<Self, DataFusionError> {
        let properties = normalized_properties(node);
        // Guard before calling partition_statistics: DataFusion's estimate_join_cardinality
        // (joins/utils.rs) does an unchecked `column_statistics[index]` access and panics
        // when an on-key column index is out of bounds.
        //
        // How the stale index arises during transform_up:
        //
        // Before optimizer adds projection=[0,1] to inner join:
        // ```
        // HashJoinExec (outer)
        //   on: [left.outer_key @ idx=3 = r.outer_key @ idx=0]
        //   left  -> HashJoinExec (inner)   schema: [key(0), val(1), key(2), outer_key(3)]
        //   right -> r                      schema: [outer_key(0), result(1)]
        // ```
        //
        // After optimizer narrows inner join output to projection=[0,1]:
        // ```
        // HashJoinExec (outer)                       <- on-keys NOT re-validated
        //   on: [left.outer_key @ idx=3 = ...]       <- idx=3 is now STALE
        //   left  -> CuDFHashJoinExec (inner)  schema: [key(0), val(1)]  <- only 2 cols
        //   right -> r
        // ```
        //
        // Calling partition_statistics on the outer join then triggers:
        //   column_statistics[3] on a Vec with 2 entries -> PANIC
        //
        // HashJoinExec::with_new_children does not re-validate on-key indices against the
        // new child schemas, so this state is reachable. We skip statistics rather than panic.
        let on_indices_valid = node.on().iter().all(|(l, r)| {
            let l_ok = l
                .as_any()
                .downcast_ref::<Column>()
                .map(|c| c.index() < node.left().schema().fields().len())
                .unwrap_or(true);
            let r_ok = r
                .as_any()
                .downcast_ref::<Column>()
                .map(|c| c.index() < node.right().schema().fields().len())
                .unwrap_or(true);
            l_ok && r_ok
        });
        let statistics = if on_indices_valid {
            node.partition_statistics(None)
                .unwrap_or_else(|_| Statistics::new_unknown(&node.schema()))
        } else {
            Statistics::new_unknown(&node.schema())
        };
        Ok(Self {
            left: node.left().clone(),
            right: node.right().clone(),
            on: node.on().to_vec(),
            join_type: *node.join_type(),
            projection: node.projection.clone(),
            partition_mode: *node.partition_mode(),
            properties,
            statistics,
            shared_table: Arc::new(OnceCell::new()),
        })
    }
}

impl DisplayAs for CuDFHashJoinExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        let on_keys: Vec<String> = self.on.iter().map(|(l, r)| format!("{l} = {r}")).collect();
        write!(
            f,
            "CuDFHashJoinExec: mode={:?}, join_type={:?}, on=[{}]",
            self.partition_mode,
            self.join_type,
            on_keys.join(", ")
        )
    }
}

impl ExecutionPlan for CuDFHashJoinExec {
    fn name(&self) -> &str {
        "CuDFHashJoinExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn partition_statistics(
        &self,
        partition: Option<usize>,
    ) -> Result<Statistics, DataFusionError> {
        if partition.is_some() {
            return Ok(Statistics::new_unknown(&self.schema()));
        }
        Ok(self.statistics.clone())
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let right = children.swap_remove(1);
        let left = children.swap_remove(0);
        Ok(Arc::new(Self {
            left,
            right,
            on: self.on.clone(),
            join_type: self.join_type,
            projection: self.projection.clone(),
            partition_mode: self.partition_mode,
            properties: self.properties.clone(),
            statistics: Statistics::new_unknown(&self.schema()),
            shared_table: Arc::new(OnceCell::new()),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::common::Result<SendableRecordBatchStream> {
        let right_stream = Arc::clone(&self.right).execute(partition, Arc::clone(&context))?;

        // CollectLeft: all right-side partition streams share one left table via OnceCell,
        // so the left child is executed at most once regardless of output partition count.
        // Partitioned/Auto: each partition builds its own left table independently.
        let left_fut = match &self.partition_mode {
            PartitionMode::CollectLeft => collect_shared(
                Arc::clone(&self.shared_table),
                Arc::clone(&self.left),
                Arc::clone(&context),
            ),
            _ => {
                let left_stream = self.left.execute(partition, Arc::clone(&context))?;
                Box::pin(async move {
                    let batches: Vec<RecordBatch> = left_stream.try_collect().await?;
                    batches_to_table(&batches).map(Arc::new).map_err(cudf_to_df)
                })
            }
        };

        let params = JoinParams {
            join_type: self.join_type,
            left_on: self
                .on
                .iter()
                .map(|(l, _)| {
                    l.as_any()
                        .downcast_ref::<Column>()
                        .ok_or_else(|| {
                            DataFusionError::Internal(
                                "CuDFHashJoinExec: left join key is not a Column expression".into(),
                            )
                        })
                        .map(|c| c.index())
                })
                .collect::<Result<_, _>>()?,
            right_on: self
                .on
                .iter()
                .map(|(_, r)| {
                    r.as_any()
                        .downcast_ref::<Column>()
                        .ok_or_else(|| {
                            DataFusionError::Internal(
                                "CuDFHashJoinExec: right join key is not a Column expression"
                                    .into(),
                            )
                        })
                        .map(|c| c.index())
                })
                .collect::<Result<_, _>>()?,
            projection: self.projection.clone(),
            output_schema: self.schema(),
            join_schema: cudf_schema_compatibility_map(Arc::new(Schema::new(
                self.left
                    .schema()
                    .fields()
                    .iter()
                    .chain(self.right.schema().fields())
                    .cloned()
                    .collect::<Vec<_>>(),
            ))),
        };

        let stream = futures::stream::once(async move {
            let left = left_fut.await?;
            let right_batches: Vec<RecordBatch> = right_stream.try_collect().await?;
            let join_result = perform_join(&left, &right_batches, &params)?;
            Ok(join_result)
        })
        .filter_map(|result| async move {
            match result {
                Ok(Some(batch)) => Some(Ok(batch)),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            }
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}

/// Materialize the left (build) side once and share across partitions via `OnceCell`.
fn collect_shared(
    shared: Arc<OnceCell<Arc<CuDFTable>>>,
    left_child: Arc<dyn ExecutionPlan>,
    ctx: Arc<TaskContext>,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<Arc<CuDFTable>, DataFusionError>> + Send>,
> {
    Box::pin(async move {
        shared
            .get_or_try_init(|| async move {
                let stream = execute_stream(left_child, ctx).map_err(Arc::new)?;
                let batches: Vec<RecordBatch> = stream.try_collect().await.map_err(Arc::new)?;
                batches_to_table(&batches)
                    .map(Arc::new)
                    .map_err(|e| Arc::new(cudf_to_df(e)))
            })
            .await
            .map(Arc::clone)
            .map_err(|e: Arc<DataFusionError>| DataFusionError::External(Box::new(e)))
    })
}

/// Everything `perform_join` needs, extracted from `self`.
///
/// Two schemas are required because cuDF joins work in two steps:
/// 1. The raw join result has all left and right columns (`join_schema`).
///   `to_record_batch_with_schema` uses this to restore correct types.
/// 2. `apply_projection` then selects the final output columns and attaches `output_schema`,
///    which is what downstream operators expect.
struct JoinParams {
    join_type: JoinType,
    left_on: Vec<usize>,
    right_on: Vec<usize>,
    projection: Option<Vec<usize>>,
    join_schema: SchemaRef,
    output_schema: SchemaRef,
}

/// Run the cuDF join kernel and apply the output projection. Returns `None`
/// for inner joins with no matching rows.
fn perform_join(
    left: &Arc<CuDFTable>,
    right_batches: &[RecordBatch],
    params: &JoinParams,
) -> Result<Option<RecordBatch>, DataFusionError> {
    let JoinParams {
        join_type,
        left_on,
        right_on,
        projection,
        join_schema,
        output_schema,
    } = params;
    let right_empty = right_batches.is_empty() || right_batches.iter().all(|b| b.num_rows() == 0);

    // Inner join with no right rows: no matches possible, emit nothing.
    if matches!(join_type, JoinType::Inner) && right_empty {
        return Ok(None);
    }

    // TODO: Left/Full join with a completely empty right partition (no batches, not even a
    // zero-row batch) should return all left rows with nulls in the right columns.
    // host_batches_to_table panics on batches[0] when the slice is empty.
    // Fix: carry the right child schema here and synthesise an empty CuDFTable from it.
    if right_batches.is_empty() {
        return Err(DataFusionError::NotImplemented(
            "CuDFHashJoinExec: Left/Full join with zero right-side batches is not yet supported"
                .to_string(),
        ));
    }

    let left_view = Arc::clone(left).view();
    let right_table = host_batches_to_table(right_batches)?;
    let right_view = right_table.into_view();

    let result = match join_type {
        JoinType::Inner => inner_join(&left_view, &right_view, left_on, right_on),
        JoinType::Left => left_join(&left_view, &right_view, left_on, right_on),
        JoinType::Full => full_join(&left_view, &right_view, left_on, right_on),
        other => {
            return Err(DataFusionError::NotImplemented(format!(
                "CuDFHashJoinExec: unsupported join type {other:?}"
            )))
        }
    }
    .map_err(cudf_to_df)?;

    let batch = result
        .into_view()
        .to_record_batch_with_schema(join_schema)
        .map_err(cudf_to_df)?;
    apply_projection(batch, projection, Arc::clone(output_schema)).map(Some)
}

/// Concat GPU-resident record batches (already `CuDFColumnView` arrays) into one table.
fn batches_to_table(batches: &[RecordBatch]) -> Result<CuDFTable, libcudf_rs::CuDFError> {
    let views: Vec<CuDFTableView> = batches
        .iter()
        .map(CuDFTableView::from_record_batch)
        .collect::<Result<_, _>>()?;
    CuDFTable::concat(views)
}

/// Concat host batches on CPU, then upload to GPU in one transfer.
fn host_batches_to_table(batches: &[RecordBatch]) -> Result<CuDFTable, DataFusionError> {
    let schema = cudf_schema_compatibility_map(batches[0].schema());
    let cast: Vec<RecordBatch> = batches
        .iter()
        .map(|b| cast_to_target_schema(b.clone(), Arc::clone(&schema)))
        .collect::<Result<_, _>>()?;
    let batch = concat_batches(&schema, &cast)?;
    CuDFTable::from_arrow_host(batch).map_err(cudf_to_df)
}

/// Select output columns per the join's projection and attach the output schema.
fn apply_projection(
    batch: RecordBatch,
    projection: &Option<Vec<usize>>,
    schema: SchemaRef,
) -> Result<RecordBatch, DataFusionError> {
    match projection {
        Some(proj) => {
            let cols: Vec<Arc<dyn Array>> =
                proj.iter().map(|&i| Arc::clone(batch.column(i))).collect();
            Ok(RecordBatch::try_new(schema, cols)?)
        }
        None => Ok(RecordBatch::try_new(schema, batch.columns().to_vec())?),
    }
}

/// Try to convert a `HashJoinExec` to GPU. Returns `None` for unsupported
/// configurations: non-column keys, non-equi filters, unsupported join types.
pub fn try_as_cudf_hash_join(
    node: &HashJoinExec,
) -> Result<Option<Arc<dyn ExecutionPlan>>, DataFusionError> {
    for (l, r) in node.on() {
        if l.as_any().downcast_ref::<Column>().is_none()
            || r.as_any().downcast_ref::<Column>().is_none()
        {
            return Ok(None);
        }
    }

    match node.join_type() {
        JoinType::Inner | JoinType::Left | JoinType::Full => {}
        _ => return Ok(None),
    }

    if node.null_equality() != NullEquality::NullEqualsNothing {
        return Ok(None);
    }

    if node.filter().is_some() {
        return Ok(None);
    }

    Ok(Some(Arc::new(CuDFHashJoinExec::try_new(node)?)))
}

#[cfg(test)]
mod test {
    use super::{try_as_cudf_hash_join, CuDFHashJoinExec};
    use crate::physical::{CuDFLoadExec, CuDFUnloadExec};
    use arrow::array::record_batch;
    use arrow::array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::common::{JoinSide, JoinType, NullEquality};
    use datafusion::execution::TaskContext;
    use datafusion_physical_plan::expressions::Column;
    use datafusion_physical_plan::joins::utils::{ColumnIndex, JoinFilter};
    use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
    use datafusion_physical_plan::test::TestMemoryExec;
    use datafusion_physical_plan::{ExecutionPlan, PhysicalExpr};
    use futures_util::TryStreamExt;
    use std::error::Error;
    use std::sync::Arc;

    fn left_batch() -> RecordBatch {
        record_batch!(
            ("key", Int32, [1, 2, 3, 4]),
            ("val", Int32, [10, 20, 30, 40])
        )
        .unwrap()
    }

    fn right_batch() -> RecordBatch {
        record_batch!(("key", Int32, [2, 3, 5]), ("val", Int32, [200, 300, 500])).unwrap()
    }

    fn empty_right() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("val", DataType::Int32, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(Int32Array::from(Vec::<i32>::new())),
            ],
        )
        .unwrap()
    }

    async fn run_join(
        left: RecordBatch,
        right: RecordBatch,
        join_type: JoinType,
        partition_mode: PartitionMode,
    ) -> Result<Vec<RecordBatch>, Box<dyn Error>> {
        let ls = left.schema();
        let rs = right.schema();
        // Left (build) side: optimizer inserts CuDFLoadExec, so simulate that here.
        let left_in = Arc::new(CuDFLoadExec::try_new(Arc::new(TestMemoryExec::try_new(
            &[vec![left]],
            ls.clone(),
            None,
        )?))?);
        // Right (probe) side: CuDFHashJoinExec owns the upload, no CuDFLoadExec.
        let right_in = Arc::new(TestMemoryExec::try_new(&[vec![right]], rs.clone(), None)?);
        let on = vec![(
            Arc::new(Column::new("key", 0)) as Arc<dyn PhysicalExpr>,
            Arc::new(Column::new("key", 0)) as Arc<dyn PhysicalExpr>,
        )];
        let inner = HashJoinExec::try_new(
            left_in,
            right_in,
            on,
            None,
            &join_type,
            None,
            partition_mode,
            NullEquality::NullEqualsNothing,
        )?;
        let unload = CuDFUnloadExec::new(Arc::new(CuDFHashJoinExec::try_new(&inner)?));
        let stream = unload.execute(0, Arc::new(TaskContext::default()))?;
        Ok(stream.try_collect::<Vec<_>>().await?)
    }

    fn total_rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(|b| b.num_rows()).sum()
    }

    #[tokio::test]
    async fn test_inner_join() -> Result<(), Box<dyn Error>> {
        let out = run_join(
            left_batch(),
            right_batch(),
            JoinType::Inner,
            PartitionMode::CollectLeft,
        )
        .await?;
        assert_eq!(total_rows(&out), 2); // keys 2 and 3 match
        assert_eq!(out[0].num_columns(), 4);
        Ok(())
    }

    #[tokio::test]
    async fn test_inner_join_empty_right() -> Result<(), Box<dyn Error>> {
        let out = run_join(
            left_batch(),
            empty_right(),
            JoinType::Inner,
            PartitionMode::CollectLeft,
        )
        .await?;
        assert_eq!(total_rows(&out), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_inner_join_partitioned() -> Result<(), Box<dyn Error>> {
        // Partitioned mode builds the left table per-partition rather than once globally.
        let out = run_join(
            left_batch(),
            right_batch(),
            JoinType::Inner,
            PartitionMode::Partitioned,
        )
        .await?;
        assert_eq!(total_rows(&out), 2); // keys 2 and 3 match
        assert_eq!(out[0].num_columns(), 4);
        Ok(())
    }

    #[tokio::test]
    async fn test_left_join() -> Result<(), Box<dyn Error>> {
        let out = run_join(
            left_batch(),
            right_batch(),
            JoinType::Left,
            PartitionMode::CollectLeft,
        )
        .await?;
        assert_eq!(total_rows(&out), 4); // all 4 left rows preserved
        assert_eq!(out[0].num_columns(), 4);
        Ok(())
    }

    #[tokio::test]
    async fn test_full_join() -> Result<(), Box<dyn Error>> {
        let out = run_join(
            left_batch(),
            right_batch(),
            JoinType::Full,
            PartitionMode::CollectLeft,
        )
        .await?;
        // 2 matches + 2 left-only + 1 right-only = 5
        assert_eq!(total_rows(&out), 5);
        assert_eq!(out[0].num_columns(), 4);
        Ok(())
    }

    #[test]
    fn test_conversion_with_narrowed_child_schema() -> Result<(), Box<dyn Error>> {
        // Three-table schema: ll join lr on key=key, then (ll join lr) join r on outer_key=outer_key.
        let ll_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("val", DataType::Int32, false),
        ]));
        let lr_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("outer_key", DataType::Int32, false),
        ]));
        let r_schema = Arc::new(Schema::new(vec![
            Field::new("outer_key", DataType::Int32, false),
            Field::new("result", DataType::Int32, false),
        ]));

        let ll = Arc::new(TestMemoryExec::try_new(&[], ll_schema, None)?);
        let lr = Arc::new(TestMemoryExec::try_new(&[], lr_schema, None)?);
        let r = Arc::new(TestMemoryExec::try_new(&[], r_schema, None)?);

        let inner_on = vec![(
            Arc::new(Column::new("key", 0)) as Arc<dyn PhysicalExpr>,
            Arc::new(Column::new("key", 0)) as Arc<dyn PhysicalExpr>,
        )];

        // Inner join without projection; full output: [ll.key(0), ll.val(1), lr.key(2), lr.outer_key(3)].
        let inner_full = HashJoinExec::try_new(
            ll.clone(),
            lr.clone(),
            inner_on.clone(),
            None,
            &JoinType::Inner,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
        )?;

        // Outer join references lr.outer_key at index 3 of the full inner output.
        let outer_on = vec![(
            Arc::new(Column::new("outer_key", 3)) as Arc<dyn PhysicalExpr>,
            Arc::new(Column::new("outer_key", 0)) as Arc<dyn PhysicalExpr>,
        )];
        let outer_join = Arc::new(HashJoinExec::try_new(
            Arc::new(inner_full),
            r.clone(),
            outer_on,
            None,
            &JoinType::Inner,
            None,
            PartitionMode::Partitioned,
            NullEquality::NullEqualsNothing,
        )?);

        // Optimizer adds projection=[0,1] to the inner join, narrowing its output to
        // [ll.key, ll.val] and dropping lr.outer_key.
        let inner_projected = HashJoinExec::try_new(
            ll,
            lr,
            inner_on,
            None,
            &JoinType::Inner,
            Some(vec![0, 1]),
            PartitionMode::Partitioned,
            NullEquality::NullEqualsNothing,
        )?;
        let cudf_inner = Arc::new(CuDFHashJoinExec::try_new(&inner_projected)?);

        // transform_up calls with_new_children to inject the narrowed GPU child into the
        // outer join. HashJoinExec::with_new_children does not re-validate on-key columns,
        // so the outer join now holds cudf_inner (schema: [ll.key, ll.val]) as its left
        // child while its on-key still references outer_key at index 3.
        let outer_narrowed = outer_join.with_new_children(vec![cudf_inner, r])?;
        let outer_hj = outer_narrowed
            .as_any()
            .downcast_ref::<HashJoinExec>()
            .unwrap();

        // Must convert this without calling HashJoinExec::try_new (which would
        // re-validate the stale index-3 reference against the 2-column left child).
        let converted = try_as_cudf_hash_join(outer_hj)?;
        assert!(converted.is_some());
        Ok(())
    }

    #[test]
    fn test_join_filter_bails_to_cpu() -> Result<(), Box<dyn Error>> {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
        let left = Arc::new(TestMemoryExec::try_new(&[], schema.clone(), None)?);
        let right = Arc::new(TestMemoryExec::try_new(&[], schema.clone(), None)?);
        let on = vec![(
            Arc::new(Column::new("key", 0)) as Arc<dyn PhysicalExpr>,
            Arc::new(Column::new("key", 0)) as Arc<dyn PhysicalExpr>,
        )];
        let filter = JoinFilter::new(
            Arc::new(Column::new("key", 0)) as Arc<dyn PhysicalExpr>,
            vec![ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            }],
            Arc::clone(&schema),
        );
        let join = HashJoinExec::try_new(
            left,
            right,
            on,
            Some(filter),
            &JoinType::Inner,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
        )?;
        assert!(try_as_cudf_hash_join(&join)?.is_none());
        Ok(())
    }
}

#[cfg(test)]
mod integration {
    use crate::test_utils::TestFramework;
    use datafusion::common::assert_contains;
    use std::error::Error;

    async fn check(sql: &str) -> Result<(), Box<dyn Error>> {
        let tf = TestFramework::new().await;
        let cudf = tf.execute(&format!("SET cudf.enable=true; {sql}")).await?;
        let cpu = tf.execute(sql).await?;
        assert_contains!(&cudf.plan, "CuDFHashJoinExec");
        assert_eq!(cpu.pretty_print, cudf.pretty_print);
        Ok(())
    }

    async fn check_correct(sql: &str) -> Result<(), Box<dyn Error>> {
        let tf = TestFramework::new().await;
        let cudf = tf.execute(&format!("SET cudf.enable=true; {sql}")).await?;
        let cpu = tf.execute(sql).await?;
        assert_eq!(cpu.pretty_print, cudf.pretty_print);
        Ok(())
    }

    #[tokio::test]
    async fn test_inner_join() -> Result<(), Box<dyn Error>> {
        check(
            r#"SELECT a."MinTemp", b."MaxTemp" FROM weather a
               JOIN weather b ON a."MinTemp" = b."MinTemp"
               ORDER BY a."MinTemp", b."MaxTemp" LIMIT 10"#,
        )
        .await
    }

    #[tokio::test]
    async fn test_inner_join_multi_key() -> Result<(), Box<dyn Error>> {
        check(
            r#"SELECT a."MinTemp", a."MaxTemp" FROM weather a
               JOIN weather b ON a."MinTemp" = b."MinTemp" AND a."MaxTemp" = b."MaxTemp"
               ORDER BY a."MinTemp", a."MaxTemp" LIMIT 10"#,
        )
        .await
    }

    #[tokio::test]
    async fn test_full_join() -> Result<(), Box<dyn Error>> {
        check_correct(
            r#"SELECT a."MinTemp", b."MaxTemp" FROM weather a
               FULL JOIN weather b ON a."MinTemp" = b."MinTemp"
               ORDER BY a."MinTemp", b."MaxTemp" LIMIT 10"#,
        )
        .await
    }
}

use crate::errors::cudf_to_df;
use crate::physical::cudf_load::{cast_to_target_schema, cudf_schema_compatibility_map};
use arrow::array::{Array, RecordBatch};
use arrow::compute::concat_batches;
use arrow_schema::SchemaRef;
use datafusion::common::{JoinType, NullEquality, Statistics};
use datafusion::error::DataFusionError;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::expressions::Column;
use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    execute_stream, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};
use delegate::delegate;
use futures::StreamExt;
use futures_util::TryStreamExt;
use libcudf_rs::{full_join, inner_join, left_join, CuDFTable, CuDFTableView};
use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;
use tokio::sync::OnceCell;

pub struct CuDFHashJoinExec {
    inner: HashJoinExec,
    /// Shared table built once, shared across all probe-side partitions.
    shared_table: Arc<OnceCell<Arc<CuDFTable>>>,
}

impl std::fmt::Debug for CuDFHashJoinExec {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.debug_struct("CuDFHashJoinExec")
            .field("inner", &self.inner)
            .finish()
    }
}

impl CuDFHashJoinExec {
    pub fn try_new(inner: HashJoinExec) -> Result<Self, DataFusionError> {
        Ok(Self {
            inner,
            shared_table: Arc::new(OnceCell::new()),
        })
    }
}

impl DisplayAs for CuDFHashJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "CuDF")?;
        self.inner.fmt_as(t, f)
    }
}

impl ExecutionPlan for CuDFHashJoinExec {
    fn name(&self) -> &str {
        "CuDFHashJoinExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let right = children.swap_remove(1);
        let left = children.swap_remove(0);
        let inner = HashJoinExec::try_new(
            left,
            right,
            self.inner.on().to_vec(),
            self.inner.filter().cloned(),
            self.inner.join_type(),
            self.inner.projection.clone(),
            *self.inner.partition_mode(),
            self.inner.null_equality(),
        )?;
        Ok(Arc::new(Self::try_new(inner)?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::common::Result<SendableRecordBatchStream> {
        let right_stream = self
            .inner
            .right()
            .execute(partition, Arc::clone(&context))?;

        // CollectLeft: all right-side partition streams share one left table via OnceCell,
        // so the left child is executed at most once regardless of output partition count.
        // Partitioned/Auto: each partition builds its own left table independently.
        let left_fut = match self.inner.partition_mode() {
            PartitionMode::CollectLeft => collect_shared(
                Arc::clone(&self.shared_table),
                Arc::clone(self.inner.left()),
                Arc::clone(&context),
            ),
            _ => {
                let left_stream = self.inner.left().execute(partition, Arc::clone(&context))?;
                Box::pin(async move {
                    let batches: Vec<RecordBatch> = left_stream.try_collect().await?;
                    batches_to_table(&batches).map(Arc::new).map_err(cudf_to_df)
                })
            }
        };

        let join_type = *self.inner.join_type();
        let left_on: Vec<usize> = self
            .inner
            .on()
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
            .collect::<Result<_, _>>()?;
        let right_on: Vec<usize> = self
            .inner
            .on()
            .iter()
            .map(|(_, r)| {
                r.as_any()
                    .downcast_ref::<Column>()
                    .ok_or_else(|| {
                        DataFusionError::Internal(
                            "CuDFHashJoinExec: right join key is not a Column expression".into(),
                        )
                    })
                    .map(|c| c.index())
            })
            .collect::<Result<_, _>>()?;
        let projection = self.inner.projection.clone();
        let output_schema = self.schema();

        let stream = futures::stream::once(async move {
            let left = left_fut.await?;
            let right_batches: Vec<RecordBatch> = right_stream.try_collect().await?;
            let join_result = perform_join(
                &left,
                &right_batches,
                join_type,
                &left_on,
                &right_on,
                &projection,
                &output_schema,
            )?;
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

    delegate! {
        to self.inner {
            fn properties(&self) -> &PlanProperties;
            fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>>;
            fn partition_statistics(&self, partition: Option<usize>) -> Result<Statistics, DataFusionError>;
        }
    }
}

/// Builds the left `CuDFTable` once for `CollectLeft` mode. Concurrent callers wait for the
/// first to complete via `OnceCell`; the error type is wrapped in `Arc` to satisfy the `Clone`
/// bound required by `get_or_try_init`.
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

fn perform_join(
    left: &Arc<CuDFTable>,
    right_batches: &[RecordBatch],
    join_type: JoinType,
    left_on: &[usize],
    right_on: &[usize],
    projection: &Option<Vec<usize>>,
    output_schema: &SchemaRef,
) -> Result<Option<RecordBatch>, DataFusionError> {
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

    let batch = result.into_view().to_record_batch().map_err(cudf_to_df)?;
    apply_projection(batch, projection, Arc::clone(output_schema)).map(Some)
}

fn batches_to_table(batches: &[RecordBatch]) -> Result<CuDFTable, libcudf_rs::CuDFError> {
    let views: Vec<CuDFTableView> = batches
        .iter()
        .map(CuDFTableView::from_record_batch)
        .collect::<Result<_, _>>()?;
    CuDFTable::concat(views)
}

/// Upload host Arrow batches to GPU in a single bulk transfer.
///
/// All batches are concatenated on the CPU first (cheap pointer work for Arrow), then uploaded
/// via one `CuDFTable::from_arrow_host` call. This avoids the per-batch DMA overhead and the
/// GPU-to-GPU concat that would otherwise be needed if each batch were uploaded separately.
fn host_batches_to_table(batches: &[RecordBatch]) -> Result<CuDFTable, DataFusionError> {
    let schema = cudf_schema_compatibility_map(batches[0].schema());
    let cast: Vec<RecordBatch> = batches
        .iter()
        .map(|b| cast_to_target_schema(b.clone(), Arc::clone(&schema)))
        .collect::<Result<_, _>>()?;
    let batch = concat_batches(&schema, &cast)?;
    CuDFTable::from_arrow_host(batch).map_err(cudf_to_df)
}

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

/// Returns `Some(CuDFHashJoinExec)` if all equi-join keys are simple column references,
/// the join type is `Inner`, `Left`, or `Full`, and null equality is the SQL default
/// (`NullEqualsNothing`); otherwise `None` (CPU fallback).
///
/// TODO: wire up `LeftSemi`, `LeftAnti`, and `Right` join types - the libcudf_rs layer
/// already exposes `left_semi_join`, `left_anti_join` but `perform_join` does not handle them.
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

    // cuDF inner/left/full_join uses UNEQUAL null semantics (nulls don't match nulls),
    // matching SQL standard. Fall back to CPU for non-default NullEqualsNull to avoid
    // silently producing wrong results.
    if node.null_equality() != NullEquality::NullEqualsNothing {
        return Ok(None);
    }

    // HashJoinExec doesn't impl Clone; reconstruct from its public fields.
    let inner = HashJoinExec::try_new(
        node.left().clone(),
        node.right().clone(),
        node.on().to_vec(),
        node.filter().cloned(),
        node.join_type(),
        node.projection.clone(),
        *node.partition_mode(),
        node.null_equality(),
    )?;
    Ok(Some(Arc::new(CuDFHashJoinExec::try_new(inner)?)))
}

/// Unit tests: exercise `CuDFHashJoinExec` directly with controlled in-memory data.
///
/// Pipeline: `TestMemoryExec` -> `CuDFLoadExec` -> `CuDFHashJoinExec` -> `CuDFUnloadExec`.
/// Left:  key [1,2,3,4], val [10,20,30,40]
/// Right: key [2,3,5],   val [200,300,500]
#[cfg(test)]
mod test {
    use super::CuDFHashJoinExec;
    use crate::physical::{CuDFLoadExec, CuDFUnloadExec};
    use arrow::array::record_batch;
    use arrow::array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::common::{JoinType, NullEquality};
    use datafusion::execution::TaskContext;
    use datafusion_physical_plan::expressions::Column;
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
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
        )?;
        let unload = CuDFUnloadExec::new(Arc::new(CuDFHashJoinExec::try_new(inner)?));
        let stream = unload.execute(0, Arc::new(TaskContext::default()))?;
        Ok(stream.try_collect::<Vec<_>>().await?)
    }

    fn total_rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(|b| b.num_rows()).sum()
    }

    #[tokio::test]
    async fn test_inner_join() -> Result<(), Box<dyn Error>> {
        let out = run_join(left_batch(), right_batch(), JoinType::Inner).await?;
        assert_eq!(total_rows(&out), 2); // keys 2 and 3 match
        assert_eq!(out[0].num_columns(), 4);
        Ok(())
    }

    #[tokio::test]
    async fn test_inner_join_empty_right() -> Result<(), Box<dyn Error>> {
        let out = run_join(left_batch(), empty_right(), JoinType::Inner).await?;
        assert_eq!(total_rows(&out), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_left_join() -> Result<(), Box<dyn Error>> {
        let out = run_join(left_batch(), right_batch(), JoinType::Left).await?;
        assert_eq!(total_rows(&out), 4); // all 4 left rows preserved
        assert_eq!(out[0].num_columns(), 4);
        Ok(())
    }

    #[tokio::test]
    async fn test_full_join() -> Result<(), Box<dyn Error>> {
        let out = run_join(left_batch(), right_batch(), JoinType::Full).await?;
        // 2 matches + 2 left-only + 1 right-only = 5
        assert_eq!(total_rows(&out), 5);
        assert_eq!(out[0].num_columns(), 4);
        Ok(())
    }
}

/// Integration tests: full SQL pipeline via `TestFramework` against real weather data.
///
/// Each test runs the same query on GPU and CPU and asserts results are identical.
/// `check()` additionally verifies the plan contains `CuDFHashJoinExec`.
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

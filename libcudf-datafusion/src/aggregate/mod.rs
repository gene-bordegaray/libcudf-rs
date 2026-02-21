use arrow_schema::Schema;
use datafusion::error::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::projection::ProjectionMapping;
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use datafusion_physical_plan::udaf::AggregateFunctionExpr;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, InputOrderMode, PlanProperties,
};
use std::any::{type_name, Any};
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

mod op;
mod stream;

pub(crate) use op::CuDFAggregationOp;

/// GPU-accelerated GROUP BY aggregate execution node.
///
/// Replaces DataFusion's `AggregateExec` for queries where all aggregate
/// functions have cuDF implementations.
#[derive(Debug)]
pub struct CuDFAggregateExec {
    input: Arc<dyn ExecutionPlan>,
    mode: AggregateMode,
    group_by: PhysicalGroupBy,
    aggr_expr: Vec<Arc<AggregateFunctionExpr>>,

    plan_properties: PlanProperties,
}

impl CuDFAggregateExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        mode: AggregateMode,
        group_by: PhysicalGroupBy,
        aggr_expr: Vec<Arc<AggregateFunctionExpr>>,
    ) -> Result<Self> {
        let input_schema = input.schema();

        // Non-single grouping sets (CUBE, ROLLUP) add an extra column for the grouping ID.
        let group_by_fields = {
            let num_exprs = group_by.expr().len();
            if !group_by.is_single() {
                num_exprs + 1
            } else {
                num_exprs
            }
        };

        let group_by_schema = group_by.group_schema(&input_schema)?;
        let group_by_exprs = group_by_schema.fields.iter().take(group_by_fields).cloned();

        let mut fields = Vec::with_capacity(group_by_fields + aggr_expr.len());

        fields.extend(group_by_exprs);

        // Partial mode emits intermediate state columns (e.g., AVG emits [count, sum]).
        // All other modes emit the final result column (e.g., AVG emits [avg]).
        if mode == AggregateMode::Partial {
            for expr in &aggr_expr {
                for field in expr.state_fields()? {
                    fields.push(field);
                }
            }
        } else {
            for expr in &aggr_expr {
                fields.push(expr.field());
            }
        }

        let output_schema = Arc::new(Schema::new_with_metadata(
            fields,
            input_schema.metadata.clone(),
        ));

        let group_by_expr_mapping =
            ProjectionMapping::try_new(group_by.expr().iter().cloned(), &input.schema())?;

        let plan_properties = AggregateExec::compute_properties(
            &input,
            output_schema,
            &group_by_expr_mapping,
            &mode,
            &InputOrderMode::Linear,
            &aggr_expr,
        )?;

        Ok(Self {
            input,
            mode,
            group_by,
            aggr_expr,
            plan_properties,
        })
    }
}

impl DisplayAs for CuDFAggregateExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "CuDFAggregateExec: ")?;
        write!(f, "mode={:?}, ", self.mode)?;
        write!(f, "group_by=[")?;
        for (i, (expr, alias)) in self.group_by.expr().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}@{}", alias, expr)?;
        }
        write!(f, "], aggr_expr=[")?;
        for (i, expr) in self.aggr_expr.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", expr.name())?;
        }
        write!(f, "]")
    }
}

impl ExecutionPlan for CuDFAggregateExec {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let new = Self::try_new(
            children[0].clone(),
            self.mode,
            self.group_by.clone(),
            self.aggr_expr.clone(),
        )?;

        Ok(Arc::new(new))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let stream = stream::Stream::new(
            input,
            self.schema(),
            self.mode,
            self.group_by.clone(),
            self.aggr_expr.clone(),
        );
        Ok(Box::pin(stream))
    }
}
#[cfg(test)]
mod test {
    use crate::aggregate::op::avg::avg;
    use crate::aggregate::op::count::count;
    use crate::aggregate::op::max::max;
    use crate::aggregate::op::min::min;
    use crate::aggregate::op::sum::sum;
    use crate::aggregate::CuDFAggregateExec;
    use crate::assert_snapshot;
    use crate::physical::{CuDFLoadExec, CuDFUnloadExec};
    use arrow::array::record_batch;
    use arrow::util::pretty::pretty_format_batches;
    use datafusion::execution::TaskContext;
    use datafusion::physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_expr::AggregateUDF;
    use datafusion_physical_plan::aggregates::{AggregateMode, PhysicalGroupBy};
    use datafusion_physical_plan::expressions::col;
    use datafusion_physical_plan::test::TestMemoryExec;
    use datafusion_physical_plan::ExecutionPlan;
    use futures_util::TryStreamExt;
    use std::error::Error;
    use std::sync::Arc;

    /// Run a GROUP BY aggregation through the full GPU pipeline:
    /// TestMemoryExec -> CuDFLoadExec -> CuDFAggregateExec -> CuDFUnloadExec.
    ///
    /// Sends 3 identical batches (to exercise cross-batch rolling merge) and
    /// groups by column "c".
    async fn run_group_by_test(
        agg_fn: Arc<AggregateUDF>,
        agg_column: &str,
        agg_alias: &str,
    ) -> Result<String, Box<dyn Error>> {
        let batch = record_batch!(
            ("a", Int64, [1, 4, 3]),
            ("b", Float64, [Some(4.0), None, Some(5.0)]),
            ("c", Utf8, ["hello", "hello", "world"]),
            ("d", Float64, [4.0, 5.0, 5.0])
        )
        .expect("created batch");

        let schema = batch.schema();

        let root = TestMemoryExec::try_new(
            &[vec![batch.clone(), batch.clone(), batch]],
            schema.clone(),
            None,
        )?;
        let load = CuDFLoadExec::try_new(Arc::new(root))?;

        let group_by = PhysicalGroupBy::new_single(vec![(col("c", &schema)?, "c".to_string())]);

        let agg = AggregateExprBuilder::new(agg_fn, vec![col(agg_column, &schema)?])
            .schema(schema)
            .alias(agg_alias)
            .build()?;

        let aggregate = CuDFAggregateExec::try_new(
            Arc::new(load),
            AggregateMode::Single,
            group_by,
            vec![Arc::new(agg)],
        )?;

        let unload = CuDFUnloadExec::new(Arc::new(aggregate));

        let task = Arc::new(TaskContext::default());

        let result = unload.execute(0, task)?;
        let batches = result.try_collect::<Vec<_>>().await?;

        let output = pretty_format_batches(&batches)?.to_string();
        Ok(output)
    }

    #[tokio::test]
    async fn test_group_by_sum() -> Result<(), Box<dyn Error>> {
        let output = run_group_by_test(sum(), "a", "SUM(a)").await?;

        // Note: cuDF's SUM always returns Int64 for integer inputs
        assert_snapshot!(output, @r"
        +-------+--------+
        | c     | SUM(a) |
        +-------+--------+
        | hello | 15     |
        | world | 9      |
        +-------+--------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn test_group_by_min() -> Result<(), Box<dyn Error>> {
        let output = run_group_by_test(min(), "a", "MIN(a)").await?;

        assert_snapshot!(output, @r"
        +-------+--------+
        | c     | MIN(a) |
        +-------+--------+
        | hello | 1      |
        | world | 3      |
        +-------+--------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn test_group_by_max() -> Result<(), Box<dyn Error>> {
        let output = run_group_by_test(max(), "a", "MAX(a)").await?;

        assert_snapshot!(output, @r"
        +-------+--------+
        | c     | MAX(a) |
        +-------+--------+
        | hello | 4      |
        | world | 3      |
        +-------+--------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn test_group_by_count() -> Result<(), Box<dyn Error>> {
        let output = run_group_by_test(count(), "a", "COUNT(a)").await?;

        assert_snapshot!(output, @r"
        +-------+----------+
        | c     | COUNT(a) |
        +-------+----------+
        | hello | 6        |
        | world | 3        |
        +-------+----------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn test_group_by_avg() -> Result<(), Box<dyn Error>> {
        let output = run_group_by_test(avg(), "a", "AVG(a)").await?;

        assert_snapshot!(output, @r"
        +-------+--------+
        | c     | AVG(a) |
        +-------+--------+
        | hello | 2.5    |
        | world | 3.0    |
        +-------+--------+
        ");

        Ok(())
    }
}

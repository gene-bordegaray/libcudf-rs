use crate::execution::execute_cudf;
use crate::physical::aggregate::op::udf::CuDFAggregateUDF;
use crate::physical::aggregate::{CuDFAggregationOp, ExprKey, ReusableStateKey};
use arrow_schema::{DataType, SchemaRef};
use datafusion::common::exec_err;
use datafusion::error::Result;
use datafusion::functions_aggregate::count::Count;
use datafusion_expr::AggregateUDF;
use datafusion_physical_plan::PhysicalExpr;
use libcudf_rs::{Aggregation, CuDFColumn, CuDFColumnView, GroupByRequest};
use std::sync::Arc;

pub fn count() -> Arc<AggregateUDF> {
    let udf = CuDFAggregateUDF::new(Arc::new(Count::default()), Arc::new(CuDFCount::default()));
    Arc::new(AggregateUDF::new_from_impl(udf))
}

#[derive(Debug, Default)]
pub struct CuDFCount {
    count_nulls: bool,
}

impl CuDFCount {
    pub(crate) fn count_all() -> Self {
        Self { count_nulls: true }
    }
}

impl CuDFAggregationOp for CuDFCount {
    fn num_state_columns(&self) -> usize {
        1
    }

    fn reusable_state_keys(
        &self,
        args: &[Arc<dyn PhysicalExpr>],
        input_schema: &SchemaRef,
    ) -> Result<Vec<(ReusableStateKey, usize)>> {
        if self.count_nulls {
            return Ok(vec![(ReusableStateKey::CountStar, 0)]);
        }

        Ok(ExprKey::try_from_single_arg(args, input_schema)?
            .map(|key| vec![(ReusableStateKey::Count(key), 0)])
            .unwrap_or_default())
    }

    fn partial_requests(&self, args: &[CuDFColumnView]) -> Result<Vec<GroupByRequest>> {
        if args.len() != 1 {
            return exec_err!("COUNT expects 1 argument, got {}", args.len());
        }

        let aggregation = if self.count_nulls {
            Aggregation::CountAll
        } else {
            Aggregation::Count
        };
        Ok(vec![GroupByRequest::new(args[0].clone()).with(aggregation)])
    }

    fn merge_requests(&self, state_cols: &[CuDFColumnView]) -> Result<Vec<GroupByRequest>> {
        if state_cols.len() != 1 {
            return exec_err!(
                "COUNT merge expects 1 state column, got {}",
                state_cols.len()
            );
        }

        Ok(vec![
            GroupByRequest::new(state_cols[0].clone()).with(Aggregation::Sum)
        ])
    }

    fn normalize_partial_state(&self, mut cols: Vec<CuDFColumn>) -> Result<Vec<CuDFColumn>> {
        // cuDF COUNT returns Int32 -> cast to Int64 to match merge_requests (SUM) output type
        let state = cols.remove(0).into_view();
        let casted = execute_cudf(state.cast(&DataType::Int64))?;
        Ok(vec![casted])
    }

    fn finalize(
        &self,
        state_cols: &[CuDFColumnView],
        _output_type: &DataType,
    ) -> Result<CuDFColumnView> {
        if state_cols.len() != 1 {
            return exec_err!(
                "COUNT finalize expects 1 state column, got {}",
                state_cols.len()
            );
        }
        // cuDF COUNT returns Int32, DataFusion expects Int64
        let casted = execute_cudf(state_cols[0].cast(&DataType::Int64))?;
        Ok(casted.into_view())
    }
}

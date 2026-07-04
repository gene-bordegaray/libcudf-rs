use crate::physical::aggregate::op::udf::CuDFAggregateUDF;
use crate::physical::aggregate::{CuDFAggregationOp, ExprKey, ReusableStateKey};
use arrow_schema::SchemaRef;
use datafusion::common::exec_err;
use datafusion::error::Result;
use datafusion::functions_aggregate::sum::Sum;
use datafusion_expr::AggregateUDF;
use datafusion_physical_plan::PhysicalExpr;
use libcudf_rs::{Aggregation, CuDFColumnView, GroupByRequest};
use std::fmt::Debug;
use std::sync::Arc;

pub fn sum() -> Arc<AggregateUDF> {
    let udf = CuDFAggregateUDF::new(Arc::new(Sum::default()), Arc::new(CuDFSum));
    Arc::new(AggregateUDF::new_from_impl(udf))
}

#[derive(Debug, Default)]
pub struct CuDFSum;

impl CuDFAggregationOp for CuDFSum {
    fn num_state_columns(&self) -> usize {
        1
    }

    fn reusable_state_keys(
        &self,
        args: &[Arc<dyn PhysicalExpr>],
        input_schema: &SchemaRef,
    ) -> Result<Vec<(ReusableStateKey, usize)>> {
        Ok(ExprKey::try_from_single_arg(args, input_schema)?
            .map(|key| vec![(ReusableStateKey::Sum(key), 0)])
            .unwrap_or_default())
    }

    fn partial_requests(&self, args: &[CuDFColumnView]) -> Result<Vec<GroupByRequest>> {
        if args.len() != 1 {
            return exec_err!("SUM expects 1 argument, received: {}", args.len());
        }

        Ok(vec![
            GroupByRequest::new(args[0].clone()).with(Aggregation::Sum)
        ])
    }

    fn merge_requests(&self, state_cols: &[CuDFColumnView]) -> Result<Vec<GroupByRequest>> {
        if state_cols.len() != 1 {
            return exec_err!(
                "SUM merge expects 1 state column, received: {}",
                state_cols.len()
            );
        }

        Ok(vec![
            GroupByRequest::new(state_cols[0].clone()).with(Aggregation::Sum)
        ])
    }

    fn finalize(
        &self,
        state_cols: &[CuDFColumnView],
        _output_type: &arrow_schema::DataType,
    ) -> Result<CuDFColumnView> {
        if state_cols.len() != 1 {
            return exec_err!(
                "SUM finalize expects 1 state column, received: {}",
                state_cols.len()
            );
        }
        Ok(state_cols[0].clone())
    }
}

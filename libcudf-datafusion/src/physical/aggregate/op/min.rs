use crate::physical::aggregate::op::udf::CuDFAggregateUDF;
use crate::physical::aggregate::CuDFAggregationOp;
use datafusion::common::exec_err;
use datafusion::error::Result;
use datafusion::functions_aggregate::min_max::Min;
use datafusion_expr::AggregateUDF;
use libcudf_rs::{Aggregation, CuDFColumnView, GroupByRequest};
use std::sync::Arc;

pub fn min() -> Arc<AggregateUDF> {
    let udf = CuDFAggregateUDF::new(Arc::new(Min::default()), Arc::new(CuDFMin));
    Arc::new(AggregateUDF::new_from_impl(udf))
}

#[derive(Debug, Default)]
pub struct CuDFMin;

impl CuDFAggregationOp for CuDFMin {
    fn num_state_columns(&self) -> usize {
        1
    }

    fn partial_requests(&self, args: &[CuDFColumnView]) -> Result<Vec<GroupByRequest>> {
        if args.len() != 1 {
            return exec_err!("MIN expects 1 argument, got {}", args.len());
        }

        Ok(vec![
            GroupByRequest::new(args[0].clone()).with(Aggregation::Min)
        ])
    }

    fn merge_requests(&self, state_cols: &[CuDFColumnView]) -> Result<Vec<GroupByRequest>> {
        if state_cols.len() != 1 {
            return exec_err!("MIN merge expects 1 state column, got {}", state_cols.len());
        }

        Ok(vec![
            GroupByRequest::new(state_cols[0].clone()).with(Aggregation::Min)
        ])
    }

    fn finalize(
        &self,
        state_cols: &[CuDFColumnView],
        _output_type: &arrow_schema::DataType,
    ) -> Result<CuDFColumnView> {
        if state_cols.len() != 1 {
            return exec_err!(
                "MIN finalize expects 1 state column, got {}",
                state_cols.len()
            );
        }
        Ok(state_cols[0].clone())
    }
}

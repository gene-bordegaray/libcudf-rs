use datafusion::error::Result;
use libcudf_rs::{AggregationRequest, CuDFColumnView};
use std::fmt::Debug;

pub mod avg;
pub mod count;
pub mod max;
pub mod min;
pub mod sum;
pub mod udf;

pub trait CuDFAggregationOp: Debug + Send + Sync {
    fn num_state_columns(&self) -> usize;
    fn partial_requests(&self, args: &[CuDFColumnView]) -> Result<Vec<AggregationRequest>>;
    fn merge_requests(&self, args: &[CuDFColumnView]) -> Result<Vec<AggregationRequest>>;
    fn finalize(&self, args: &[CuDFColumnView]) -> Result<CuDFColumnView>;
}

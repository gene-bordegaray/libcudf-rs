use crate::errors::cudf_to_df;
use datafusion::error::DataFusionError;
use libcudf_rs::{CuDFExecutionContext, CuDFOperation};

/// Execute a root-crate cuDF operation and map errors into DataFusion's error type.
pub(crate) fn execute_cudf<O>(operation: O) -> Result<O::Output, DataFusionError>
where
    O: CuDFOperation,
{
    let ctx = CuDFExecutionContext::try_default_stream().map_err(cudf_to_df)?;
    let output = ctx.execute(operation).map_err(cudf_to_df)?;
    ctx.synchronize().map_err(cudf_to_df)?;
    Ok(output)
}

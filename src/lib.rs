//! Safe, idiomatic Rust bindings for cuDF
//!
//! This crate provides a safe wrapper around the cuDF C++ library,
//! enabling GPU-accelerated dataframe operations in Rust.
//!
//! # Examples
//!
//! ```no_run
//! use libcudf_rs::CuDFTable;
//!
//! // Read a Parquet file
//! let table = CuDFTable::read_parquet("data.parquet").expect("Failed to read Parquet");
//! println!("Loaded table with {} rows and {} columns",
//!          table.num_rows(), table.num_columns());
//!
//! // Write to Parquet
//! table.to_parquet("output.parquet").expect("Failed to write Parquet");
//! ```

mod ast;
mod config;
mod cudf_array;
mod data_type;
mod device_resource;
mod errors;
mod io;
mod operations;
mod storage;
mod stream;
#[cfg(test)]
mod test_activity;

pub use ast::{CuDFAstExpression, CuDFAstNode, CuDFAstOperator, CuDFAstTableReference};
pub use cudf_array::*;
pub use errors::{CuDFError, Result};
pub use io::*;
pub use operations::*;
pub use storage::*;
pub use stream::{CuDFStream, CuDFStreamFlags};

/// Get cuDF version information
///
/// # Examples
///
/// ```
/// use libcudf_rs::version;
///
/// println!("cuDF version: {}", version());
/// ```
pub fn version() -> String {
    libcudf_sys::ffi::get_cudf_version()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let ver = version();
        assert!(!ver.is_empty());
    }
}

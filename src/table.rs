use crate::deferred_operation::deferred;
use crate::execution_policy;
use crate::stream_readiness::{CuDFStreamDependency, CuDFStreamReady, CuDFTableReadiness};
use crate::table_view::CuDFTableView;
use crate::{CuDFColumn, CuDFError, CuDFOperation, CuDFViewStorage};
use cxx::UniquePtr;
use libcudf_sys::ffi;
use std::sync::Arc;

/// A GPU-accelerated table.
///
/// This is the owning Rust wrapper around cuDF's table type. Use
/// [`CuDFTableView`] for non-owning table operations.
pub struct CuDFTable {
    inner: UniquePtr<ffi::Table>,
    stream_readiness: CuDFTableReadiness,
}

impl CuDFTable {
    pub(crate) fn from_inner(inner: UniquePtr<ffi::Table>) -> Self {
        Self {
            inner,
            stream_readiness: CuDFTableReadiness::None,
        }
    }

    pub(crate) fn with_stream_readiness(mut self, dependency: CuDFStreamDependency) -> Self {
        self.stream_readiness = CuDFTableReadiness::whole(dependency);
        self
    }

    pub(crate) fn from_owned_columns(mut columns: Vec<CuDFColumn>) -> Self {
        let dependencies = columns
            .iter()
            .map(|col| col.stream_readiness().cloned())
            .collect::<Vec<_>>();
        let ptrs: Vec<_> = columns
            .iter_mut()
            .map(|col| col.inner_mut().as_mut_ptr())
            .collect();
        let inner = ffi::create_table_from_columns_move(&ptrs);
        let num_columns = inner.num_columns();
        Self {
            inner,
            stream_readiness: CuDFTableReadiness::columns(dependencies, num_columns),
        }
    }

    /// Returns the number of rows in the table.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use libcudf_rs::CuDFTable;
    ///
    /// let table = CuDFTable::default();
    /// assert_eq!(table.num_rows(), 0);
    /// ```
    pub fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }

    /// Returns the number of columns in the table.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use libcudf_rs::CuDFTable;
    ///
    /// let table = CuDFTable::default();
    /// assert_eq!(table.num_columns(), 0);
    /// ```
    pub fn num_columns(&self) -> usize {
        self.inner.num_columns()
    }

    /// Returns true when the table contains no rows.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use libcudf_rs::CuDFTable;
    ///
    /// let table = CuDFTable::default();
    /// assert!(table.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.num_rows() == 0
    }

    /// Returns a non-owning view of this table.
    ///
    /// The returned view borrows from this table and remains valid as long as
    /// the table exists. The table is held behind an [`Arc`] so the view can
    /// keep its storage alive.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use libcudf_rs::CuDFTable;
    /// use std::sync::Arc;
    ///
    /// let table = Arc::new(CuDFTable::default());
    /// let view = table.view();
    /// assert_eq!(view.num_columns(), 0);
    /// ```
    pub fn view(self: Arc<Self>) -> CuDFTableView {
        let readiness = self.stream_readiness.clone();
        let view = self.inner.view();
        let storage: CuDFViewStorage = self;
        CuDFTableView::from_view(view, Some(storage), readiness)
    }

    /// Converts this table into a non-owning view that keeps the table alive.
    ///
    /// This is a convenience wrapper around [`CuDFTable::view`] for callers
    /// that do not already store the table in an [`Arc`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use libcudf_rs::CuDFTable;
    ///
    /// let view = CuDFTable::default().into_view();
    /// assert!(view.is_empty());
    /// ```
    pub fn into_view(self) -> CuDFTableView {
        Arc::new(self).view()
    }

    /// Releases the table into its owned columns.
    ///
    /// This consumes the table structure and returns its columns as independent
    /// [`CuDFColumn`] values.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use libcudf_rs::CuDFTable;
    ///
    /// let columns = CuDFTable::default().into_columns();
    /// assert!(columns.is_empty());
    /// ```
    pub fn into_columns(mut self) -> Vec<CuDFColumn> {
        let mut columns = self.inner.pin_mut().release();
        let mut result = Vec::with_capacity(columns.len());
        for i in 0..columns.len() {
            let col = columns.pin_mut().release(i);
            let mut column = CuDFColumn::from_inner(col);
            if let Some(dependency) = self.stream_readiness.column(i) {
                column = column.with_stream_readiness(dependency);
            }
            result.push(column);
        }
        result
    }

    /// Creates a deferred operation that concatenates table views into one table.
    ///
    /// Execution waits for each input table view before launching cuDF concat on
    /// the target context.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tables have incompatible schemas
    /// - There is insufficient GPU memory
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use libcudf_rs::{CuDFExecutionContext, CuDFTable};
    ///
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let table1 = ctx.execute(CuDFTable::read_parquet("data1.parquet"))?;
    /// let table2 = ctx.execute(CuDFTable::read_parquet("data2.parquet"))?;
    ///
    /// let views = vec![table1.into_view(), table2.into_view()];
    /// let concatenated = ctx.execute(CuDFTable::concat(views))?;
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn concat(views: Vec<CuDFTableView>) -> impl CuDFOperation<Output = Self> {
        deferred(move |ctx| {
            let mut launch = execution_policy::launch(ctx)?;
            let mut inner_views = Vec::with_capacity(views.len());
            for view in views {
                launch.wait_table(&view)?;
                inner_views.push(view.into_inner());
            }
            let inner = ffi::concat_table_views(&inner_views, launch.stream()?, launch.resource())?;
            launch.ready_table(Self::from_inner(inner))
        })
    }
}

impl CuDFStreamReady for CuDFTable {
    fn wait_ready_on_stream(&self, stream: &ffi::CudaStreamView) -> Result<(), CuDFError> {
        self.stream_readiness.wait_on_stream(stream)
    }
}

impl Default for CuDFTable {
    /// Creates an empty table with no rows and no columns.
    fn default() -> Self {
        Self {
            inner: ffi::create_empty_table(),
            stream_readiness: CuDFTableReadiness::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_table() {
        let table = CuDFTable::default();
        assert_eq!(table.num_rows(), 0);
        assert_eq!(table.num_columns(), 0);
        assert!(table.is_empty());
    }

    #[test]
    fn test_parquet_file_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let table = crate::execute_cudf(CuDFTable::read_parquet(
            "testdata/weather/result-000000.parquet",
        ))?;
        let output = std::env::temp_dir().join(format!(
            "libcudf-rs-roundtrip-{}-{}.parquet",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));

        let result = (|| -> Result<(), Box<dyn std::error::Error>> {
            let expected_rows = table.num_rows();
            let expected_columns = table.num_columns();
            crate::execute_cudf(table.into_view().write_parquet(&output))?;
            let roundtrip = crate::execute_cudf(CuDFTable::read_parquet(&output))?;

            assert_eq!(roundtrip.num_rows(), expected_rows);
            assert_eq!(roundtrip.num_columns(), expected_columns);
            Ok(())
        })();

        let _ = std::fs::remove_file(output);
        result
    }

    #[test]
    fn test_read_all_weather_files() {
        for i in 0..3 {
            let filename = format!("testdata/weather/result-{:06}.parquet", i);
            let table = crate::execute_cudf(CuDFTable::read_parquet(&filename))
                .unwrap_or_else(|_| panic!("Failed to read {}", filename));

            assert!(table.num_rows() > 0);
            assert!(table.num_columns() > 0);
        }
    }

    #[test]
    fn test_read_multiple_parquet_files_with_columns() -> Result<(), Box<dyn std::error::Error>> {
        let files = [
            "testdata/weather/result-000000.parquet",
            "testdata/weather/result-000001.parquet",
        ];
        let columns = vec!["MinTemp".to_string(), "MaxTemp".to_string()];

        let projected = crate::execute_cudf(CuDFTable::read_parquet_files(
            crate::CuDFParquetReadOptions {
                paths: &files,
                columns: Some(&columns),
                row_groups: None,
                filter: None,
                allow_mismatched_pq_schemas: false,
                ignore_missing_columns: true,
            },
        ))?
        .table;

        assert!(projected.num_rows() > 0);
        assert_eq!(projected.num_columns(), columns.len());

        Ok(())
    }
}

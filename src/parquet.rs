use crate::config::ensure_pools_configured;
use crate::deferred_operation::deferred;
use crate::execution_policy;
use crate::keep_alive::CuDFKeepAlive;
use crate::{CuDFAstExpression, CuDFError, CuDFOperation, CuDFTable, CuDFTableView};
use arrow_schema::ArrowError;
use cxx::UniquePtr;
use libcudf_sys::ffi;
use std::path::Path;

/// Table and file metadata returned by a cuDF Parquet read.
///
/// cuDF can return fewer columns than the target Arrow schema when projected
/// columns are missing and `ignore_missing_columns` is enabled. The metadata in
/// this result lets callers align returned columns by name and synthesize any
/// missing columns with the correct row count.
pub struct CuDFParquetReadResult {
    /// Table returned by cuDF.
    pub table: CuDFTable,
    /// Top-level column names returned by the cuDF reader, in table order.
    pub column_names: Vec<String>,
    /// Number of rows returned after row group and filter pruning.
    pub num_rows: usize,
}

/// Options for reading Parquet files with cuDF.
///
/// This is the full Parquet reader API. Use [`CuDFTable::read_parquet`] for a
/// single-file read that does not need projection, row-group selection, or
/// metadata.
pub struct CuDFParquetReadOptions<'a, P, S>
where
    P: AsRef<Path>,
    S: AsRef<str>,
{
    /// Parquet file paths to read.
    pub paths: &'a [P],
    /// Optional projected column names.
    pub columns: Option<&'a [S]>,
    /// Optional row group indices for each input file.
    ///
    /// When provided, this slice must contain one entry per path.
    pub row_groups: Option<&'a [Vec<i32>]>,
    /// Optional cuDF AST filter applied by the Parquet reader.
    ///
    /// cuDF stores this as a borrowed AST expression reference, so the filter
    /// must outlive execution of the returned read operation.
    pub filter: Option<&'a CuDFAstExpression>,
    /// Whether cuDF should read matching columns from sources with different schemas.
    pub allow_mismatched_pq_schemas: bool,
    /// Whether cuDF should ignore projected columns missing from an input source.
    ///
    /// Callers that enable this usually need [`CuDFParquetReadResult::column_names`]
    /// and [`CuDFParquetReadResult::num_rows`] to rebuild their requested schema.
    pub ignore_missing_columns: bool,
}

impl CuDFTable {
    /// Create a deferred operation that reads a table from a Parquet file.
    ///
    /// # Errors
    ///
    /// Execution returns an error if:
    /// - The file does not exist or cannot be read
    /// - The Parquet format is invalid
    /// - There is insufficient GPU memory
    /// - The path contains invalid UTF-8
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use libcudf_rs::{CuDFExecutionContext, CuDFTable};
    ///
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let table = ctx.execute(CuDFTable::read_parquet("data.parquet"))?;
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn read_parquet<P: AsRef<Path>>(path: P) -> impl CuDFOperation<Output = Self> {
        deferred(move |ctx| {
            let paths = [path];
            let options = Self::parquet_reader_options(CuDFParquetReadOptions {
                paths: &paths,
                columns: Option::<&[String]>::None,
                row_groups: None,
                filter: None,
                allow_mismatched_pq_schemas: false,
                ignore_missing_columns: true,
            })?;
            Ok(Self::read_parquet_with_options(ctx, options, None)?.table)
        })
    }

    /// Create a deferred operation that reads one or more Parquet files with metadata.
    ///
    /// Use this API when you need projection, row-group selection, AST filter
    /// pushdown, or metadata for schema alignment.
    ///
    /// # Errors
    ///
    /// Execution returns an error if:
    /// - no paths are provided
    /// - a path contains invalid UTF-8
    /// - row-group selections do not match the number of paths
    /// - cuDF cannot read the requested Parquet data
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use libcudf_rs::{CuDFExecutionContext, CuDFParquetReadOptions, CuDFTable};
    ///
    /// let files = ["part-0.parquet", "part-1.parquet"];
    /// let columns = ["id", "value"];
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let read = ctx.execute(CuDFTable::read_parquet_files(CuDFParquetReadOptions {
    ///     paths: &files,
    ///     columns: Some(&columns),
    ///     row_groups: None,
    ///     filter: None,
    ///     allow_mismatched_pq_schemas: false,
    ///     ignore_missing_columns: true,
    /// }))?;
    ///
    /// assert_eq!(read.column_names.len(), read.table.num_columns());
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn read_parquet_files<P, S>(
        read_options: CuDFParquetReadOptions<'_, P, S>,
    ) -> impl CuDFOperation<Output = CuDFParquetReadResult> + '_
    where
        P: AsRef<Path>,
        S: AsRef<str>,
    {
        let filter_keepalive = read_options.filter.cloned();
        deferred(move |ctx| {
            let options = Self::parquet_reader_options(read_options)?;
            Self::read_parquet_with_options(ctx, options, filter_keepalive)
        })
    }

    /// Create a deferred operation that reads Parquet files in cuDF chunks.
    ///
    /// Execution invokes `callback` for each chunk. Returning `Ok(false)` from
    /// the callback stops reading early.
    ///
    /// # Errors
    ///
    /// Execution returns an error if reader option construction fails, cuDF
    /// cannot read a chunk, or `callback` returns an error.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use libcudf_rs::{CuDFExecutionContext, CuDFParquetReadOptions, CuDFTable};
    ///
    /// let files = ["data.parquet"];
    /// let mut rows = 0;
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// ctx.execute(CuDFTable::for_each_parquet_chunk(
    ///     CuDFParquetReadOptions {
    ///         paths: &files,
    ///         columns: Option::<&[String]>::None,
    ///         row_groups: None,
    ///         filter: None,
    ///         allow_mismatched_pq_schemas: false,
    ///         ignore_missing_columns: true,
    ///     },
    ///     256 * 1024 * 1024,
    ///     256 * 1024 * 1024,
    ///     |chunk| {
    ///         rows += chunk.num_rows;
    ///         Ok(true)
    ///     },
    /// ))?;
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn for_each_parquet_chunk<'a, P, S, F>(
        read_options: CuDFParquetReadOptions<'a, P, S>,
        chunk_read_limit_bytes: usize,
        pass_read_limit_bytes: usize,
        callback: F,
    ) -> impl CuDFOperation<Output = ()> + 'a
    where
        P: AsRef<Path> + 'a,
        S: AsRef<str> + 'a,
        F: FnMut(CuDFParquetReadResult) -> Result<bool, CuDFError> + 'a,
    {
        let filter_keepalive = read_options.filter.cloned();
        deferred(move |ctx| {
            let options = Self::parquet_reader_options(read_options)?;
            Self::for_each_parquet_chunk_with_options(
                ctx,
                options,
                chunk_read_limit_bytes,
                pass_read_limit_bytes,
                filter_keepalive,
                callback,
            )
        })
    }

    fn read_parquet_with_options(
        ctx: &crate::CuDFExecutionContext,
        options: UniquePtr<ffi::ParquetReaderOptions>,
        filter_keepalive: Option<CuDFAstExpression>,
    ) -> Result<CuDFParquetReadResult, CuDFError> {
        let mut launch = execution_policy::launch(ctx)?;
        if let Some(filter) = filter_keepalive {
            launch.keep_alive(CuDFKeepAlive::AstExpression {
                _expression: filter,
            });
        }
        let mut result = ffi::read_parquet(
            options
                .as_ref()
                .ok_or(CuDFError::NullHandle("parquet reader options"))?,
            launch.stream()?,
            launch.resource(),
        )?;
        let (inner, column_names, num_rows) = Self::release_parquet_read_result(&mut result);
        let table = launch.ready_table(Self::from_inner(inner))?;
        Ok(CuDFParquetReadResult {
            table,
            column_names,
            num_rows,
        })
    }

    fn parquet_reader_options<P, S>(
        read_options: CuDFParquetReadOptions<'_, P, S>,
    ) -> Result<UniquePtr<ffi::ParquetReaderOptions>, CuDFError>
    where
        P: AsRef<Path>,
        S: AsRef<str>,
    {
        ensure_pools_configured();
        let CuDFParquetReadOptions {
            paths,
            columns,
            row_groups,
            filter,
            allow_mismatched_pq_schemas,
            ignore_missing_columns,
        } = read_options;

        if paths.is_empty() {
            return Err(ArrowError::InvalidArgumentError(
                "at least one Parquet file is required".to_string(),
            )
            .into());
        }

        let path_strings = paths
            .iter()
            .map(|path| {
                path.as_ref()
                    .to_str()
                    .map(ToOwned::to_owned)
                    .ok_or_else(|| {
                        ArrowError::InvalidArgumentError("Path contains invalid UTF-8".to_string())
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let source = ffi::source_info_from_file_paths(path_strings);
        let source = source
            .as_ref()
            .ok_or(CuDFError::NullHandle("source info"))?;
        let mut options = ffi::parquet_reader_options_create(source);
        if let Some(columns) = columns {
            options.pin_mut().set_columns(
                columns
                    .iter()
                    .map(|column| column.as_ref().to_string())
                    .collect(),
            );
        }
        Self::set_parquet_row_groups(&mut options, row_groups, paths.len())?;
        Self::set_parquet_filter(&mut options, filter)?;
        options
            .pin_mut()
            .enable_allow_mismatched_pq_schemas(allow_mismatched_pq_schemas);
        options
            .pin_mut()
            .enable_ignore_missing_columns(ignore_missing_columns);

        Ok(options)
    }

    fn set_parquet_row_groups(
        options: &mut UniquePtr<ffi::ParquetReaderOptions>,
        row_groups: Option<&[Vec<i32>]>,
        path_count: usize,
    ) -> Result<(), CuDFError> {
        let Some(row_groups) = row_groups else {
            return Ok(());
        };
        if row_groups.len() != path_count {
            return Err(ArrowError::InvalidArgumentError(format!(
                "expected row groups for {path_count} Parquet files, got {}",
                row_groups.len()
            ))
            .into());
        }

        let mut indices = Vec::new();
        let mut offsets = Vec::with_capacity(row_groups.len() + 1);
        offsets.push(0);
        for source_groups in row_groups {
            indices.extend(source_groups.iter().copied());
            offsets.push(indices.len());
        }
        options.pin_mut().set_row_groups(indices, offsets);
        Ok(())
    }

    fn set_parquet_filter(
        options: &mut UniquePtr<ffi::ParquetReaderOptions>,
        filter: Option<&CuDFAstExpression>,
    ) -> Result<(), CuDFError> {
        if let Some(filter) = filter {
            options.pin_mut().set_filter(
                filter
                    .inner()
                    .as_ref()
                    .ok_or(CuDFError::NullHandle("AST expression tree"))?,
            )?;
        }
        Ok(())
    }

    fn for_each_parquet_chunk_with_options<F>(
        ctx: &crate::CuDFExecutionContext,
        options: UniquePtr<ffi::ParquetReaderOptions>,
        chunk_read_limit_bytes: usize,
        pass_read_limit_bytes: usize,
        filter_keepalive: Option<CuDFAstExpression>,
        mut callback: F,
    ) -> Result<(), CuDFError>
    where
        F: FnMut(CuDFParquetReadResult) -> Result<bool, CuDFError>,
    {
        let launch = execution_policy::launch(ctx)?;
        let reader = ffi::chunked_parquet_reader_create(
            chunk_read_limit_bytes,
            pass_read_limit_bytes,
            options
                .as_ref()
                .ok_or(CuDFError::NullHandle("parquet reader options"))?,
            launch.stream()?,
            launch.resource(),
        )?;

        while reader.has_next() {
            let chunk =
                Self::read_single_parquet_chunk(&reader, &launch, filter_keepalive.clone())?;
            if !callback(chunk)? {
                break;
            }
        }
        Ok(())
    }

    fn read_single_parquet_chunk(
        reader: &ffi::ChunkedParquetReader,
        launch: &execution_policy::OperationLaunch<'_>,
        filter_keepalive: Option<CuDFAstExpression>,
    ) -> Result<CuDFParquetReadResult, CuDFError> {
        let mut result = reader.read_chunk()?;
        let (inner, column_names, num_rows) = Self::release_parquet_read_result(&mut result);
        let keepalives = filter_keepalive
            .map(|filter| CuDFKeepAlive::AstExpression {
                _expression: filter,
            })
            .into_iter()
            .collect();
        let dependency = launch.record_stream_dependency(keepalives)?;
        Ok(CuDFParquetReadResult {
            table: Self::from_inner(inner).with_stream_readiness(dependency),
            column_names,
            num_rows,
        })
    }

    fn release_parquet_read_result(
        result: &mut UniquePtr<ffi::TableWithMetadata>,
    ) -> (UniquePtr<ffi::Table>, Vec<String>, usize) {
        let column_names = (0..result.schema_info_count())
            .map(|index| result.schema_info_name(index))
            .collect();
        let inner = result.pin_mut().release_table();
        let num_rows = inner.num_rows();
        (inner, column_names, num_rows)
    }
}

impl CuDFTableView {
    /// Create a deferred operation that writes this table view to a Parquet file.
    ///
    /// # Errors
    ///
    /// Execution returns an error if:
    /// - The file cannot be created or written
    /// - There is insufficient GPU memory
    /// - The path contains invalid UTF-8
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use libcudf_rs::{CuDFExecutionContext, CuDFTable};
    /// # let table: CuDFTable = todo!();
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let view = table.into_view();
    /// ctx.execute(view.write_parquet("output.parquet"))?;
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn write_parquet<'a, P: AsRef<Path> + 'a>(
        &'a self,
        path: P,
    ) -> impl CuDFOperation<Output = ()> + 'a {
        deferred(move |ctx| {
            let path_str = path.as_ref().to_str().ok_or_else(|| {
                ArrowError::InvalidArgumentError("Path contains invalid UTF-8".to_string())
            })?;

            let sink = ffi::sink_info_from_file_path(path_str);
            let options = ffi::parquet_writer_options_create(
                sink.as_ref().ok_or(CuDFError::NullHandle("sink info"))?,
                self.inner(),
            );
            let mut launch = execution_policy::launch(ctx)?;
            launch.wait_table(self)?;
            let _metadata = ffi::write_parquet(
                options
                    .as_ref()
                    .ok_or(CuDFError::NullHandle("parquet writer options"))?,
                launch.stream()?,
            )?;
            launch.stream()?.synchronize()?;
            Ok(())
        })
    }
}

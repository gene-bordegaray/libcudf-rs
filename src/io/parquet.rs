use crate::device_resource::resource_ref;
use crate::stream::stream_ref;
use crate::{CuDFAstExpression, CuDFError, CuDFTable};
use arrow_schema::ArrowError;
use cxx::UniquePtr;
use libcudf_sys::ffi;
use std::path::Path;

/// Result of reading Parquet with cuDF, including schema metadata.
pub struct CuDFParquetReadResult {
    /// Table returned by cuDF.
    pub table: CuDFTable,
    /// Top-level column names returned by the reader, in table order.
    pub column_names: Vec<String>,
    /// Number of rows returned after row-group and filter pruning.
    pub num_rows: usize,
}

/// Options for reading one or more Parquet files with cuDF.
pub struct CuDFParquetReadOptions<'a, P, S>
where
    P: AsRef<Path>,
    S: AsRef<str>,
{
    /// Parquet file paths to read.
    pub paths: &'a [P],
    /// Optional projected column names.
    pub columns: Option<&'a [S]>,
    /// Optional row-group indices per input file.
    pub row_groups: Option<&'a [Vec<i32>]>,
    /// Optional cuDF AST filter applied by the Parquet reader.
    pub filter: Option<&'a CuDFAstExpression>,
    /// Whether matching columns may be read from mismatched schemas.
    pub allow_mismatched_pq_schemas: bool,
    /// Whether projected columns missing from an input source are ignored.
    pub ignore_missing_columns: bool,
}

impl CuDFTable {
    /// Read a table from a Parquet file.
    pub fn read_parquet<P: AsRef<Path>>(path: P) -> Result<Self, CuDFError> {
        Self::read_parquet_files(std::slice::from_ref(&path))
    }

    /// Read a table from one or more Parquet files.
    pub fn read_parquet_files<P: AsRef<Path>>(paths: &[P]) -> Result<Self, CuDFError> {
        Self::read_parquet_files_with_columns(paths, Option::<&[String]>::None)
    }

    /// Read selected columns from one or more Parquet files.
    pub fn read_parquet_files_with_columns<P, S>(
        paths: &[P],
        columns: Option<&[S]>,
    ) -> Result<Self, CuDFError>
    where
        P: AsRef<Path>,
        S: AsRef<str>,
    {
        Ok(Self::read_parquet_with_options(CuDFParquetReadOptions {
            paths,
            columns,
            row_groups: None,
            filter: None,
            allow_mismatched_pq_schemas: false,
            ignore_missing_columns: true,
        })?
        .table)
    }

    /// Read Parquet files and return cuDF schema metadata.
    pub fn read_parquet_with_options<P, S>(
        read_options: CuDFParquetReadOptions<'_, P, S>,
    ) -> Result<CuDFParquetReadResult, CuDFError>
    where
        P: AsRef<Path>,
        S: AsRef<str>,
    {
        let options = parquet_reader_options(read_options)?;
        read_parquet_options_with_metadata(options)
    }

    /// Read all Parquet chunks and collect their schema metadata.
    pub fn read_parquet_files_chunked<P, S>(
        read_options: CuDFParquetReadOptions<'_, P, S>,
        chunk_read_limit: usize,
        pass_read_limit: usize,
    ) -> Result<Vec<CuDFParquetReadResult>, CuDFError>
    where
        P: AsRef<Path>,
        S: AsRef<str>,
    {
        let mut chunks = Vec::new();
        Self::for_each_parquet_chunk(read_options, chunk_read_limit, pass_read_limit, |chunk| {
            chunks.push(chunk);
            Ok(true)
        })?;
        Ok(chunks)
    }

    /// Read Parquet chunks until `callback` returns `false`.
    pub fn for_each_parquet_chunk<P, S, F>(
        read_options: CuDFParquetReadOptions<'_, P, S>,
        chunk_read_limit: usize,
        pass_read_limit: usize,
        callback: F,
    ) -> Result<(), CuDFError>
    where
        P: AsRef<Path>,
        S: AsRef<str>,
        F: FnMut(CuDFParquetReadResult) -> Result<bool, CuDFError>,
    {
        let options = parquet_reader_options(read_options)?;
        for_each_parquet_options_chunk(options, chunk_read_limit, pass_read_limit, callback)
    }

    /// Write this table to a Parquet file.
    pub fn to_parquet<P: AsRef<Path>>(&self, path: P) -> Result<(), CuDFError> {
        let path = path.as_ref().to_str().ok_or_else(|| {
            ArrowError::InvalidArgumentError("Path contains invalid UTF-8".into())
        })?;
        let sink = ffi::sink_info_from_file_path(path);
        let sink = sink
            .as_ref()
            .ok_or(CuDFError::NullHandle("Parquet sink info"))?;
        let options = ffi::parquet_writer_options_create(sink, self.view_inner())?;
        let options = options
            .as_ref()
            .ok_or(CuDFError::NullHandle("Parquet writer options"))?;
        let stream = ffi::get_default_stream();
        let _metadata = ffi::write_parquet(options, stream_ref(&stream)?)?;
        Ok(())
    }
}

fn parquet_reader_options<P, S>(
    read_options: CuDFParquetReadOptions<'_, P, S>,
) -> Result<UniquePtr<ffi::ParquetReaderOptions>, CuDFError>
where
    P: AsRef<Path>,
    S: AsRef<str>,
{
    crate::config::ensure_pools_configured()?;
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
            "at least one Parquet file is required".into(),
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
                    ArrowError::InvalidArgumentError("Path contains invalid UTF-8".into())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let source = ffi::source_info_from_file_paths(path_strings);
    let source = source
        .as_ref()
        .ok_or(CuDFError::NullHandle("Parquet source info"))?;
    let mut options = ffi::parquet_reader_options_create(source);
    if options.is_null() {
        return Err(CuDFError::NullHandle("Parquet reader options"));
    }
    set_parquet_columns(&mut options, columns);
    set_parquet_row_groups(&mut options, row_groups, paths.len())?;
    set_parquet_filter(&mut options, filter)?;
    options
        .pin_mut()
        .enable_allow_mismatched_pq_schemas(allow_mismatched_pq_schemas);
    options
        .pin_mut()
        .enable_ignore_missing_columns(ignore_missing_columns);
    Ok(options)
}

fn set_parquet_columns<S>(options: &mut UniquePtr<ffi::ParquetReaderOptions>, columns: Option<&[S]>)
where
    S: AsRef<str>,
{
    if let Some(columns) = columns {
        options.pin_mut().set_columns(
            columns
                .iter()
                .map(|column| column.as_ref().to_owned())
                .collect(),
        );
    }
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
        let filter = filter
            .inner()
            .as_ref()
            .ok_or(CuDFError::NullHandle("AST expression tree"))?;
        options.pin_mut().set_filter(filter)?;
    }
    Ok(())
}

fn read_parquet_options_with_metadata(
    options: UniquePtr<ffi::ParquetReaderOptions>,
) -> Result<CuDFParquetReadResult, CuDFError> {
    let stream = ffi::get_default_stream();
    let resource = ffi::get_current_device_resource_ref();
    let options = options
        .as_ref()
        .ok_or(CuDFError::NullHandle("Parquet reader options"))?;
    let mut result = ffi::read_parquet(options, stream_ref(&stream)?, resource_ref(&resource)?)?;
    parquet_read_result_from_metadata(&mut result)
}

fn for_each_parquet_options_chunk<F>(
    options: UniquePtr<ffi::ParquetReaderOptions>,
    chunk_read_limit: usize,
    pass_read_limit: usize,
    mut callback: F,
) -> Result<(), CuDFError>
where
    F: FnMut(CuDFParquetReadResult) -> Result<bool, CuDFError>,
{
    let stream = ffi::get_default_stream();
    let resource = ffi::get_current_device_resource_ref();
    let reader = ffi::chunked_parquet_reader_create(
        chunk_read_limit,
        pass_read_limit,
        options
            .as_ref()
            .ok_or(CuDFError::NullHandle("Parquet reader options"))?,
        stream_ref(&stream)?,
        resource_ref(&resource)?,
    )?;

    while reader.has_next()? {
        let mut result = reader.read_chunk()?;
        if !callback(parquet_read_result_from_metadata(&mut result)?)? {
            break;
        }
    }
    Ok(())
}

fn parquet_read_result_from_metadata(
    result: &mut UniquePtr<ffi::TableWithMetadata>,
) -> Result<CuDFParquetReadResult, CuDFError> {
    let metadata = result.pin_mut().release_metadata()?;
    let metadata = metadata
        .as_ref()
        .ok_or(CuDFError::NullHandle("Parquet table metadata"))?;
    let mut column_names = Vec::with_capacity(metadata.schema_info_len());
    for index in 0..metadata.schema_info_len() {
        let info = metadata.schema_info(index)?;
        let info = info
            .as_ref()
            .ok_or(CuDFError::NullHandle("Parquet column metadata"))?;
        column_names.push(info.name());
    }
    let table = CuDFTable::try_from_inner(result.pin_mut().release_table()?)?;
    let num_rows = table.num_rows();
    Ok(CuDFParquetReadResult {
        table,
        column_names,
        num_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parquet_arrow_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let batch = CuDFTable::read_parquet("testdata/weather/result-000000.parquet")?
            .into_view()
            .to_arrow_host()?;
        let expected_rows = batch.num_rows();
        let expected_columns = batch.num_columns();
        let table = CuDFTable::try_from_arrow_host(batch)?;
        assert_eq!(table.num_rows(), expected_rows);
        assert_eq!(table.num_columns(), expected_columns);
        Ok(())
    }

    #[test]
    fn parquet_file_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let table = CuDFTable::read_parquet("testdata/weather/result-000000.parquet")?;
        let output = std::env::temp_dir().join(format!(
            "libcudf-rs-roundtrip-{}-{}.parquet",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        let result = (|| -> Result<(), Box<dyn std::error::Error>> {
            table.to_parquet(&output)?;
            let roundtrip = CuDFTable::read_parquet(&output)?;
            assert_eq!(roundtrip.num_rows(), table.num_rows());
            assert_eq!(roundtrip.num_columns(), table.num_columns());
            Ok(())
        })();
        let _ = std::fs::remove_file(output);
        result
    }

    #[test]
    fn all_weather_files_are_readable() -> Result<(), Box<dyn std::error::Error>> {
        for index in 0..3 {
            let path = format!("testdata/weather/result-{index:06}.parquet");
            let table = CuDFTable::read_parquet(path)?;
            assert!(table.num_rows() > 0);
            assert!(table.num_columns() > 0);
        }
        Ok(())
    }

    #[test]
    fn multiple_files_support_projection() -> Result<(), Box<dyn std::error::Error>> {
        let files = [
            "testdata/weather/result-000000.parquet",
            "testdata/weather/result-000001.parquet",
        ];
        let columns = ["MinTemp", "MaxTemp"];
        let table = CuDFTable::read_parquet_files_with_columns(&files, Some(&columns))?;
        assert!(table.num_rows() > 0);
        assert_eq!(table.num_columns(), columns.len());
        Ok(())
    }
}

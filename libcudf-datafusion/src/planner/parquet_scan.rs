use crate::physical::{CuDFParquetScanConfig, CuDFParquetScanExec};
use crate::planner::CuDFConfig;
use datafusion::common::Result;
use datafusion::datasource::physical_plan::{FileScanConfig, FileSource, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::physical_expr::expressions::Column;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::ExecutionPlan;
use std::path::PathBuf;
use std::sync::Arc;

pub(crate) fn try_as_cudf_parquet_scan(
    plan: &Arc<dyn ExecutionPlan>,
    cudf_config: &CuDFConfig,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    if !cudf_config.parquet_scan {
        return Ok(None);
    }

    let plan = match plan.as_any().downcast_ref::<CoalescePartitionsExec>() {
        Some(coalesce) => coalesce.input(),
        None => plan,
    };

    let Some(data_source) = plan.as_any().downcast_ref::<DataSourceExec>() else {
        return Ok(None);
    };

    let Some((file_scan, source)) = data_source.downcast_to_file_source::<ParquetSource>() else {
        return Ok(None);
    };

    if source.table_schema().table_partition_cols().is_empty()
        && file_scan.limit.is_none()
        && !file_scan.preserve_order
        && file_scan.output_ordering.is_empty()
    {
        let Some(files) = local_file_paths(file_scan) else {
            return Ok(None);
        };
        let Some(projection) = projected_file_columns(source) else {
            return Ok(None);
        };

        let scan_config =
            CuDFParquetScanConfig::new(files, Arc::clone(source.table_schema().file_schema()))
                .with_projection(projection)
                .with_files_per_batch(cudf_config.parquet_scan_files_per_batch);

        return Ok(Some(Arc::new(CuDFParquetScanExec::try_new(scan_config)?)));
    }

    Ok(None)
}

fn local_file_paths(file_scan: &FileScanConfig) -> Option<Vec<PathBuf>> {
    if file_scan.object_store_url.as_str() != "file:///" {
        return None;
    }

    let mut files = Vec::new();
    for group in &file_scan.file_groups {
        for file in group.iter() {
            if !file.partition_values.is_empty() {
                return None;
            }
            if let Some(range) = &file.range {
                if range.start != 0 || range.end < 0 || range.end as u64 != file.object_meta.size {
                    return None;
                }
            }

            let location = file.object_meta.location.as_ref();
            let path = if location.starts_with('/') {
                PathBuf::from(location)
            } else {
                PathBuf::from("/").join(location)
            };
            files.push(path);
        }
    }

    (!files.is_empty()).then_some(files)
}

fn projected_file_columns(source: &ParquetSource) -> Option<Option<Vec<usize>>> {
    let file_schema = source.table_schema().file_schema();
    let Some(projection) = source.projection() else {
        return Some(None);
    };

    let mut indices = Vec::new();
    for expr in projection.iter() {
        let column = expr.expr.as_any().downcast_ref::<Column>()?;
        let field = file_schema.fields().get(column.index())?;
        if expr.alias != field.name().as_str() || column.name() != field.name() {
            return None;
        }
        indices.push(column.index());
    }

    let is_full_file_projection = indices.len() == file_schema.fields().len()
        && indices.iter().copied().eq(0..file_schema.fields().len());

    if is_full_file_projection {
        Some(None)
    } else {
        Some(Some(indices))
    }
}

#[cfg(test)]
mod tests {
    use super::{projected_file_columns, try_as_cudf_parquet_scan};
    use crate::physical::CuDFParquetScanExec;
    use crate::planner::CuDFConfig;
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use datafusion::datasource::listing::PartitionedFile;
    use datafusion::datasource::physical_plan::{FileScanConfigBuilder, FileSource, ParquetSource};
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::execution::object_store::ObjectStoreUrl;
    use datafusion::physical_expr::expressions::{CastExpr, Column};
    use datafusion::physical_expr::projection::{ProjectionExpr, ProjectionExprs};
    use datafusion_physical_plan::ExecutionPlan;
    use std::sync::Arc;

    #[test]
    fn accepts_full_file_ranges() -> Result<(), Box<dyn std::error::Error>> {
        let plan =
            parquet_data_source(PartitionedFile::new("/tmp/file.parquet", 100).with_range(0, 100))?;

        let scan = try_as_cudf_parquet_scan(&plan, &CuDFConfig::default())?
            .expect("full file ranges should be accepted");

        assert!(scan.as_any().is::<CuDFParquetScanExec>());
        Ok(())
    }

    #[test]
    fn rejects_partial_file_ranges() -> Result<(), Box<dyn std::error::Error>> {
        let plan =
            parquet_data_source(PartitionedFile::new("/tmp/file.parquet", 100).with_range(25, 75))?;

        let scan = try_as_cudf_parquet_scan(&plan, &CuDFConfig::default())?;

        assert!(scan.is_none());
        Ok(())
    }

    #[test]
    fn projected_file_columns_accepts_simple_columns() -> Result<(), Box<dyn std::error::Error>> {
        let source = projected_source(ProjectionExprs::from_indices(&[1, 0], &file_schema()))?;
        let source = source
            .as_any()
            .downcast_ref::<ParquetSource>()
            .expect("projected source should remain parquet");

        assert_eq!(projected_file_columns(source), Some(Some(vec![1, 0])));
        Ok(())
    }

    #[test]
    fn projected_file_columns_rejects_aliases() -> Result<(), Box<dyn std::error::Error>> {
        let projection = ProjectionExprs::new([ProjectionExpr::new(
            Arc::new(Column::new("a", 0)),
            "renamed",
        )]);
        let source = projected_source(projection)?;
        let source = source
            .as_any()
            .downcast_ref::<ParquetSource>()
            .expect("projected source should remain parquet");

        assert_eq!(projected_file_columns(source), None);
        Ok(())
    }

    #[test]
    fn projected_file_columns_rejects_expressions() -> Result<(), Box<dyn std::error::Error>> {
        let projection = ProjectionExprs::new([ProjectionExpr::new(
            Arc::new(CastExpr::new(
                Arc::new(Column::new("a", 0)),
                DataType::Int64,
                None,
            )),
            "a",
        )]);
        let source = projected_source(projection)?;
        let source = source
            .as_any()
            .downcast_ref::<ParquetSource>()
            .expect("projected source should remain parquet");

        assert_eq!(projected_file_columns(source), None);
        Ok(())
    }

    fn parquet_data_source(
        file: PartitionedFile,
    ) -> Result<Arc<dyn ExecutionPlan>, Box<dyn std::error::Error>> {
        let source = Arc::new(ParquetSource::new(file_schema()));
        let config = FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), source)
            .with_projection_indices(Some(vec![0, 1]))?
            .with_file(file)
            .build();

        Ok(DataSourceExec::from_data_source(config))
    }

    fn projected_source(
        projection: ProjectionExprs,
    ) -> Result<Arc<dyn FileSource>, Box<dyn std::error::Error>> {
        let source = ParquetSource::new(file_schema());
        let source = source
            .try_pushdown_projection(&projection)?
            .expect("parquet projection pushdown should be supported");
        Ok(source)
    }

    fn file_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]))
    }
}

use crate::errors::cudf_to_df;
use crate::metrics::CuDFBaselineMetrics;
use arrow::array::Array;
use arrow_schema::{Schema, SchemaRef};
use datafusion::common::{assert_eq_or_internal_err, plan_err};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet, Time,
};
use datafusion_physical_plan::stream::RecordBatchReceiverStream;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use libcudf_rs::{synchronize_default_stream, CuDFTable};
use std::any::Any;
use std::fmt::Formatter;
use std::path::PathBuf;
use std::sync::Arc;

const DEFAULT_FILES_PER_BATCH: usize = 8;

#[derive(Debug)]
pub struct CuDFParquetScanExec {
    files: Arc<[PathBuf]>,
    projected_columns: Option<Arc<[String]>>,
    files_per_batch: usize,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl CuDFParquetScanExec {
    /// Create a cuDF Parquet scan that reads all columns from the provided files.
    pub fn try_new(files: Vec<PathBuf>, schema: SchemaRef) -> datafusion::common::Result<Self> {
        Self::try_new_with_projection(files, schema, None)
    }

    /// Create a cuDF Parquet scan with an optional projection by schema index.
    pub fn try_new_with_projection(
        files: Vec<PathBuf>,
        file_schema: SchemaRef,
        projection: Option<Vec<usize>>,
    ) -> datafusion::common::Result<Self> {
        Self::try_new_with_projection_and_files_per_batch(
            files,
            file_schema,
            projection,
            DEFAULT_FILES_PER_BATCH,
        )
    }

    /// Create a cuDF Parquet scan with projection and bounded file grouping.
    pub fn try_new_with_projection_and_files_per_batch(
        files: Vec<PathBuf>,
        file_schema: SchemaRef,
        projection: Option<Vec<usize>>,
        files_per_batch: usize,
    ) -> datafusion::common::Result<Self> {
        if files.is_empty() {
            return plan_err!("CuDFParquetScanExec requires at least one parquet file");
        }
        if files_per_batch == 0 {
            return plan_err!("CuDFParquetScanExec files_per_batch must be greater than zero");
        }

        let (projected_columns, output_schema) = match projection {
            Some(projection) => {
                let mut columns = Vec::with_capacity(projection.len());
                let mut fields = Vec::with_capacity(projection.len());
                for index in projection {
                    let Some(field) = file_schema.fields().get(index) else {
                        return plan_err!(
                            "CuDFParquetScanExec projection index {index} out of bounds for schema with {} fields",
                            file_schema.fields().len()
                        );
                    };
                    columns.push(field.name().clone());
                    fields.push(Arc::clone(field));
                }
                (Some(Arc::from(columns)), Arc::new(Schema::new(fields)))
            }
            None => (None, file_schema),
        };

        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(super::cudf_schema_compatibility_map(output_schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        let files_per_batch = files_per_batch.min(files.len());

        Ok(Self {
            files: files.into(),
            projected_columns,
            files_per_batch,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl DisplayAs for CuDFParquetScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "CuDFParquetScanExec: files={}, files_per_batch={}, projected_columns={}",
            self.files.len(),
            self.files_per_batch,
            self.projected_columns
                .as_ref()
                .map_or(0, |columns| columns.len())
        )
    }
}

impl ExecutionPlan for CuDFParquetScanExec {
    fn name(&self) -> &str {
        "CuDFParquetScanExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return plan_err!(
                "CuDFParquetScanExec expects no children, {} were provided",
                children.len()
            );
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> datafusion::common::Result<SendableRecordBatchStream> {
        assert_eq_or_internal_err!(
            partition,
            0,
            "CuDFParquetScanExec invalid partition {partition}"
        );

        let schema = self.schema();
        let files = Arc::clone(&self.files);
        let metrics = CuDFParquetScanMetrics::new(&self.metrics, partition)
            .with_scan_config(self.projected_columns.clone(), self.files_per_batch);
        let mut builder = RecordBatchReceiverStream::builder(Arc::clone(&schema), 1);
        let output = builder.tx();

        builder.spawn_blocking(move || {
            let compute_timer = metrics.baseline.elapsed_compute().timer();

            for chunk in files.chunks(metrics.files_per_batch) {
                metrics.record_files(chunk);

                let read_timer = metrics.read_time.timer();
                let table = match metrics.projected_columns.as_deref() {
                    Some(columns) => {
                        CuDFTable::from_parquet_files_with_columns(chunk, Some(columns))
                            .map_err(cudf_to_df)?
                    }
                    None => CuDFTable::from_parquet_files(chunk).map_err(cudf_to_df)?,
                };
                read_timer.done();

                let sync_timer = metrics.sync_time.timer();
                synchronize_default_stream().map_err(cudf_to_df)?;
                sync_timer.done();

                let output_batch_timer = metrics.output_batch_time.timer();
                let num_rows = table.num_rows();
                let cudf_cols: Vec<Arc<dyn Array>> = table
                    .into_columns()
                    .into_iter()
                    .map(|c| Arc::new(c.into_view()) as Arc<dyn Array>)
                    .collect();
                let batch = libcudf_rs::record_batch_with_schema(cudf_cols, &schema, num_rows)?;
                output_batch_timer.done();

                metrics.baseline.record_output(&batch);
                if output.blocking_send(Ok(batch)).is_err() {
                    break;
                }
            }

            compute_timer.done();
            Ok(())
        });

        Ok(builder.build())
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

#[derive(Clone)]
struct CuDFParquetScanMetrics {
    baseline: CuDFBaselineMetrics,
    projected_columns: Option<Arc<[String]>>,
    files_per_batch: usize,
    read_time: Time,
    sync_time: Time,
    output_batch_time: Time,
    files: Count,
    file_bytes: Count,
}

impl CuDFParquetScanMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            baseline: CuDFBaselineMetrics::new(metrics, partition),
            projected_columns: None,
            files_per_batch: 1,
            read_time: MetricBuilder::new(metrics).subset_time("read_time", partition),
            sync_time: MetricBuilder::new(metrics).subset_time("sync_time", partition),
            output_batch_time: MetricBuilder::new(metrics)
                .subset_time("output_batch_time", partition),
            files: MetricBuilder::new(metrics).counter("files", partition),
            file_bytes: MetricBuilder::new(metrics).counter("file_bytes", partition),
        }
    }

    fn with_scan_config(
        mut self,
        projected_columns: Option<Arc<[String]>>,
        files_per_batch: usize,
    ) -> Self {
        self.projected_columns = projected_columns;
        self.files_per_batch = files_per_batch.max(1);
        self
    }

    fn record_files(&self, paths: &[PathBuf]) {
        self.files.add(paths.len());
        for path in paths {
            if let Ok(metadata) = std::fs::metadata(path) {
                self.file_bytes.add(metadata.len() as usize);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CuDFParquetScanExec;
    use datafusion::execution::TaskContext;
    use datafusion_physical_plan::{execute_stream, ExecutionPlan};
    use futures_util::TryStreamExt;
    use libcudf_rs::is_cudf_array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;
    use std::path::PathBuf;
    use std::sync::Arc;

    #[tokio::test]
    async fn reads_parquet_as_cudf_batches() -> Result<(), Box<dyn std::error::Error>> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../testdata/weather/result-000000.parquet");
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path)?)?;
        let schema = reader.schema().clone();
        let expected_rows = reader.metadata().file_metadata().num_rows() as usize;
        let expected_columns = schema.fields().len();

        let plan: Arc<dyn ExecutionPlan> =
            Arc::new(CuDFParquetScanExec::try_new(vec![path], schema)?);
        let batches = execute_stream(plan, Arc::new(TaskContext::default()))?
            .try_collect::<Vec<_>>()
            .await?;

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), expected_rows);
        assert_eq!(batches[0].num_columns(), expected_columns);
        assert!(batches[0]
            .columns()
            .iter()
            .all(|column| is_cudf_array(column.as_ref())));

        Ok(())
    }

    #[tokio::test]
    async fn reads_projected_parquet_columns() -> Result<(), Box<dyn std::error::Error>> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../testdata/weather/result-000000.parquet");
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path)?)?;
        let schema = reader.schema().clone();
        let expected_rows = reader.metadata().file_metadata().num_rows() as usize;

        let plan: Arc<dyn ExecutionPlan> = Arc::new(CuDFParquetScanExec::try_new_with_projection(
            vec![path],
            schema,
            Some(vec![0, 1]),
        )?);
        let batches = execute_stream(plan, Arc::new(TaskContext::default()))?
            .try_collect::<Vec<_>>()
            .await?;

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), expected_rows);
        assert_eq!(batches[0].num_columns(), 2);
        assert!(batches[0]
            .columns()
            .iter()
            .all(|column| is_cudf_array(column.as_ref())));

        Ok(())
    }
}

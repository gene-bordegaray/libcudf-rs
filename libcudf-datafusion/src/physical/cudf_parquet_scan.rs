use crate::errors::cudf_to_df;
use crate::metrics::CuDFBaselineMetrics;
use arrow::array::Array;
use arrow_schema::SchemaRef;
use datafusion::common::{assert_eq_or_internal_err, plan_err};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet, Time,
};
use datafusion_physical_plan::stream::RecordBatchReceiverStream;
use datafusion_physical_plan::{
    project_schema, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};
use libcudf_rs::{cast, synchronize_default_stream, CuDFTable};
use std::any::Any;
use std::fmt::Formatter;
use std::path::PathBuf;
use std::sync::Arc;

const DEFAULT_FILES_PER_BATCH: usize = 8;

/// Configuration for a cuDF-backed Parquet scan.
#[derive(Debug, Clone)]
pub struct CuDFParquetScanConfig {
    files: Vec<PathBuf>,
    file_schema: SchemaRef,
    projection: Option<Vec<usize>>,
    files_per_batch: usize,
}

impl CuDFParquetScanConfig {
    /// Create a scan config that reads every column from the provided files.
    pub fn new(files: Vec<PathBuf>, file_schema: SchemaRef) -> Self {
        Self {
            files,
            file_schema,
            projection: None,
            files_per_batch: DEFAULT_FILES_PER_BATCH,
        }
    }

    /// Set an optional projection using file schema column indices.
    pub fn with_projection(mut self, projection: Option<Vec<usize>>) -> Self {
        self.projection = projection;
        self
    }

    /// Set the maximum number of files per cuDF read.
    pub fn with_files_per_batch(mut self, files_per_batch: usize) -> Self {
        self.files_per_batch = files_per_batch;
        self
    }
}

#[derive(Debug, Clone)]
struct CuDFParquetReadPlan {
    batches: Arc<[CuDFParquetFileBatch]>,
    projected_columns: Option<Arc<[String]>>,
    file_count: usize,
    files_per_batch: usize,
}

#[derive(Debug, Clone)]
struct CuDFParquetFileBatch {
    files: Arc<[PathBuf]>,
    file_bytes: usize,
}

/// DataFusion execution plan that scans Parquet files directly with cuDF.
#[derive(Debug)]
pub struct CuDFParquetScanExec {
    read_plan: CuDFParquetReadPlan,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl CuDFParquetScanExec {
    /// Create a cuDF Parquet scan from a validated scan config.
    pub fn try_new(config: CuDFParquetScanConfig) -> datafusion::common::Result<Self> {
        let CuDFParquetScanConfig {
            files,
            file_schema,
            projection,
            files_per_batch,
        } = config;

        if files.is_empty() {
            return plan_err!("CuDFParquetScanExec requires at least one parquet file");
        }
        if files_per_batch == 0 {
            return plan_err!("CuDFParquetScanExec files_per_batch must be greater than zero");
        }

        let output_schema = project_schema(&file_schema, projection.as_ref())?;
        let projected_columns = match projection.as_ref() {
            Some(projection) => {
                let mut columns = Vec::with_capacity(projection.len());
                for &index in projection {
                    let Some(field) = file_schema.fields().get(index) else {
                        return plan_err!(
                            "CuDFParquetScanExec projection index {index} out of bounds for schema with {} fields",
                            file_schema.fields().len()
                        );
                    };
                    columns.push(field.name().clone());
                }
                Some(Arc::from(columns))
            }
            None => None,
        };

        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(super::cudf_schema_compatibility_map(output_schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        let files_per_batch = files_per_batch.min(files.len());
        let read_plan = CuDFParquetReadPlan::new(files, projected_columns, files_per_batch);

        Ok(Self {
            read_plan,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl CuDFParquetReadPlan {
    fn new(
        files: Vec<PathBuf>,
        projected_columns: Option<Arc<[String]>>,
        files_per_batch: usize,
    ) -> Self {
        let file_count = files.len();
        let batches = files
            .chunks(files_per_batch)
            .map(CuDFParquetFileBatch::new)
            .collect::<Vec<_>>()
            .into();

        Self {
            batches,
            projected_columns,
            file_count,
            files_per_batch,
        }
    }
}

impl CuDFParquetFileBatch {
    fn new(files: &[PathBuf]) -> Self {
        let file_bytes = files
            .iter()
            .filter_map(|path| std::fs::metadata(path).ok())
            .map(|metadata| metadata.len() as usize)
            .sum();

        Self {
            files: Arc::from(files.to_vec()),
            file_bytes,
        }
    }
}

impl DisplayAs for CuDFParquetScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "CuDFParquetScanExec: files={}, batches={}, files_per_batch={}, projected_columns={}",
            self.read_plan.file_count,
            self.read_plan.batches.len(),
            self.read_plan.files_per_batch,
            self.read_plan
                .projected_columns
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
        let read_plan = self.read_plan.clone();
        let metrics = CuDFParquetScanMetrics::new(&self.metrics, partition);
        let mut builder = RecordBatchReceiverStream::builder(Arc::clone(&schema), 1);
        let output = builder.tx();

        builder.spawn_blocking(move || {
            let compute_timer = metrics.baseline.elapsed_compute().timer();

            for batch in read_plan.batches.iter() {
                metrics.record_file_batch(batch);

                let read_timer = metrics.read_time.timer();
                let table = match read_plan.projected_columns.as_deref() {
                    Some(columns) => CuDFTable::from_parquet_files_with_columns(
                        batch.files.as_ref(),
                        Some(columns),
                    )
                    .map_err(cudf_to_df)?,
                    None => {
                        CuDFTable::from_parquet_files(batch.files.as_ref()).map_err(cudf_to_df)?
                    }
                };
                read_timer.done();

                let sync_timer = metrics.sync_time.timer();
                synchronize_default_stream().map_err(cudf_to_df)?;
                sync_timer.done();

                let cast_timer = metrics.cast_time.timer();
                let num_rows = table.num_rows();
                let mut cudf_cols: Vec<Arc<dyn Array>> = Vec::with_capacity(table.num_columns());
                for (column, field) in table.into_columns().into_iter().zip(schema.fields()) {
                    let view = column.into_view();
                    let column: Arc<dyn Array> = if view.data_type() == field.data_type() {
                        Arc::new(view)
                    } else {
                        Arc::new(
                            cast(&view, field.data_type())
                                .map_err(cudf_to_df)?
                                .into_view(),
                        )
                    };
                    cudf_cols.push(column);
                }
                cast_timer.done();

                let output_batch_timer = metrics.output_batch_time.timer();
                let batch = libcudf_rs::record_batch_with_schema(cudf_cols, &schema, num_rows)?;
                output_batch_timer.done();

                metrics.baseline.record_output(&batch);
                let send_timer = metrics.output_send_time.timer();
                let send_result = output.blocking_send(Ok(batch));
                send_timer.done();
                if send_result.is_err() {
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
    read_time: Time,
    cast_time: Time,
    sync_time: Time,
    output_batch_time: Time,
    output_send_time: Time,
    files: Count,
    file_bytes: Count,
}

impl CuDFParquetScanMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            baseline: CuDFBaselineMetrics::new(metrics, partition),
            read_time: MetricBuilder::new(metrics).subset_time("read_time", partition),
            cast_time: MetricBuilder::new(metrics).subset_time("cast_time", partition),
            sync_time: MetricBuilder::new(metrics).subset_time("sync_time", partition),
            output_batch_time: MetricBuilder::new(metrics)
                .subset_time("output_batch_time", partition),
            output_send_time: MetricBuilder::new(metrics)
                .subset_time("output_send_time", partition),
            files: MetricBuilder::new(metrics).counter("files", partition),
            file_bytes: MetricBuilder::new(metrics).counter("file_bytes", partition),
        }
    }

    fn record_file_batch(&self, batch: &CuDFParquetFileBatch) {
        self.files.add(batch.files.len());
        self.file_bytes.add(batch.file_bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::{CuDFParquetScanConfig, CuDFParquetScanExec};
    use arrow::array::Decimal128Array;
    use arrow::record_batch::RecordBatch;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::execution::TaskContext;
    use datafusion_physical_plan::{execute_stream, ExecutionPlan};
    use futures_util::TryStreamExt;
    use libcudf_rs::is_cudf_array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::arrow::ArrowWriter;
    use std::fs::{remove_file, File};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[tokio::test]
    async fn reads_parquet_as_cudf_batches() -> Result<(), Box<dyn std::error::Error>> {
        let path = weather_file("result-000000.parquet");
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path)?)?;
        let schema = reader.schema().clone();
        let expected_rows = reader.metadata().file_metadata().num_rows() as usize;
        let expected_columns = schema.fields().len();

        let config = CuDFParquetScanConfig::new(vec![path], schema);
        let plan: Arc<dyn ExecutionPlan> = Arc::new(CuDFParquetScanExec::try_new(config)?);
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
        let path = weather_file("result-000000.parquet");
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path)?)?;
        let schema = reader.schema().clone();
        let expected_rows = reader.metadata().file_metadata().num_rows() as usize;

        let config =
            CuDFParquetScanConfig::new(vec![path], schema).with_projection(Some(vec![0, 1]));
        let plan: Arc<dyn ExecutionPlan> = Arc::new(CuDFParquetScanExec::try_new(config)?);
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

    #[tokio::test]
    async fn reads_multiple_files_in_bounded_batches() -> Result<(), Box<dyn std::error::Error>> {
        let paths = vec![
            weather_file("result-000000.parquet"),
            weather_file("result-000001.parquet"),
            weather_file("result-000002.parquet"),
        ];
        let mut expected_rows = 0;
        for path in &paths {
            let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
            expected_rows += reader.metadata().file_metadata().num_rows() as usize;
        }

        let schema = ParquetRecordBatchReaderBuilder::try_new(File::open(&paths[0])?)?
            .schema()
            .clone();
        let config = CuDFParquetScanConfig::new(paths, schema).with_files_per_batch(2);
        let plan: Arc<dyn ExecutionPlan> = Arc::new(CuDFParquetScanExec::try_new(config)?);
        let batches = execute_stream(plan, Arc::new(TaskContext::default()))?
            .try_collect::<Vec<_>>()
            .await?;

        assert_eq!(batches.len(), 2);
        assert_eq!(
            batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
            expected_rows
        );
        assert!(batches.iter().all(|batch| batch
            .columns()
            .iter()
            .all(|column| is_cudf_array(column.as_ref()))));

        Ok(())
    }

    #[tokio::test]
    async fn aligns_decimal_output_to_file_schema() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_parquet_file("decimal-schema");
        let schema = Arc::new(Schema::new(vec![Field::new(
            "amount",
            DataType::Decimal128(15, 2),
            false,
        )]));
        let amounts = Decimal128Array::from_iter_values([12345_i128, 67890_i128])
            .with_precision_and_scale(15, 2)?;
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(amounts)])?;
        write_parquet(&path, &batch)?;

        let config = CuDFParquetScanConfig::new(vec![path.clone()], Arc::clone(&schema));
        let plan: Arc<dyn ExecutionPlan> = Arc::new(CuDFParquetScanExec::try_new(config)?);
        let batches = execute_stream(plan, Arc::new(TaskContext::default()))?
            .try_collect::<Vec<_>>()
            .await?;

        remove_file(path).ok();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].schema().as_ref(), schema.as_ref());
        assert_eq!(
            batches[0].column(0).data_type(),
            &DataType::Decimal128(15, 2)
        );
        assert!(is_cudf_array(batches[0].column(0).as_ref()));

        Ok(())
    }

    fn weather_file(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../testdata/weather/{name}"))
    }

    fn temp_parquet_file(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "libcudf-datafusion-{name}-{}-{nanos}.parquet",
            std::process::id()
        ))
    }

    fn write_parquet(
        path: &PathBuf,
        batch: &RecordBatch,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
        writer.write(batch)?;
        writer.close()?;
        Ok(())
    }
}

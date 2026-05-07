//! Compare the current CPU Parquet scan -> CuDF load path with a POC cuDF
//! Parquet scan that emits GPU-backed batches directly.

use arrow_schema::SchemaRef;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use datafusion::execution::{SessionStateBuilder, TaskContext};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_physical_plan::{execute_stream, ExecutionPlan};
use futures_util::TryStreamExt;
use libcudf_datafusion::{CuDFLoadExec, CuDFParquetScanExec};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::env;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::runtime::Runtime;

const SCALE_FACTOR_ENV: &str = "LIBCUDF_PARQUET_SCAN_SCALE_FACTOR";
const TABLE_ENV: &str = "LIBCUDF_PARQUET_SCAN_TABLE";
const COLUMNS_ENV: &str = "LIBCUDF_PARQUET_SCAN_COLUMNS";
const FILES_PER_BATCH_ENV: &str = "LIBCUDF_PARQUET_SCAN_FILES_PER_BATCH";

fn scale_factor() -> String {
    env::var(SCALE_FACTOR_ENV).unwrap_or_else(|_| "1".to_string())
}

fn table_name() -> String {
    env::var(TABLE_ENV).unwrap_or_else(|_| "orders".to_string())
}

fn projected_columns() -> Vec<String> {
    env::var(COLUMNS_ENV)
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|column| !column.is_empty())
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn files_per_batch() -> usize {
    env::var(FILES_PER_BATCH_ENV)
        .ok()
        .map(|raw| {
            raw.parse::<usize>()
                .unwrap_or_else(|err| panic!("invalid {FILES_PER_BATCH_ENV} value {raw:?}: {err}"))
        })
        .unwrap_or(8)
}

fn table_dir(scale_factor: &str, table: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("data/tpch/correctness_sf{scale_factor}/{table}"));
    if !dir.exists() {
        panic!("TPC-H table data not found at {}", dir.display());
    }
    dir
}

fn parquet_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "parquet"))
        .collect();
    files.sort();
    if files.is_empty() {
        panic!("no parquet files found at {}", dir.display());
    }
    files
}

fn parquet_schema(path: &Path) -> SchemaRef {
    let file = fs::File::open(path).unwrap();
    ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .schema()
        .clone()
}

fn parquet_bytes(files: &[PathBuf]) -> u64 {
    files
        .iter()
        .map(|path| fs::metadata(path).unwrap().len())
        .sum()
}

async fn current_load_plan(
    dir: &Path,
    table: &str,
    columns: &[String],
) -> (Arc<dyn ExecutionPlan>, Arc<TaskContext>) {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(1))
        .build();
    let ctx = SessionContext::from(state);
    ctx.register_parquet(table, dir.to_str().unwrap(), ParquetReadOptions::new())
        .await
        .unwrap();
    let select = if columns.is_empty() {
        "*".to_string()
    } else {
        columns.join(", ")
    };
    let plan = ctx
        .sql(&format!("SELECT {select} FROM {table}"))
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    (
        Arc::new(CuDFLoadExec::try_new(plan).unwrap()),
        ctx.task_ctx(),
    )
}

fn cudf_scan_plan(
    files: Vec<PathBuf>,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    files_per_batch: usize,
) -> Arc<dyn ExecutionPlan> {
    Arc::new(
        CuDFParquetScanExec::try_new_with_projection_and_files_per_batch(
            files,
            schema,
            projection,
            files_per_batch,
        )
        .unwrap(),
    )
}

fn projection_for_columns(schema: &SchemaRef, columns: &[String]) -> Option<Vec<usize>> {
    if columns.is_empty() {
        return None;
    }

    Some(
        columns
            .iter()
            .map(|column| schema.index_of(column).unwrap())
            .collect(),
    )
}

async fn drain(plan: Arc<dyn ExecutionPlan>, ctx: Arc<TaskContext>) -> usize {
    let batches = execute_stream(plan, ctx)
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let rows = batches.iter().map(|batch| batch.num_rows()).sum();
    black_box(rows)
}

fn bench_parquet_scan(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let scale_factor = scale_factor();
    let table = table_name();
    let columns = projected_columns();
    let files_per_batch = files_per_batch();
    let dir = table_dir(&scale_factor, &table);
    let files = parquet_files(&dir);
    let schema = parquet_schema(&files[0]);
    let projection = projection_for_columns(&schema, &columns);
    let bytes = parquet_bytes(&files);

    let direct_ctx = Arc::new(TaskContext::default());

    // Warm the OS page cache and cuDF initialization before timed iterations.
    rt.block_on(async {
        let (plan, ctx) = current_load_plan(&dir, &table, &columns).await;
        drain(plan, ctx).await;
        drain(
            cudf_scan_plan(
                files.clone(),
                Arc::clone(&schema),
                projection.clone(),
                files_per_batch,
            ),
            Arc::clone(&direct_ctx),
        )
        .await;
    });

    let projection_label = if columns.is_empty() {
        "all".to_string()
    } else {
        format!("{}cols", columns.len())
    };
    let mut group = c.benchmark_group(format!(
        "parquet_scan/{table}/sf{scale_factor}/{projection_label}/files_per_batch_{files_per_batch}"
    ));
    group.throughput(Throughput::Bytes(bytes));

    group.bench_function(
        BenchmarkId::new("cpu_scan_plus_cudf_load", files.len()),
        |b| {
            b.iter_batched(
                || rt.block_on(current_load_plan(&dir, &table, &columns)),
                |(plan, ctx)| rt.block_on(drain(plan, ctx)),
                BatchSize::PerIteration,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new("cudf_parquet_scan_poc", files.len()),
        |b| {
            b.iter_batched(
                || {
                    cudf_scan_plan(
                        files.clone(),
                        Arc::clone(&schema),
                        projection.clone(),
                        files_per_batch,
                    )
                },
                |plan| rt.block_on(drain(plan, Arc::clone(&direct_ctx))),
                BatchSize::PerIteration,
            );
        },
    );

    group.finish();
}

criterion_group!(benches, bench_parquet_scan);
criterion_main!(benches);

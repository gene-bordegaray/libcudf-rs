//! # TPC-H Pipeline Benchmark
//!
//! ## What this benchmark measures
//!
//! End-to-end wall-clock latency for a TPC-H Q5-style pipeline:
//! Parquet scan -> hash join -> aggregate. Both CPU and GPU paths read the same
//! files from disk; the GPU path uploads the raw input once over PCIe, runs
//! the join and aggregate entirely on-device, then downloads only the 25-row
//! result.
//!
//! This is an **integration benchmark**, not a microbenchmark. It measures
//! query latency as a user would observe it. It does not isolate individual
//! operators; use `nsys` / `ncu` for per-kernel attribution.
//!
//! ## Per-operator GPU profiling (nsys)
//!
//! To attribute time between CuDFLoadExec, CuDFHashJoinExec, and
//! CuDFAggregateExec, profile the benchmark binary directly:
//!
//! ```sh
//! cargo build --release --bench tpch_pipeline
//! nsys profile --stats=true \
//!     ./target/release/deps/tpch_pipeline-* \
//!     --bench --profile-time 5
//! nsys stats last_report.nsys-rep
//! ```
//!
//! ## Data
//!
//! TPC-H SF=1 Parquet files from `testdata/tpch/correctness_sf1/`.
//!
//! | Table    | Rows    | Role               |
//! |----------|---------|--------------------|
//! | orders   | ~1.5 M  | probe (fact) side  |
//! | customer | ~150 K  | build (dim) side   |
//! | result   | 25 rows | one per nation     |

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_physical_plan::{execute_stream, ExecutionPlan};
use futures_util::TryStreamExt;
use libcudf_datafusion::aggregate::count;
use libcudf_datafusion::{CuDFConfig, HostToCuDFRule};
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn tpch_sf1_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/tpch/correctness_sf1")
}

const PIPELINE_SQL: &str = "
    SELECT c.c_nationkey, COUNT(c.c_custkey) AS cnt
    FROM orders o
    JOIN customer c ON o.o_custkey = c.c_custkey
    GROUP BY c.c_nationkey
";

async fn cpu_ctx() -> SessionContext {
    let ctx = SessionContext::new();
    let base = tpch_sf1_dir();
    ctx.register_parquet(
        "orders",
        base.join("orders").to_str().unwrap(),
        ParquetReadOptions::new(),
    )
    .await
    .unwrap();
    ctx.register_parquet(
        "customer",
        base.join("customer").to_str().unwrap(),
        ParquetReadOptions::new(),
    )
    .await
    .unwrap();
    ctx
}

async fn gpu_ctx() -> SessionContext {
    let mut cudf_config = CuDFConfig::default();
    cudf_config.enable = true;
    // Single partition: DataFusion then plans AggregateMode::Single instead of
    // Partial -> RepartitionExec -> Final. The two-phase path sends partial-state
    // batches through a CPU RepartitionExec, which strips the CuDFColumnView
    // wrappers, causing the Final CuDFAggregateExec to fail. The GPU join and
    // aggregate are both single-kernel operations and get no benefit from
    // multiple CPU feed partitions.
    let config = SessionConfig::new()
        .with_target_partitions(1)
        .with_option_extension(cudf_config);
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_physical_optimizer_rule(Arc::new(HostToCuDFRule))
        .build();
    let ctx = SessionContext::from(state);
    ctx.register_udaf((*count()).clone());
    let base = tpch_sf1_dir();
    ctx.register_parquet(
        "orders",
        base.join("orders").to_str().unwrap(),
        ParquetReadOptions::new(),
    )
    .await
    .unwrap();
    ctx.register_parquet(
        "customer",
        base.join("customer").to_str().unwrap(),
        ParquetReadOptions::new(),
    )
    .await
    .unwrap();
    ctx
}

async fn build_plan(ctx: &SessionContext, sql: &str) -> Arc<dyn ExecutionPlan> {
    ctx.sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap()
}

fn bench_tpch_pipeline(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("tpch_pipeline/orders_x_customer");

    let cpu = rt.block_on(cpu_ctx());
    let gpu = rt.block_on(gpu_ctx());
    let cpu_task_ctx = cpu.task_ctx();
    let gpu_task_ctx = gpu.task_ctx();

    // Warm the OS page cache so Parquet I/O does not appear in timed iterations.
    rt.block_on(async {
        let plan = build_plan(&cpu, PIPELINE_SQL).await;
        execute_stream(plan, cpu_task_ctx.clone())
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let plan = build_plan(&gpu, PIPELINE_SQL).await;
        execute_stream(plan, gpu_task_ctx.clone())
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
    });

    group.bench_function("cpu", |b| {
        b.iter_batched(
            || rt.block_on(build_plan(&cpu, PIPELINE_SQL)),
            |plan| {
                rt.block_on(async {
                    black_box(
                        execute_stream(plan, cpu_task_ctx.clone())
                            .unwrap()
                            .try_collect::<Vec<_>>()
                            .await
                            .unwrap(),
                    );
                })
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("gpu", |b| {
        b.iter_batched(
            || rt.block_on(build_plan(&gpu, PIPELINE_SQL)),
            |plan| {
                rt.block_on(async {
                    black_box(
                        execute_stream(plan, gpu_task_ctx.clone())
                            .unwrap()
                            .try_collect::<Vec<_>>()
                            .await
                            .unwrap(),
                    );
                })
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_tpch_pipeline);
criterion_main!(benches);

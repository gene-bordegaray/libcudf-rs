# libcudf-rs

Rust bindings for [libcudf](https://docs.rapids.ai/api/libcudf/stable/), the GPU-accelerated DataFrame library from RAPIDS.

## Overview

This project provides safe, idiomatic Rust bindings to cuDF using the [cxx](https://cxx.rs/) library for seamless 
C++/Rust interoperability. cuDF enables GPU-accelerated operations on DataFrames, offering significant performance 
improvements for data processing tasks.

## Executing SQL workloads on GPU

For SQL execution, this project uses [Apache DataFusion](https://github.com/apache/datafusion) with a physical
optimizer rule that replaces vanilla DataFusion nodes with GPU variants.

Taking the following query from the TPCH benchmark:

```sql
select
    l_returnflag,
    l_linestatus,
    sum(l_quantity) as sum_qty,
    sum(l_extendedprice) as sum_base_price,
    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
    avg(l_quantity) as avg_qty,
    avg(l_extendedprice) as avg_price,
    avg(l_discount) as avg_disc,
    count(*) as count_order
from
    lineitem
where
        l_shipdate <= date '1998-09-02'
group by
    l_returnflag,
    l_linestatus
order by
    l_returnflag,
    l_linestatus;
```

DataFusion will produce the following executable plan:

```dart
SortPreservingMergeExec: [...]
  SortExec: expr=[...], preserve_partitioning=[...]
    ProjectionExec: expr=[...]
      AggregateExec: mode=FinalPartitioned, gby=[...], aggr=[...]
        RepartitionExec: partitioning=Hash([...], 4), input_partitions=4
          AggregateExec: mode=Partial, gby=[...], aggr=[...]
            ProjectionExec: expr=[...]
              FilterExec: <expr>, projection=[...]
                DataSourceExec: file_groups={4 groups: [...]}, projection=[...]
```

This project inspects the plan and replaces nodes with their cuDF (GPU)-based variants,
producing a different executable plan that looks like this:

```dart
CuDFUnloadExec, metrics=[...]
  CuDFSortExec: expr=[...], preserve_partitioning=[...]
    CuDFProjectionExec: expr=[...]
      CuDFAggregateExec: mode=Single, group_by=[...], aggr_expr=[...]
        CuDFProjectionExec: expr=[...]
          CuDFFilterExec: l_shipdate@6 <= 1998-09-02, projection=[...]
            CuDFLoadExec, metrics=[...]
              DataSourceExec: file_groups={4 groups: [...]}, projection=[...]
```

The cuDF-based plan is indeed cheaper and faster to execute than the pure CPU one.
This was measured by comparing the execution latency in two different machines:

1. **m5.4xlarge | 16vCPU 64Gb RAM | ~$625 monthly | 906 ms TPCH Q1**
2. **g4dn.xlarge | 4vCPU 16Gb NVIDIA T4 | ~$423 monthly | 813 ms TPCH Q1**

Even if the GPU-based machine is cheaper because of having fewer vCPUs and less RAM, it's
still capable of executing TPCH Q1, so doing some basic math, the conclusion is
that, for the same latency, **executing on GPU is 1.65x cheaper** with the current
state of this project.

## What's next?

This project is the result of a couple of weeks' hackathon, and there are several low-hanging
fruit to be addressed that could make GPU execution significantly more performant.

Even though the focus of this project is to get TPCH Q1 working faster and cheaper in GPU
vs CPU, it's capable of running the full TPCH suite on GPU. Rather than implementing
a wide breadth of features, it focuses on laying the foundations for executing relational
algebra on GPUs for a wide variety of use cases.

Follow-up work will bring further performance improvements and support for new relational
algebra operations.

## Project Structure

The project is organized as a Rust workspace with the following crates:

- **libcudf-sys**: Low-level FFI bindings to libcudf using cxx
- **libcudf-rs**: Safe, high-level Rust API wrapping the FFI bindings
- **libcudf-datafusion**: Integration with Apache DataFusion

## Prerequisites

Before building this project, you need:

1. **CUDA Toolkit**: Required for GPU operations
   - Install from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

2. **libcudf**: The cuDF C++ library
   - Build from source: [cuDF build instructions](https://github.com/rapidsai/cudf)
   - Or install via conda: `conda install -c rapidsai -c conda-forge cudf`

3. **Rust toolchain**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

4. **C++ compiler**: GCC 9+ or Clang that supports C++17

## Building

Once dependencies are installed:

```bash
# Build the project
cargo build

# Run tests (requires CUDA-capable GPU)
cargo test

# Build with release optimizations
cargo build --release
```

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
libcudf-rs = { path = "path/to/libcudf-rs" }
```

## Resources

- [libcudf Documentation](https://docs.rapids.ai/api/libcudf/stable/)
- [cxx Documentation](https://cxx.rs/)
- [RAPIDS cuDF Python API](https://docs.rapids.ai/api/cudf/stable/)
- [cuDF GitHub Repository](https://github.com/rapidsai/cudf)

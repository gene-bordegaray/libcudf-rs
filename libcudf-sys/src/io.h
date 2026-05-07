#pragma once

#include <memory>
#include "rust/cxx.h"
#include "memory_resource.h"
#include "stream.h"
#include "table.h"

#include <cudf/io/parquet.hpp>

namespace libcudf_bridge {
    // Opaque wrapper for cudf::io::source_info.
    struct SourceInfo {
        cudf::io::source_info inner;

        SourceInfo();

        explicit SourceInfo(cudf::io::source_info source);

        ~SourceInfo();

        [[nodiscard]] size_t num_sources() const;
    };

    // Opaque wrapper for cudf::io::parquet_reader_options.
    struct ParquetReaderOptions {
        cudf::io::parquet_reader_options inner;

        ParquetReaderOptions();

        explicit ParquetReaderOptions(cudf::io::parquet_reader_options options);

        ~ParquetReaderOptions();

        void set_source(const SourceInfo& source);

        void set_columns(rust::Vec<rust::String> col_names);
    };

    std::unique_ptr<SourceInfo> source_info_from_file_path(rust::Str file_path);

    std::unique_ptr<SourceInfo> source_info_from_file_paths(rust::Vec<rust::String> file_paths);

    std::unique_ptr<ParquetReaderOptions> parquet_reader_options_create(const SourceInfo& source);

    void parquet_reader_options_set_source(ParquetReaderOptions& options, const SourceInfo& source);

    void parquet_reader_options_set_columns(
        ParquetReaderOptions& options,
        rust::Vec<rust::String> col_names);

    std::unique_ptr<Table> read_parquet_with_options(
        const ParquetReaderOptions& options,
        const CudaStreamView& stream,
        const DeviceAsyncResourceRef& mr);

    // Parquet I/O
    std::unique_ptr<Table> read_parquet(rust::Str filename);
    void write_parquet(const TableView &table, rust::Str filename);
} // namespace libcudf_bridge

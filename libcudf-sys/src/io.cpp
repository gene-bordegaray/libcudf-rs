#include "io.h"
#include "libcudf-sys/src/lib.rs.h"

#include <cudf/io/parquet.hpp>

namespace libcudf_bridge {
    namespace {
        std::string to_std_string(const rust::Str value) {
            return std::string(value.data(), value.size());
        }

        std::vector<std::string> to_std_strings(const rust::Vec<rust::String>& values) {
            std::vector<std::string> result;
            result.reserve(values.size());
            for (const auto& value: values) {
                result.emplace_back(static_cast<std::string>(value));
            }
            return result;
        }
    } // namespace

    SourceInfo::SourceInfo() : inner() {}

    SourceInfo::SourceInfo(cudf::io::source_info source) : inner(std::move(source)) {}

    SourceInfo::~SourceInfo() = default;

    size_t SourceInfo::num_sources() const {
        return inner.num_sources();
    }

    ParquetReaderOptions::ParquetReaderOptions() : inner() {}

    ParquetReaderOptions::ParquetReaderOptions(cudf::io::parquet_reader_options options)
        : inner(std::move(options)) {}

    ParquetReaderOptions::~ParquetReaderOptions() = default;

    void ParquetReaderOptions::set_source(const SourceInfo& source) {
        inner.set_source(source.inner);
    }

    void ParquetReaderOptions::set_columns(rust::Vec<rust::String> col_names) {
        inner.set_columns(to_std_strings(col_names));
    }

    std::unique_ptr<SourceInfo> source_info_from_file_path(const rust::Str file_path) {
        return std::make_unique<SourceInfo>(cudf::io::source_info{to_std_string(file_path)});
    }

    std::unique_ptr<SourceInfo> source_info_from_file_paths(rust::Vec<rust::String> file_paths) {
        return std::make_unique<SourceInfo>(cudf::io::source_info{to_std_strings(file_paths)});
    }

    std::unique_ptr<ParquetReaderOptions> parquet_reader_options_create(const SourceInfo& source) {
        return std::make_unique<ParquetReaderOptions>(
            cudf::io::parquet_reader_options::builder(source.inner).build());
    }

    void parquet_reader_options_set_source(ParquetReaderOptions& options, const SourceInfo& source) {
        options.set_source(source);
    }

    void parquet_reader_options_set_columns(
        ParquetReaderOptions& options,
        rust::Vec<rust::String> col_names) {
        options.set_columns(std::move(col_names));
    }

    std::unique_ptr<Table> read_parquet_with_options(
        const ParquetReaderOptions& options,
        const CudaStreamView& stream,
        const DeviceAsyncResourceRef& mr) {
        auto [tbl, metadata] = cudf::io::read_parquet(options.inner, stream.inner, mr.inner);

        auto table = std::make_unique<Table>();
        table->inner = std::move(tbl);
        return table;
    }

    // Parquet I/O
    std::unique_ptr<Table> read_parquet(rust::Str filename) {
        std::string filename_str = to_std_string(filename);
        auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filename_str});
        auto [tbl, metadata] = cudf::io::read_parquet(options.build());

        auto table = std::make_unique<Table>();
        table->inner = std::move(tbl);
        return table;
    }

    void write_parquet(const TableView &table, const rust::Str filename) {
        const std::string filename_str = to_std_string(filename);
        auto options = cudf::io::parquet_writer_options::builder(
            cudf::io::sink_info{filename_str},
            *table.inner
        );
        cudf::io::write_parquet(options.build());
    }
} // namespace libcudf_bridge

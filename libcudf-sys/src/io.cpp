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

    SinkInfo::SinkInfo() : inner() {}

    SinkInfo::SinkInfo(cudf::io::sink_info sink) : inner(std::move(sink)) {}

    SinkInfo::~SinkInfo() = default;

    size_t SinkInfo::num_sinks() const {
        return inner.num_sinks();
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

    ParquetWriterOptions::ParquetWriterOptions() : inner() {}

    ParquetWriterOptions::ParquetWriterOptions(cudf::io::parquet_writer_options options)
        : inner(std::move(options)) {}

    ParquetWriterOptions::~ParquetWriterOptions() = default;

    TableWithMetadata::TableWithMetadata(cudf::io::table_with_metadata result)
        : inner(std::move(result)) {}

    TableWithMetadata::~TableWithMetadata() = default;

    std::unique_ptr<Table> TableWithMetadata::release_table() {
        auto table = std::make_unique<Table>();
        table->inner = std::move(inner.tbl);
        return table;
    }

    size_t TableWithMetadata::num_rows_per_source_count() const {
        return inner.metadata.num_rows_per_source.size();
    }

    size_t TableWithMetadata::num_rows_per_source(size_t index) const {
        return inner.metadata.num_rows_per_source.at(index);
    }

    int32_t TableWithMetadata::num_input_row_groups() const {
        return inner.metadata.num_input_row_groups;
    }

    HostByteVector::HostByteVector(std::unique_ptr<std::vector<uint8_t>> bytes)
        : inner(std::move(bytes)) {}

    HostByteVector::~HostByteVector() = default;

    size_t HostByteVector::size() const {
        return inner ? inner->size() : 0;
    }

    std::unique_ptr<SourceInfo> source_info_from_file_path(const rust::Str file_path) {
        return std::make_unique<SourceInfo>(cudf::io::source_info{to_std_string(file_path)});
    }

    std::unique_ptr<SourceInfo> source_info_from_file_paths(rust::Vec<rust::String> file_paths) {
        return std::make_unique<SourceInfo>(cudf::io::source_info{to_std_strings(file_paths)});
    }

    std::unique_ptr<SinkInfo> sink_info_from_file_path(const rust::Str file_path) {
        return std::make_unique<SinkInfo>(cudf::io::sink_info{to_std_string(file_path)});
    }

    std::unique_ptr<SinkInfo> sink_info_from_file_paths(rust::Vec<rust::String> file_paths) {
        return std::make_unique<SinkInfo>(cudf::io::sink_info{to_std_strings(file_paths)});
    }

    std::unique_ptr<ParquetReaderOptions> parquet_reader_options_create(const SourceInfo& source) {
        return std::make_unique<ParquetReaderOptions>(
            cudf::io::parquet_reader_options::builder(source.inner).build());
    }

    std::unique_ptr<ParquetWriterOptions> parquet_writer_options_create(
        const SinkInfo& sink,
        const TableView& table) {
        return std::make_unique<ParquetWriterOptions>(
            cudf::io::parquet_writer_options::builder(sink.inner, *table.inner).build());
    }

    std::unique_ptr<TableWithMetadata> read_parquet(
        const ParquetReaderOptions& options,
        const CudaStreamView& stream,
        const DeviceAsyncResourceRef& mr) {
        auto result = cudf::io::read_parquet(options.inner, stream.inner, mr.inner);
        return std::make_unique<TableWithMetadata>(std::move(result));
    }

    std::unique_ptr<HostByteVector> write_parquet(
        const ParquetWriterOptions& options,
        const CudaStreamView& stream) {
        return std::make_unique<HostByteVector>(
            cudf::io::write_parquet(options.inner, stream.inner));
    }
} // namespace libcudf_bridge

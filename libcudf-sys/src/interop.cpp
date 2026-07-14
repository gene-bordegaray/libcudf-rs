#include "interop.h"
#include "libcudf-sys/src/lib.rs.h"

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow_device.h>
#include <stdexcept>

namespace libcudf_bridge {
    static_assert(ARROW_DEVICE_CPU == 1);
    static_assert(ARROW_DEVICE_CUDA == 2);
    static_assert(ARROW_DEVICE_CUDA_HOST == 3);
    static_assert(ARROW_DEVICE_OPENCL == 4);
    static_assert(ARROW_DEVICE_VULKAN == 7);
    static_assert(ARROW_DEVICE_METAL == 8);
    static_assert(ARROW_DEVICE_VPI == 9);
    static_assert(ARROW_DEVICE_ROCM == 10);
    static_assert(ARROW_DEVICE_ROCM_HOST == 11);
    static_assert(ARROW_DEVICE_EXT_DEV == 12);
    static_assert(ARROW_DEVICE_CUDA_MANAGED == 13);
    static_assert(ARROW_DEVICE_ONEAPI == 14);
    static_assert(ARROW_DEVICE_WEBGPU == 15);
    static_assert(ARROW_DEVICE_HEXAGON == 16);

    std::unique_ptr<Table> from_arrow_host(
        std::uint8_t const* schema_ptr,
        std::uint8_t const* device_array_ptr,
        const CudaStreamView& stream,
        const DeviceAsyncResourceRef& mr) {
        auto const* schema = reinterpret_cast<ArrowSchema const*>(schema_ptr);
        auto const* device_array = reinterpret_cast<ArrowDeviceArray const*>(device_array_ptr);

        auto result = std::make_unique<Table>();
        result->inner = cudf::from_arrow_host(schema, device_array, stream.inner, mr.inner);
        return result;
    }

    std::unique_ptr<Column> from_arrow_column(
        std::uint8_t const* schema_ptr,
        std::uint8_t const* array_ptr,
        const CudaStreamView& stream,
        const DeviceAsyncResourceRef& mr) {
        auto const* schema = reinterpret_cast<ArrowSchema const*>(schema_ptr);
        auto const* array = reinterpret_cast<ArrowArray const*>(array_ptr);

        auto result = std::make_unique<Column>();
        result->inner = cudf::from_arrow_column(schema, array, stream.inner, mr.inner);
        return result;
    }

    ColumnMetadata::ColumnMetadata(cudf::column_metadata metadata)
        : inner(std::move(metadata)) {}

    rust::String ColumnMetadata::name() const { return inner.name; }

    void ColumnMetadata::set_name(const rust::Str value) {
        inner.name.assign(value.data(), value.size());
    }

    rust::String ColumnMetadata::timezone() const { return inner.timezone; }

    void ColumnMetadata::set_timezone(const rust::Str value) {
        inner.timezone.assign(value.data(), value.size());
    }

    bool ColumnMetadata::has_precision() const { return inner.precision.has_value(); }

    std::int32_t ColumnMetadata::precision() const {
        if (!inner.precision.has_value()) {
            throw std::runtime_error("Column metadata precision is not set");
        }
        return *inner.precision;
    }

    void ColumnMetadata::set_precision(const std::int32_t value) { inner.precision = value; }

    void ColumnMetadata::clear_precision() { inner.precision.reset(); }

    std::size_t ColumnMetadata::children_len() const { return inner.children_meta.size(); }

    std::unique_ptr<ColumnMetadata> ColumnMetadata::child(const std::size_t index) const {
        if (index >= inner.children_meta.size()) {
            throw std::out_of_range("Column metadata child index is out of range");
        }
        return std::make_unique<ColumnMetadata>(inner.children_meta[index]);
    }

    void ColumnMetadata::set_child(
        const std::size_t index,
        const ColumnMetadata& child) {
        if (index >= inner.children_meta.size()) {
            throw std::out_of_range("Column metadata child index is out of range");
        }
        inner.children_meta[index] = child.inner;
    }

    void ColumnMetadata::push_child(const ColumnMetadata& child) {
        inner.children_meta.push_back(child.inner);
    }

    ColumnMetadataVector::ColumnMetadataVector(
        std::vector<cudf::column_metadata> metadata)
        : inner(std::move(metadata)) {}

    std::size_t ColumnMetadataVector::len() const { return inner.size(); }

    bool ColumnMetadataVector::is_empty() const { return inner.empty(); }

    std::unique_ptr<ColumnMetadata> ColumnMetadataVector::get(const std::size_t index) const {
        if (index >= inner.size()) {
            throw std::out_of_range("Column metadata index is out of range");
        }
        return std::make_unique<ColumnMetadata>(inner[index]);
    }

    void ColumnMetadataVector::set(
        const std::size_t index,
        const ColumnMetadata& metadata) {
        if (index >= inner.size()) {
            throw std::out_of_range("Column metadata index is out of range");
        }
        inner[index] = metadata.inner;
    }

    void ColumnMetadataVector::push(const ColumnMetadata& metadata) {
        inner.push_back(metadata.inner);
    }

    std::unique_ptr<ColumnMetadata> make_column_metadata() {
        return std::make_unique<ColumnMetadata>(cudf::column_metadata{});
    }

    std::unique_ptr<ColumnMetadata> make_column_metadata_with_name(const rust::Str name) {
        return std::make_unique<ColumnMetadata>(
            cudf::column_metadata{std::string{name.data(), name.size()}});
    }

    std::unique_ptr<ColumnMetadataVector> make_column_metadata_vector() {
        return std::make_unique<ColumnMetadataVector>(std::vector<cudf::column_metadata>{});
    }

    std::unique_ptr<ColumnMetadata> get_column_metadata(const ColumnView& input) {
        if (!input.inner) {
            throw std::runtime_error("Cannot get metadata for null column view");
        }
        return std::make_unique<ColumnMetadata>(cudf::interop::get_column_metadata(*input.inner));
    }

    std::unique_ptr<ColumnMetadataVector> get_table_metadata(const TableView& input) {
        if (!input.inner) {
            throw std::runtime_error("Cannot get metadata for null table view");
        }
        return std::make_unique<ColumnMetadataVector>(
            cudf::interop::get_table_metadata(*input.inner));
    }

    void to_arrow_schema(
        const TableView& input,
        const ColumnMetadataVector& metadata,
        std::uint8_t* out_schema_ptr) {
        if (!input.inner) {
            throw std::runtime_error("Cannot convert null table view to Arrow schema");
        }
        auto schema = cudf::to_arrow_schema(*input.inner, metadata.inner);
        auto* out = reinterpret_cast<ArrowSchema*>(out_schema_ptr);
        *out = *schema;
        schema->release = nullptr;
    }

    void to_arrow_host(
        const TableView& input,
        std::uint8_t* out_array_ptr,
        const CudaStreamView& stream,
        const DeviceAsyncResourceRef& mr) {
        if (!input.inner) {
            throw std::runtime_error("Cannot convert null table view to Arrow");
        }
        auto array = cudf::to_arrow_host(*input.inner, stream.inner, mr.inner);
        auto* out = reinterpret_cast<ArrowDeviceArray*>(out_array_ptr);
        *out = *array;
        array->array.release = nullptr;
    }

    void to_arrow_host(
        const ColumnView& input,
        std::uint8_t* out_array_ptr,
        const CudaStreamView& stream,
        const DeviceAsyncResourceRef& mr) {
        if (!input.inner) {
            throw std::runtime_error("Cannot convert null column view to Arrow");
        }
        auto array = cudf::to_arrow_host(*input.inner, stream.inner, mr.inner);
        auto* out = reinterpret_cast<ArrowDeviceArray*>(out_array_ptr);
        *out = *array;
        array->array.release = nullptr;
    }
}  // namespace libcudf_bridge

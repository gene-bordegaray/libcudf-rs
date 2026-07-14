#include "data_type.h"
#include "libcudf-sys/src/lib.rs.h"

#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

namespace libcudf_bridge {

    DataType::DataType(int32_t type_id)
        : inner(static_cast<cudf::type_id>(type_id)) {}

    DataType::DataType(int32_t type_id, int32_t scale)
        : inner(static_cast<cudf::type_id>(type_id), scale) {}

    DataType::DataType(cudf::data_type type) : inner(type) {}

    int32_t DataType::id() const {
        return static_cast<int32_t>(inner.id());
    }

    int32_t DataType::scale() const {
        return inner.scale();
    }

    std::unique_ptr<DataType> new_data_type(int32_t type_id) {
        return std::make_unique<DataType>(type_id);
    }

    std::unique_ptr<DataType> new_data_type_with_scale(int32_t type_id, int32_t scale) {
        return std::make_unique<DataType>(type_id, scale);
    }

    bool is_fixed_width(const DataType& type) {
        return cudf::is_fixed_width(type.inner);
    }

    std::size_t size_of(const DataType& type) {
        return cudf::size_of(type.inner);
    }

    std::size_t bitmask_allocation_size_bytes(
        const std::int32_t number_of_bits,
        const std::size_t padding_boundary) {
        return cudf::bitmask_allocation_size_bytes(number_of_bits, padding_boundary);
    }

} // namespace libcudf_bridge

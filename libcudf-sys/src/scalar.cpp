#include "scalar.h"
#include "data_type.h"
#include "libcudf-sys/src/lib.rs.h"

#include <cudf/scalar/scalar.hpp>
#include <stdexcept>

namespace libcudf_bridge {
    // Scalar implementation
    Scalar::Scalar() : inner(nullptr) {
    }

    Scalar::~Scalar() = default;

    // Check if the scalar is valid (not null)
    bool Scalar::is_valid(const CudaStreamView& stream) const {
        if (!inner) {
            throw std::runtime_error("Cannot inspect validity of null scalar");
        }
        return inner->is_valid(stream.inner);
    }

    // Get the data type of the scalar
    [[nodiscard]] std::unique_ptr<DataType> Scalar::type() const {
        if (!inner) {
            throw std::runtime_error("Cannot get data type of null scalar");
        }
        return std::make_unique<DataType>(inner->type());
    }

} // namespace libcudf_bridge

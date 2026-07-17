#pragma once

#include <memory>
#include <cstdint>

#include "rust/cxx.h"
#include "stream.h"
#include "memory_resource.h"

// Forward declarations
namespace cudf {
    class scalar;
}

namespace libcudf_bridge {
    struct DataType;

    // Opaque wrapper for cuDF scalar
    struct Scalar {
        std::unique_ptr<cudf::scalar> inner;

        Scalar();

        ~Scalar();

        // Check if the scalar is valid (not null)
        [[nodiscard]] bool is_valid(const CudaStreamView& stream) const;

        // Get the data type of the scalar
        [[nodiscard]] std::unique_ptr<DataType> type() const;

    };
} // namespace libcudf_bridge

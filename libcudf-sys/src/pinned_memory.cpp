#include "pinned_memory.h"
#include "libcudf-sys/src/lib.rs.h"

namespace libcudf_bridge {
    PinnedMrOptions::PinnedMrOptions() = default;

    PinnedMrOptions::PinnedMrOptions(const std::size_t pool_size)
        : inner{.pool_size = pool_size} {}

    std::unique_ptr<PinnedMrOptions> make_pinned_mr_options() {
        return std::make_unique<PinnedMrOptions>();
    }

    std::unique_ptr<PinnedMrOptions> make_pinned_mr_options_with_pool_size(
        const std::size_t pool_size) {
        return std::make_unique<PinnedMrOptions>(pool_size);
    }

    bool config_default_pinned_memory_resource(const PinnedMrOptions& options) {
        return cudf::config_default_pinned_memory_resource(options.inner);
    }

    void set_allocate_host_as_pinned_threshold(const std::size_t threshold_bytes) {
        cudf::set_allocate_host_as_pinned_threshold(threshold_bytes);
    }
}  // namespace libcudf_bridge

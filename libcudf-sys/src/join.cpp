#include "join.h"
#include <cudf/join/join.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <limits>

namespace libcudf_bridge {

constexpr cudf::size_type JoinNoneValue = std::numeric_limits<cudf::size_type>::min();

static std::unique_ptr<cudf::column> uvector_to_column(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> vec)
{
    return std::make_unique<cudf::column>(std::move(*vec), rmm::device_buffer{}, 0);
}

static std::unique_ptr<Column> uvector_to_bridge_column(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> vec)
{
    auto result = std::make_unique<Column>();
    result->inner = uvector_to_column(std::move(vec));
    return result;
}

static std::unique_ptr<JoinIndices> make_join_indices(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_idx,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_idx)
{
    auto result = std::make_unique<JoinIndices>();
    result->left = uvector_to_bridge_column(std::move(left_idx));
    result->right = uvector_to_bridge_column(std::move(right_idx));
    return result;
}

static std::unique_ptr<cudf::column> sequence_column(
    cudf::size_type size,
    cudf::size_type init,
    cudf::size_type step)
{
    auto stream = cudf::get_default_stream();
    cudf::numeric_scalar<cudf::size_type> init_scalar(init, true, stream);
    cudf::numeric_scalar<cudf::size_type> step_scalar(step, true, stream);
    return cudf::sequence(size, init_scalar, step_scalar, stream);
}

static std::unique_ptr<cudf::column> unmatched_build_indices(
    const HashJoin& join)
{
    // TODO(perf): This keeps one matched-index column per probe batch so the
    // sys crate can stay on host-callable cuDF primitives. A compact device
    // bitset would use less memory, but needs a CUDA helper or a cuDF primitive.
    if (join.matched_build_indices.empty()) {
        return sequence_column(join.build_rows, cudf::size_type{0}, cudf::size_type{1});
    }

    auto all_build_idx =
        sequence_column(join.build_rows, cudf::size_type{0}, cudf::size_type{1});

    std::vector<cudf::column_view> matched_views;
    matched_views.reserve(join.matched_build_indices.size());
    for (auto const& column : join.matched_build_indices) {
        matched_views.push_back(column->view());
    }
    auto matched_idx = cudf::concatenate(matched_views);

    cudf::table_view all_build_table{{all_build_idx->view()}};
    cudf::table_view matched_table{{matched_idx->view()}};
    cudf::filtered_join fj(matched_table, cudf::null_equality::EQUAL,
                           cudf::set_as_build_table::RIGHT, cudf::get_default_stream());
    return uvector_to_column(fj.anti_join(all_build_table));
}

// Gather left and right payloads with their respective maps and combine into one table.
static std::unique_ptr<Table> gather_combine_views(
    const cudf::column_view& left_idx,
    const cudf::column_view& right_idx,
    const cudf::table_view& left_payload,
    const cudf::table_view& right_payload,
    cudf::out_of_bounds_policy left_policy,
    cudf::out_of_bounds_policy right_policy)
{
    auto left_result  = cudf::gather(left_payload,  left_idx,  left_policy);
    auto right_result = cudf::gather(right_payload, right_idx, right_policy);

    auto cols       = left_result->release();
    auto right_cols = right_result->release();
    cols.insert(cols.end(),
        std::make_move_iterator(right_cols.begin()),
        std::make_move_iterator(right_cols.end()));

    auto result = std::make_unique<Table>();
    result->inner = std::make_unique<cudf::table>(std::move(cols));
    return result;
}

static std::unique_ptr<Table> gather_combine(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_idx,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_idx,
    const cudf::table_view& left_payload,
    const cudf::table_view& right_payload,
    cudf::out_of_bounds_policy left_policy,
    cudf::out_of_bounds_policy right_policy)
{
    auto left_col  = uvector_to_column(std::move(left_idx));
    auto right_col = uvector_to_column(std::move(right_idx));
    return gather_combine_views(left_col->view(), right_col->view(),
                                left_payload, right_payload,
                                left_policy, right_policy);
}

JoinIndices::JoinIndices() = default;

JoinIndices::~JoinIndices() = default;

std::unique_ptr<Column> JoinIndices::release_left() {
    return std::move(left);
}

std::unique_ptr<Column> JoinIndices::release_right() {
    return std::move(right);
}

std::unique_ptr<JoinIndices> inner_join_indices(
    const TableView& left_keys,
    const TableView& right_keys)
{
    auto [left_idx, right_idx] = cudf::inner_join(*left_keys.inner, *right_keys.inner);
    return make_join_indices(std::move(left_idx), std::move(right_idx));
}

std::unique_ptr<Table> inner_join_gather(
    const TableView& left_keys,  const TableView& right_keys,
    const TableView& left_payload, const TableView& right_payload)
{
    auto [left_idx, right_idx] = cudf::inner_join(*left_keys.inner, *right_keys.inner);
    return gather_combine(std::move(left_idx), std::move(right_idx),
                          *left_payload.inner, *right_payload.inner,
                          cudf::out_of_bounds_policy::DONT_CHECK,
                          cudf::out_of_bounds_policy::DONT_CHECK);
}

std::unique_ptr<Table> left_join_gather(
    const TableView& left_keys,  const TableView& right_keys,
    const TableView& left_payload, const TableView& right_payload)
{
    auto [left_idx, right_idx] = cudf::left_join(*left_keys.inner, *right_keys.inner);
    // Left map never contains sentinels (all left rows appear), right map uses INT32_MIN
    // for unmatched rows which NULLIFY converts to null output rows.
    return gather_combine(std::move(left_idx), std::move(right_idx),
                          *left_payload.inner, *right_payload.inner,
                          cudf::out_of_bounds_policy::DONT_CHECK,
                          cudf::out_of_bounds_policy::NULLIFY);
}

std::unique_ptr<Table> full_join_gather(
    const TableView& left_keys,  const TableView& right_keys,
    const TableView& left_payload, const TableView& right_payload)
{
    auto [left_idx, right_idx] = cudf::full_join(*left_keys.inner, *right_keys.inner);
    return gather_combine(std::move(left_idx), std::move(right_idx),
                          *left_payload.inner, *right_payload.inner,
                          cudf::out_of_bounds_policy::NULLIFY,
                          cudf::out_of_bounds_policy::NULLIFY);
}

HashJoin::HashJoin() = default;

HashJoin::~HashJoin() = default;

std::unique_ptr<HashJoin> hash_join_create(
    const TableView& build_keys, bool nulls_equal)
{
    auto result = std::make_unique<HashJoin>();
    auto compare_nulls = nulls_equal
        ? cudf::null_equality::EQUAL
        : cudf::null_equality::UNEQUAL;
    result->inner = std::make_unique<cudf::hash_join>(*build_keys.inner, compare_nulls);
    result->build_rows = build_keys.inner->num_rows();
    return result;
}

std::unique_ptr<Table> hash_join_inner_join_gather(
    const HashJoin& join,
    const TableView& probe_keys,
    const TableView& build_payload,
    const TableView& probe_payload)
{
    auto [probe_idx, build_idx] = join.inner->inner_join(*probe_keys.inner);
    return gather_combine(std::move(build_idx), std::move(probe_idx),
                          *build_payload.inner, *probe_payload.inner,
                          cudf::out_of_bounds_policy::DONT_CHECK,
                          cudf::out_of_bounds_policy::DONT_CHECK);
}

std::unique_ptr<Table> hash_join_inner_join_gather_and_mark(
    HashJoin& join,
    const TableView& probe_keys,
    const TableView& build_payload,
    const TableView& probe_payload)
{
    auto [probe_idx, build_idx] = join.inner->inner_join(*probe_keys.inner);
    auto build_col = uvector_to_column(std::move(build_idx));
    auto probe_col = uvector_to_column(std::move(probe_idx));
    auto result = gather_combine_views(build_col->view(), probe_col->view(),
                                       *build_payload.inner, *probe_payload.inner,
                                       cudf::out_of_bounds_policy::DONT_CHECK,
                                       cudf::out_of_bounds_policy::DONT_CHECK);
    join.matched_build_indices.push_back(std::move(build_col));
    return result;
}

std::unique_ptr<Table> hash_join_probe_left_join_gather_and_mark(
    HashJoin& join,
    const TableView& probe_keys,
    const TableView& build_payload,
    const TableView& probe_payload)
{
    auto [probe_idx, build_idx] = join.inner->left_join(*probe_keys.inner);
    auto build_col = uvector_to_column(std::move(build_idx));
    auto probe_col = uvector_to_column(std::move(probe_idx));
    auto result = gather_combine_views(build_col->view(), probe_col->view(),
                                       *build_payload.inner, *probe_payload.inner,
                                       cudf::out_of_bounds_policy::NULLIFY,
                                       cudf::out_of_bounds_policy::DONT_CHECK);
    join.matched_build_indices.push_back(std::move(build_col));
    return result;
}

std::unique_ptr<Table> hash_join_unmatched_build_gather(
    const HashJoin& join,
    const TableView& build_payload,
    const TableView& probe_payload)
{
    auto unmatched = unmatched_build_indices(join);
    auto null_probe_idx =
        sequence_column(unmatched->size(), JoinNoneValue, cudf::size_type{0});
    return gather_combine_views(unmatched->view(), null_probe_idx->view(),
                                *build_payload.inner, *probe_payload.inner,
                                cudf::out_of_bounds_policy::DONT_CHECK,
                                cudf::out_of_bounds_policy::NULLIFY);
}

std::unique_ptr<Table> left_semi_join_gather(
    const TableView& left_keys, const TableView& right_keys,
    const TableView& left_payload)
{
    cudf::filtered_join fj(*right_keys.inner, cudf::null_equality::EQUAL,
                           cudf::set_as_build_table::RIGHT, cudf::get_default_stream());
    auto idx = fj.semi_join(*left_keys.inner);
    auto idx_col = uvector_to_column(std::move(idx));
    auto result = std::make_unique<Table>();
    result->inner = cudf::gather(*left_payload.inner, idx_col->view(),
                                 cudf::out_of_bounds_policy::DONT_CHECK);
    return result;
}

std::unique_ptr<Table> left_anti_join_gather(
    const TableView& left_keys, const TableView& right_keys,
    const TableView& left_payload)
{
    cudf::filtered_join fj(*right_keys.inner, cudf::null_equality::EQUAL,
                           cudf::set_as_build_table::RIGHT, cudf::get_default_stream());
    auto idx = fj.anti_join(*left_keys.inner);
    auto idx_col = uvector_to_column(std::move(idx));
    auto result = std::make_unique<Table>();
    result->inner = cudf::gather(*left_payload.inner, idx_col->view(),
                                 cudf::out_of_bounds_policy::DONT_CHECK);
    return result;
}

std::unique_ptr<Table> cross_join(const TableView& left, const TableView& right) {
    auto result = std::make_unique<Table>();
    result->inner = cudf::cross_join(*left.inner, *right.inner);
    return result;
}

} // namespace libcudf_bridge

#include "join.h"
#include <cudf/join/join.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>

namespace libcudf_bridge {

static std::unique_ptr<cudf::column> uvector_to_column(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> vec)
{
    return std::make_unique<cudf::column>(std::move(*vec), rmm::device_buffer{}, 0);
}

static std::unique_ptr<Table> make_2col_table(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_idx,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_idx)
{
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(uvector_to_column(std::move(left_idx)));
    cols.push_back(uvector_to_column(std::move(right_idx)));
    auto result = std::make_unique<Table>();
    result->inner = std::make_unique<cudf::table>(std::move(cols));
    return result;
}

static std::unique_ptr<Table> make_one_col_table(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> idx)
{
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(uvector_to_column(std::move(idx)));
    auto result = std::make_unique<Table>();
    result->inner = std::make_unique<cudf::table>(std::move(cols));
    return result;
}

std::unique_ptr<Table> inner_join(const TableView& left_keys, const TableView& right_keys) {
    auto [left_idx, right_idx] = cudf::inner_join(*left_keys.inner, *right_keys.inner);
    return make_2col_table(std::move(left_idx), std::move(right_idx));
}

std::unique_ptr<Table> left_join(const TableView& left_keys, const TableView& right_keys) {
    auto [left_idx, right_idx] = cudf::left_join(*left_keys.inner, *right_keys.inner);
    return make_2col_table(std::move(left_idx), std::move(right_idx));
}

std::unique_ptr<Table> full_join(const TableView& left_keys, const TableView& right_keys) {
    auto [left_idx, right_idx] = cudf::full_join(*left_keys.inner, *right_keys.inner);
    return make_2col_table(std::move(left_idx), std::move(right_idx));
}

// Build hash table from right_keys (filter), probe with left_keys.
// Returns left row indices that have at least one match in right.
std::unique_ptr<Table> left_semi_join(const TableView& left_keys, const TableView& right_keys) {
    cudf::filtered_join fj(*right_keys.inner, cudf::null_equality::EQUAL,
                           cudf::set_as_build_table::RIGHT, cudf::get_default_stream());
    return make_one_col_table(fj.semi_join(*left_keys.inner));
}

// Build hash table from right_keys (filter), probe with left_keys.
// Returns left row indices that have no match in right.
std::unique_ptr<Table> left_anti_join(const TableView& left_keys, const TableView& right_keys) {
    cudf::filtered_join fj(*right_keys.inner, cudf::null_equality::EQUAL,
                           cudf::set_as_build_table::RIGHT, cudf::get_default_stream());
    return make_one_col_table(fj.anti_join(*left_keys.inner));
}

std::unique_ptr<Table> cross_join(const TableView& left, const TableView& right) {
    auto result = std::make_unique<Table>();
    result->inner = cudf::cross_join(*left.inner, *right.inner);
    return result;
}

std::unique_ptr<Table> gather_nullify(const TableView& source_table, const ColumnView& gather_map) {
    auto result = std::make_unique<Table>();
    result->inner = cudf::gather(*source_table.inner, *gather_map.inner,
                                 cudf::out_of_bounds_policy::NULLIFY);
    return result;
}

std::unique_ptr<Table> hconcat_tables(Table& left, Table& right) {
    auto left_cols = left.inner->release();
    auto right_cols = right.inner->release();
    left_cols.insert(left_cols.end(),
        std::make_move_iterator(right_cols.begin()),
        std::make_move_iterator(right_cols.end()));
    auto result = std::make_unique<Table>();
    result->inner = std::make_unique<cudf::table>(std::move(left_cols));
    return result;
}

} // namespace libcudf_bridge

#pragma once

#include <memory>
#include "rust/cxx.h"
#include "table.h"
#include "column.h"

namespace libcudf_bridge {
    // Returns a Table with 2 columns: [left_gather_map, right_gather_map] (INT32)
    std::unique_ptr<Table> inner_join(const TableView& left_keys, const TableView& right_keys);
    std::unique_ptr<Table> left_join(const TableView& left_keys, const TableView& right_keys);
    std::unique_ptr<Table> full_join(const TableView& left_keys, const TableView& right_keys);

    // Returns a Table with 1 column: [left_gather_map] (INT32)
    std::unique_ptr<Table> left_semi_join(const TableView& left_keys, const TableView& right_keys);
    std::unique_ptr<Table> left_anti_join(const TableView& left_keys, const TableView& right_keys);

    // Returns the full result table directly (Cartesian product)
    std::unique_ptr<Table> cross_join(const TableView& left, const TableView& right);

    // Gather using OUT_OF_BOUNDS_POLICY::NULLIFY for outer join unmatched rows (INT32_MIN → null)
    std::unique_ptr<Table> gather_nullify(const TableView& source_table, const ColumnView& gather_map);

    // Horizontally concatenate two tables, consuming their columns
    std::unique_ptr<Table> hconcat_tables(Table& left, Table& right);
} // namespace libcudf_bridge

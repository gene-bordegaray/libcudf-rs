#pragma once

#include <memory>
#include "rust/cxx.h"
#include "table.h"
#include "column.h"

namespace libcudf_bridge {
    // Fused join+gather functions: compute join indices and immediately gather payload
    // columns into the final output table. Keys and payload are passed separately so
    // callers can project only the needed output columns before the gather.

    std::unique_ptr<Table> inner_join_gather(
        const TableView& left_keys,  const TableView& right_keys,
        const TableView& left_payload, const TableView& right_payload);

    std::unique_ptr<Table> left_join_gather(
        const TableView& left_keys,  const TableView& right_keys,
        const TableView& left_payload, const TableView& right_payload);

    std::unique_ptr<Table> full_join_gather(
        const TableView& left_keys,  const TableView& right_keys,
        const TableView& left_payload, const TableView& right_payload);

    // Semi/anti joins only produce left-side output so only need left_payload.
    std::unique_ptr<Table> left_semi_join_gather(
        const TableView& left_keys, const TableView& right_keys,
        const TableView& left_payload);

    std::unique_ptr<Table> left_anti_join_gather(
        const TableView& left_keys, const TableView& right_keys,
        const TableView& left_payload);

    // Returns the full result table directly (Cartesian product)
    std::unique_ptr<Table> cross_join(const TableView& left, const TableView& right);
} // namespace libcudf_bridge

#pragma once

#include <memory>
#include "rust/cxx.h"
#include "table.h"
#include "column.h"

namespace libcudf_bridge {
    std::unique_ptr<Table> inner_join_gather(
        const TableView& left_keys,  const TableView& right_keys,
        const TableView& left_payload, const TableView& right_payload);

    std::unique_ptr<Table> left_join_gather(
        const TableView& left_keys,  const TableView& right_keys,
        const TableView& left_payload, const TableView& right_payload);

    std::unique_ptr<Table> full_join_gather(
        const TableView& left_keys,  const TableView& right_keys,
        const TableView& left_payload, const TableView& right_payload);

    std::unique_ptr<Table> left_semi_join_gather(
        const TableView& left_keys, const TableView& right_keys,
        const TableView& left_payload);

    std::unique_ptr<Table> left_anti_join_gather(
        const TableView& left_keys, const TableView& right_keys,
        const TableView& left_payload);

    std::unique_ptr<Table> cross_join(const TableView& left, const TableView& right);
} // namespace libcudf_bridge

#pragma once

#include <memory>
#include "rust/cxx.h"
#include "table.h"
#include "column.h"
#include <cudf/join/hash_join.hpp>
#include <cudf/column/column.hpp>
#include <vector>

namespace libcudf_bridge {
    struct JoinIndices {
        std::unique_ptr<Column> left;
        std::unique_ptr<Column> right;

        JoinIndices();

        ~JoinIndices();

        std::unique_ptr<Column> release_left();

        std::unique_ptr<Column> release_right();
    };

    struct HashJoin {
        std::unique_ptr<cudf::hash_join> inner;
        cudf::size_type build_rows = 0;
        std::vector<std::unique_ptr<cudf::column>> matched_build_indices;

        HashJoin();

        ~HashJoin();
    };

    std::unique_ptr<HashJoin> hash_join_create(
        const TableView& build_keys, bool nulls_equal);

    std::unique_ptr<Table> hash_join_inner_join_gather(
        const HashJoin& join,
        const TableView& probe_keys,
        const TableView& build_payload,
        const TableView& probe_payload);

    std::unique_ptr<Table> hash_join_inner_join_gather_and_mark(
        HashJoin& join,
        const TableView& probe_keys,
        const TableView& build_payload,
        const TableView& probe_payload);

    std::unique_ptr<Table> hash_join_probe_left_join_gather_and_mark(
        HashJoin& join,
        const TableView& probe_keys,
        const TableView& build_payload,
        const TableView& probe_payload);

    std::unique_ptr<Table> hash_join_unmatched_build_gather(
        const HashJoin& join,
        const TableView& build_payload,
        const TableView& probe_payload);

    std::unique_ptr<JoinIndices> inner_join_indices(
        const TableView& left_keys,
        const TableView& right_keys);

    std::unique_ptr<JoinIndices> left_join_indices(
        const TableView& left_keys,
        const TableView& right_keys);

    std::unique_ptr<JoinIndices> full_join_indices(
        const TableView& left_keys,
        const TableView& right_keys);

    std::unique_ptr<Table> left_semi_join_gather(
        const TableView& left_keys, const TableView& right_keys,
        const TableView& left_payload);

    std::unique_ptr<Table> left_anti_join_gather(
        const TableView& left_keys, const TableView& right_keys,
        const TableView& left_payload);

    std::unique_ptr<Table> cross_join(const TableView& left, const TableView& right);
} // namespace libcudf_bridge

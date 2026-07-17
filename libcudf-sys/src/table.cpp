#include "table.h"
#include "groupby.h"
#include "libcudf-sys/src/lib.rs.h"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>


namespace libcudf_bridge {
    std::unique_ptr<Table> create_empty_table() {
        auto table = std::make_unique<Table>();
        table->inner = std::make_unique<cudf::table>(
            std::vector<std::unique_ptr<cudf::column>>{});
        return table;
    }

    std::unique_ptr<Table> create_table_from_columns_move(
        const rust::Slice<Column *const> columns) {
        std::vector<std::unique_ptr<cudf::column>> cudf_columns;
        cudf_columns.reserve(columns.size());
        for (auto* column : columns) {
            if (column == nullptr || !column->inner) {
                throw std::invalid_argument("Cannot move a null column into a table");
            }
            cudf_columns.push_back(std::move(column->inner));
        }

        auto table = std::make_unique<Table>();
        table->inner = std::make_unique<cudf::table>(std::move(cudf_columns));
        return table;
    }

    ColumnRef::ColumnRef(const cudf::column& column) : inner(&column) {}

    size_t ColumnRef::alloc_size() const {
        if (!inner) {
            throw std::runtime_error("Cannot get allocation size of null column reference");
        }
        return inner->alloc_size();
    }

    // Table implementation
    Table::Table() : inner(nullptr) {
    }

    Table::~Table() = default;

    int32_t Table::num_columns() const {
        if (!inner) {
            throw std::runtime_error("Cannot get column count of null table");
        }
        return inner->num_columns();
    }

    int32_t Table::num_rows() const {
        if (!inner) {
            throw std::runtime_error("Cannot get row count of null table");
        }
        return inner->num_rows();
    }

    std::unique_ptr<TableView> Table::view() const {
        if (!inner) {
            throw std::runtime_error("Cannot get view of null table");
        }
        auto result = std::make_unique<TableView>();
        result->inner = std::make_unique<cudf::table_view>(inner->view());
        return result;
    }

    std::unique_ptr<ColumnRef> Table::get_column(const int32_t index) const {
        if (!inner) {
            throw std::runtime_error("Cannot get column from null table");
        }
        return std::make_unique<ColumnRef>(inner->get_column(index));
    }

    std::unique_ptr<ColumnVectorHelper> Table::release() {
        if (!inner) {
            throw std::runtime_error("Cannot release columns from null table");
        }
        auto columns = inner->release();
        auto helper = std::make_unique<ColumnVectorHelper>();
        helper->columns.reserve(columns.size());
        for (auto &cudf_col : columns) {
            auto col = column_from_unique_ptr(std::move(cudf_col));
            helper->columns.push_back(std::move(col));
        }
        return helper;
    }

    // TableView implementation
    TableView::TableView() : inner(nullptr) {
    }

    TableView::~TableView() = default;

    int32_t TableView::num_columns() const {
        if (!inner) {
            throw std::runtime_error("Cannot get column count of null table view");
        }
        return inner->num_columns();
    }

    int32_t TableView::num_rows() const {
        if (!inner) {
            throw std::runtime_error("Cannot get row count of null table view");
        }
        return inner->num_rows();
    }

    std::unique_ptr<TableView> TableView::select(const rust::Slice<const int32_t> column_indices) const {
        if (!inner) {
            throw std::runtime_error("Cannot select from null table view");
        }
        std::vector<cudf::size_type> indices;
        indices.reserve(column_indices.size());
        for (const auto idx: column_indices) {
            indices.emplace_back(idx);
        }

        auto result = std::make_unique<TableView>();
        result->inner = std::make_unique<cudf::table_view>(inner->select(indices));
        return result;
    }

    std::unique_ptr<ColumnView> TableView::column(const int32_t index) const {
        if (!inner) {
            throw std::runtime_error("Cannot get column from null table view");
        }
        auto result = std::make_unique<ColumnView>();
        result->inner = std::make_unique<cudf::column_view>(inner->column(index));
        return result;
    }

    std::unique_ptr<TableView> table_view_clone(const TableView& view) {
        if (!view.inner) {
            throw std::runtime_error("Cannot clone null table view");
        }
        auto cloned = std::make_unique<TableView>();
        cloned->inner = std::make_unique<cudf::table_view>(*view.inner);
        return cloned;
    }

    // TableView factory function
    std::unique_ptr<TableView> create_table_view_from_column_views(rust::Slice<const ColumnView *const> column_views) {
        std::vector<cudf::column_view> views;
        views.reserve(column_views.size());

        for (const auto *col_view_ptr: column_views) {
            if (col_view_ptr == nullptr || !col_view_ptr->inner) {
                throw std::invalid_argument("Cannot create a table view from a null column view");
            }
            views.push_back(*col_view_ptr->inner);
        }

        auto result = std::make_unique<TableView>();
        result->inner = std::make_unique<cudf::table_view>(views);
        return result;
    }
} // namespace libcudf_bridge

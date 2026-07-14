#pragma once

#include <memory>
#include <cstdint>
#include "rust/cxx.h"
#include "table.h"
#include "column.h"
#include "scalar.h"
#include "stream.h"
#include "memory_resource.h"

namespace libcudf_bridge {
    // Direct cuDF operations exposed through bridge-owned return types.
    std::unique_ptr<Table> apply_boolean_mask(
        const TableView &table,
        const ColumnView &boolean_mask,
        const CudaStreamView &stream,
        const DeviceAsyncResourceRef &mr);

    std::unique_ptr<Table> concat_table_views(
        rust::Slice<const std::unique_ptr<TableView>> views,
        const CudaStreamView &stream,
        const DeviceAsyncResourceRef &mr);

    std::unique_ptr<Column> concat_column_views(
        rust::Slice<const std::unique_ptr<ColumnView>> views,
        const CudaStreamView &stream,
        const DeviceAsyncResourceRef &mr);

    std::unique_ptr<Column> make_column_from_scalar(
        const Scalar &scalar,
        size_t size,
        const CudaStreamView &stream,
        const DeviceAsyncResourceRef &mr);

    std::unique_ptr<Column> sequence(
        size_t size,
        const Scalar &init,
        const Scalar &step,
        const CudaStreamView &stream,
        const DeviceAsyncResourceRef &mr);

    // Gather rows from a table based on a gather map (column of indices)
    std::unique_ptr<Table> gather(
        const TableView &source_table,
        const ColumnView &gather_map,
        const CudaStreamView &stream,
        const DeviceAsyncResourceRef &mr);

    std::unique_ptr<Table> gather_with_policy(
        const TableView &source_table,
        const ColumnView &gather_map,
        int32_t out_of_bounds_policy,
        const CudaStreamView &stream,
        const DeviceAsyncResourceRef &mr);

    std::unique_ptr<Table> scatter_scalars(
        rust::Slice<const Scalar *const> source,
        const ColumnView &indices,
        const TableView &target,
        const CudaStreamView &stream,
        const DeviceAsyncResourceRef &mr);

    std::unique_ptr<Table> distinct(
        const TableView &input,
        rust::Slice<const int32_t> keys,
        int32_t keep,
        int32_t nulls_equal,
        int32_t nans_equal,
        const CudaStreamView &stream,
        const DeviceAsyncResourceRef &mr);

    // Create a sliced view of a column
    std::unique_ptr<ColumnView> slice_column(
        const ColumnView &column,
        size_t offset,
        size_t length,
        const CudaStreamView &stream);

    // Utility functions
    rust::String get_cudf_version();

} // namespace libcudf_bridge

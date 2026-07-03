use crate::execution_policy;
use crate::{CuDFColumn, CuDFError, CuDFExecutionContext, CuDFTable, CuDFTableView};
use arrow::error::ArrowError;
use libcudf_sys::{ffi, NullOrder, Order};

/// Sort direction and null precedence for cuDF table sorting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SortOrder {
    /// Sort ascending with null values before non-null values.
    AscendingNullsFirst,
    /// Sort ascending with null values after non-null values.
    AscendingNullsLast,
    /// Sort descending with null values before non-null values.
    DescendingNullsFirst,
    /// Sort descending with null values after non-null values.
    DescendingNullsLast,
}

impl SortOrder {
    fn order(self) -> Order {
        match self {
            SortOrder::AscendingNullsFirst | SortOrder::AscendingNullsLast => Order::Ascending,
            SortOrder::DescendingNullsFirst | SortOrder::DescendingNullsLast => Order::Descending,
        }
    }

    fn null_order(self) -> NullOrder {
        match self {
            SortOrder::AscendingNullsFirst | SortOrder::DescendingNullsFirst => NullOrder::Before,
            SortOrder::AscendingNullsLast | SortOrder::DescendingNullsLast => NullOrder::After,
        }
    }
}

pub(crate) fn sort_by_on_context(
    ctx: &CuDFExecutionContext,
    table: &CuDFTableView,
    key_columns: &[usize],
    sort_orders: &[SortOrder],
) -> Result<CuDFTable, CuDFError> {
    if key_columns.is_empty() {
        return Err(CuDFError::ArrowError(ArrowError::InvalidArgumentError(
            "key_columns cannot be empty".to_string(),
        )));
    }

    validate_sort_order_len("sort_by", key_columns.len(), sort_orders.len())?;

    let key_views: Result<Vec<_>, _> = key_columns
        .iter()
        .map(|&column_index| table.column(column_index))
        .collect();
    let key_views = key_views?;
    let keys_view = CuDFTableView::from_column_views(key_views)?;

    let (column_order, null_precedence) = encoded_sort_options(sort_orders);
    let mut launch = execution_policy::launch(ctx)?;
    launch.wait_table(table)?;
    launch.wait_table(&keys_view)?;

    let inner = ffi::stable_sort_by_key(
        table.inner(),
        keys_view.inner(),
        &column_order,
        &null_precedence,
        launch.stream()?,
        launch.resource(),
    )?;
    launch.ready_table(CuDFTable::from_inner(inner))
}

pub(crate) fn sort_by_all_on_context(
    ctx: &CuDFExecutionContext,
    table: &CuDFTableView,
    sort_orders: &[SortOrder],
) -> Result<CuDFTable, CuDFError> {
    validate_sort_order_len("sort_by_all", table.num_columns(), sort_orders.len())?;
    let (column_order, null_precedence) = encoded_sort_options(sort_orders);
    let mut launch = execution_policy::launch(ctx)?;
    launch.wait_table(table)?;

    let inner = ffi::stable_sort_table(
        table.inner(),
        &column_order,
        &null_precedence,
        launch.stream()?,
        launch.resource(),
    )?;
    launch.ready_table(CuDFTable::from_inner(inner))
}

pub(crate) fn stable_sorted_order_on_context(
    ctx: &CuDFExecutionContext,
    table: &CuDFTableView,
    sort_orders: &[SortOrder],
) -> Result<CuDFColumn, CuDFError> {
    validate_sort_order_len(
        "stable_sorted_order",
        table.num_columns(),
        sort_orders.len(),
    )?;
    let (column_order, null_precedence) = encoded_sort_options(sort_orders);
    let mut launch = execution_policy::launch(ctx)?;
    launch.wait_table(table)?;

    let inner = ffi::stable_sorted_order(
        table.inner(),
        &column_order,
        &null_precedence,
        launch.stream()?,
        launch.resource(),
    )?;
    launch.ready_column(CuDFColumn::from_inner(inner))
}

fn validate_sort_order_len(
    operation: &str,
    expected_columns: usize,
    actual_orders: usize,
) -> Result<(), CuDFError> {
    if actual_orders == 0 || actual_orders == expected_columns {
        return Ok(());
    }

    Err(CuDFError::ArrowError(ArrowError::InvalidArgumentError(
        format!(
            "{operation} expected either no sort orders or one per sorted column \
             ({expected_columns}), got {actual_orders}"
        ),
    )))
}

fn encoded_sort_options(sort_orders: &[SortOrder]) -> (Vec<i32>, Vec<i32>) {
    let column_order = sort_orders
        .iter()
        .map(|&order| order.order() as i32)
        .collect();
    let null_precedence = sort_orders
        .iter()
        .map(|&order| order.null_order() as i32)
        .collect();
    (column_order, null_precedence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::*;
    use arrow::record_batch::RecordBatch;

    #[test]
    fn test_sort_ascending() -> Result<(), Box<dyn std::error::Error>> {
        let batch = record_batch!(("a", Int32, [3, 1, 4, 1, 5, 9, 2, 6]))?;

        let result = sort_by(batch, &[0], &[SortOrder::AscendingNullsLast])?;
        assert_i32_column(&result, 0, &[1, 1, 2, 3, 4, 5, 6, 9]);
        Ok(())
    }

    #[test]
    fn test_sort_descending() -> Result<(), Box<dyn std::error::Error>> {
        let batch = record_batch!(("a", Int32, [3, 1, 4, 1, 5]))?;

        let result = sort_by(batch, &[0], &[SortOrder::DescendingNullsLast])?;
        assert_i32_column(&result, 0, &[5, 4, 3, 1, 1]);
        Ok(())
    }

    #[test]
    fn test_sort_multiple_columns() -> Result<(), Box<dyn std::error::Error>> {
        let batch = record_batch!(
            ("a", Int32, [3, 1, 3, 1, 2]),
            ("b", Int32, [10, 20, 30, 40, 50])
        )?;

        let result = sort_by(
            batch,
            &[0, 1],
            &[SortOrder::AscendingNullsLast, SortOrder::AscendingNullsLast],
        )?;

        // Expected: First sort by A, then by B within same A values
        // (1, 20), (1, 40), (2, 50), (3, 10), (3, 30)
        assert_i32_columns(&result, &[&[1, 1, 2, 3, 3], &[20, 40, 50, 10, 30]]);
        Ok(())
    }

    #[test]
    fn test_sort_first_column_only() -> Result<(), Box<dyn std::error::Error>> {
        let batch = record_batch!(
            ("a", Int32, [3, 1, 3, 1, 2]),
            ("b", Int32, [10, 40, 30, 20, 50])
        )?;

        let result = sort_by(batch, &[0], &[SortOrder::AscendingNullsLast])?;

        // Expected: Sort by A only, stable sort preserves original B order within same A
        // (1, 40), (1, 20), (2, 50), (3, 10), (3, 30)
        assert_i32_columns(&result, &[&[1, 1, 2, 3, 3], &[40, 20, 50, 10, 30]]);
        Ok(())
    }

    #[test]
    fn test_sort_by_all() -> Result<(), Box<dyn std::error::Error>> {
        let batch = record_batch!(
            ("a", Int32, [3, 1, 3, 1, 2]),
            ("b", Int32, [10, 40, 30, 20, 50])
        )?;

        let result = sort_by_all(
            batch,
            &[SortOrder::AscendingNullsLast, SortOrder::AscendingNullsLast],
        )?;

        // Expected: Sort by A first, then by B as tiebreaker
        // (1, 20), (1, 40), (2, 50), (3, 10), (3, 30)
        assert_i32_columns(&result, &[&[1, 1, 2, 3, 3], &[20, 40, 50, 10, 30]]);
        Ok(())
    }

    fn sort_by(
        batch: RecordBatch,
        key_columns: &[usize],
        orders: &[SortOrder],
    ) -> Result<RecordBatch, CuDFError> {
        let table = crate::execute_cudf(CuDFTable::from_arrow_host(batch))?;
        let sorted = crate::execute_cudf(table.into_view().sort_by(key_columns, orders))?;
        crate::execute_cudf(sorted.into_view().to_arrow_host())
    }

    fn sort_by_all(batch: RecordBatch, orders: &[SortOrder]) -> Result<RecordBatch, CuDFError> {
        let table = crate::execute_cudf(CuDFTable::from_arrow_host(batch))?;
        let sorted = crate::execute_cudf(table.into_view().sort_by_all(orders))?;
        crate::execute_cudf(sorted.into_view().to_arrow_host())
    }

    fn assert_i32_columns(batch: &RecordBatch, expected_columns: &[&[i32]]) {
        for (column, expected) in expected_columns.iter().enumerate() {
            assert_i32_column(batch, column, expected);
        }
    }

    fn assert_i32_column(batch: &RecordBatch, column: usize, expected: &[i32]) {
        let col: &Int32Array = cast(batch.column(column));
        assert_eq!(col.values(), expected);
    }

    fn cast<T: 'static + Array>(arr: &dyn Array) -> &T {
        arr.as_any().downcast_ref::<T>().unwrap()
    }
}

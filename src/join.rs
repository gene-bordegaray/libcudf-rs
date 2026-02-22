use crate::{CuDFError, CuDFTable, CuDFTableView};
use libcudf_sys::ffi;

fn select_keys(view: &CuDFTableView, cols: &[usize]) -> cxx::UniquePtr<ffi::TableView> {
    let indices: Vec<i32> = cols.iter().map(|&i| i as i32).collect();
    view.inner().select(&indices)
}

fn gather_and_combine(
    left: &CuDFTableView,
    right: &CuDFTableView,
    maps: cxx::UniquePtr<ffi::Table>,
    outer: bool,
) -> Result<CuDFTable, CuDFError> {
    let maps_view = maps.view();
    let left_map = maps_view.column(0);
    let right_map = maps_view.column(1);

    let mut left_result = if outer {
        ffi::gather_nullify(left.inner(), &left_map)?
    } else {
        ffi::gather(left.inner(), &left_map)?
    };
    let mut right_result = if outer {
        ffi::gather_nullify(right.inner(), &right_map)?
    } else {
        ffi::gather(right.inner(), &right_map)?
    };

    let combined = ffi::hconcat_tables(left_result.pin_mut(), right_result.pin_mut())?;
    Ok(CuDFTable::from_inner(combined))
}

/// Perform an inner join on two tables using the specified key columns.
///
/// Returns a table containing all rows that have matching keys in both inputs.
/// Columns from both tables are concatenated: `[left_cols | right_cols]`.
pub fn inner_join(
    left: &CuDFTableView,
    right: &CuDFTableView,
    left_on: &[usize],
    right_on: &[usize],
) -> Result<CuDFTable, CuDFError> {
    let maps = ffi::inner_join(&select_keys(left, left_on), &select_keys(right, right_on))?;
    gather_and_combine(left, right, maps, false)
}

/// Perform a left outer join on two tables.
///
/// Returns a table with all rows from the left input. Unmatched right rows
/// produce nulls in the right columns.
pub fn left_join(
    left: &CuDFTableView,
    right: &CuDFTableView,
    left_on: &[usize],
    right_on: &[usize],
) -> Result<CuDFTable, CuDFError> {
    let maps = ffi::left_join(&select_keys(left, left_on), &select_keys(right, right_on))?;
    gather_and_combine(left, right, maps, true)
}

/// Perform a full outer join on two tables.
///
/// Returns all rows from both inputs. Unmatched rows produce nulls on the
/// opposite side.
pub fn full_join(
    left: &CuDFTableView,
    right: &CuDFTableView,
    left_on: &[usize],
    right_on: &[usize],
) -> Result<CuDFTable, CuDFError> {
    let maps = ffi::full_join(&select_keys(left, left_on), &select_keys(right, right_on))?;
    gather_and_combine(left, right, maps, true)
}

/// Perform a left semi join — return only left rows that have at least one match.
///
/// Only left columns are included in the output.
pub fn left_semi_join(
    left: &CuDFTableView,
    right: &CuDFTableView,
    left_on: &[usize],
    right_on: &[usize],
) -> Result<CuDFTable, CuDFError> {
    let maps = ffi::left_semi_join(&select_keys(left, left_on), &select_keys(right, right_on))?;
    let gather_map = maps.view().column(0);
    Ok(CuDFTable::from_inner(ffi::gather(
        left.inner(),
        &gather_map,
    )?))
}

/// Perform a left anti join - return only left rows that have no match.
///
/// Only left columns are included in the output.
pub fn left_anti_join(
    left: &CuDFTableView,
    right: &CuDFTableView,
    left_on: &[usize],
    right_on: &[usize],
) -> Result<CuDFTable, CuDFError> {
    let maps = ffi::left_anti_join(&select_keys(left, left_on), &select_keys(right, right_on))?;
    let gather_map = maps.view().column(0);
    Ok(CuDFTable::from_inner(ffi::gather(
        left.inner(),
        &gather_map,
    )?))
}

/// Perform a cross join (Cartesian product) of two tables.
///
/// Returns all combinations of rows from both inputs.
pub fn cross_join(left: &CuDFTableView, right: &CuDFTableView) -> Result<CuDFTable, CuDFError> {
    Ok(CuDFTable::from_inner(ffi::cross_join(
        left.inner(),
        right.inner(),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_table(keys: Vec<i32>, vals: Vec<i32>) -> CuDFTable {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("val", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(keys)),
                Arc::new(Int32Array::from(vals)),
            ],
        )
        .unwrap();
        CuDFTable::from_arrow_host(batch).unwrap()
    }

    #[test]
    fn test_inner_join() -> Result<(), Box<dyn std::error::Error>> {
        let left = make_table(vec![1, 2, 3, 4], vec![10, 20, 30, 40]);
        let right = make_table(vec![2, 3, 5], vec![200, 300, 500]);

        let result = inner_join(&left.into_view(), &right.into_view(), &[0], &[0])?;
        assert_eq!(result.num_rows(), 2); // keys 2 and 3 match
        assert_eq!(result.num_columns(), 4); // left.key, left.val, right.key, right.val
        Ok(())
    }

    #[test]
    fn test_left_join() -> Result<(), Box<dyn std::error::Error>> {
        let left = make_table(vec![1, 2, 3], vec![10, 20, 30]);
        let right = make_table(vec![2, 4], vec![200, 400]);

        let result = left_join(&left.into_view(), &right.into_view(), &[0], &[0])?;
        assert_eq!(result.num_rows(), 3); // all 3 left rows
        assert_eq!(result.num_columns(), 4);
        Ok(())
    }

    #[test]
    fn test_full_join() -> Result<(), Box<dyn std::error::Error>> {
        let left = make_table(vec![1, 2], vec![10, 20]);
        let right = make_table(vec![2, 3], vec![200, 300]);

        let result = full_join(&left.into_view(), &right.into_view(), &[0], &[0])?;
        assert_eq!(result.num_rows(), 3); // key 2 matches, key 1 left-only, key 3 right-only
        assert_eq!(result.num_columns(), 4);
        Ok(())
    }

    #[test]
    fn test_left_semi_join() -> Result<(), Box<dyn std::error::Error>> {
        let left = make_table(vec![1, 2, 3, 4], vec![10, 20, 30, 40]);
        let right = make_table(vec![2, 3, 5], vec![200, 300, 500]);

        let result = left_semi_join(&left.into_view(), &right.into_view(), &[0], &[0])?;
        assert_eq!(result.num_rows(), 2); // keys 2 and 3 match
        assert_eq!(result.num_columns(), 2); // only left columns
        Ok(())
    }

    #[test]
    fn test_left_anti_join() -> Result<(), Box<dyn std::error::Error>> {
        let left = make_table(vec![1, 2, 3, 4], vec![10, 20, 30, 40]);
        let right = make_table(vec![2, 3, 5], vec![200, 300, 500]);

        let result = left_anti_join(&left.into_view(), &right.into_view(), &[0], &[0])?;
        assert_eq!(result.num_rows(), 2); // keys 1 and 4 have no match
        assert_eq!(result.num_columns(), 2); // only left columns
        Ok(())
    }

    #[test]
    fn test_cross_join() -> Result<(), Box<dyn std::error::Error>> {
        let left = make_table(vec![1, 2], vec![10, 20]);
        let right = make_table(vec![3, 4, 5], vec![30, 40, 50]);

        let result = cross_join(&left.into_view(), &right.into_view())?;
        assert_eq!(result.num_rows(), 6); // 2 * 3 = 6
        assert_eq!(result.num_columns(), 4);
        Ok(())
    }

    #[test]
    fn test_inner_join_empty_result() -> Result<(), Box<dyn std::error::Error>> {
        let left = make_table(vec![1, 2, 3], vec![10, 20, 30]);
        let right = make_table(vec![4, 5, 6], vec![40, 50, 60]);

        let result = inner_join(&left.into_view(), &right.into_view(), &[0], &[0])?;
        assert_eq!(result.num_rows(), 0);
        assert_eq!(result.num_columns(), 4);
        Ok(())
    }
}

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
    left_out_cols: Option<&[usize]>,
    right_out_cols: Option<&[usize]>,
) -> Result<CuDFTable, CuDFError> {
    let maps_view = maps.view();
    let left_map = maps_view.column(0);
    let right_map = maps_view.column(1);

    // Pre-select output columns if requested; otherwise pass the full view directly.
    let left_sel = left_out_cols.map(|cols| select_keys(left, cols));
    let right_sel = right_out_cols.map(|cols| select_keys(right, cols));
    let left_src = left_sel.as_ref().unwrap_or_else(|| left.inner());
    let right_src = right_sel.as_ref().unwrap_or_else(|| right.inner());

    let mut left_result = if outer {
        ffi::gather_nullify(left_src, &left_map)?
    } else {
        ffi::gather(left_src, &left_map)?
    };
    let mut right_result = if outer {
        ffi::gather_nullify(right_src, &right_map)?
    } else {
        ffi::gather(right_src, &right_map)?
    };

    let combined = ffi::hconcat_tables(left_result.pin_mut(), right_result.pin_mut())?;
    Ok(CuDFTable::from_inner(combined))
}

/// Perform an inner join on two tables using the specified key columns.
///
/// Returns a table containing all rows that have matching keys in both inputs.
/// Columns from both tables are concatenated: `[left_cols | right_cols]`.
/// Pass `left_out_cols` / `right_out_cols` to gather only a subset of columns.
pub fn inner_join(
    left: &CuDFTableView,
    right: &CuDFTableView,
    left_on: &[usize],
    right_on: &[usize],
    left_out_cols: Option<&[usize]>,
    right_out_cols: Option<&[usize]>,
) -> Result<CuDFTable, CuDFError> {
    let maps = ffi::inner_join(&select_keys(left, left_on), &select_keys(right, right_on))?;
    gather_and_combine(left, right, maps, false, left_out_cols, right_out_cols)
}

/// Perform a left outer join on two tables.
///
/// Returns a table with all rows from the left input. Unmatched right rows
/// produce nulls in the right columns.
/// Pass `left_out_cols` / `right_out_cols` to gather only a subset of columns.
pub fn left_join(
    left: &CuDFTableView,
    right: &CuDFTableView,
    left_on: &[usize],
    right_on: &[usize],
    left_out_cols: Option<&[usize]>,
    right_out_cols: Option<&[usize]>,
) -> Result<CuDFTable, CuDFError> {
    let maps = ffi::left_join(&select_keys(left, left_on), &select_keys(right, right_on))?;
    gather_and_combine(left, right, maps, true, left_out_cols, right_out_cols)
}

/// Perform a full outer join on two tables.
///
/// Returns all rows from both inputs. Unmatched rows produce nulls on the
/// opposite side.
/// Pass `left_out_cols` / `right_out_cols` to gather only a subset of columns.
pub fn full_join(
    left: &CuDFTableView,
    right: &CuDFTableView,
    left_on: &[usize],
    right_on: &[usize],
    left_out_cols: Option<&[usize]>,
    right_out_cols: Option<&[usize]>,
) -> Result<CuDFTable, CuDFError> {
    let maps = ffi::full_join(&select_keys(left, left_on), &select_keys(right, right_on))?;
    gather_and_combine(left, right, maps, true, left_out_cols, right_out_cols)
}

/// Perform a left semi join - return only left rows that have at least one match.
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
    use arrow::array::{Array, Int32Array, RecordBatch};
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

        let result = inner_join(
            &left.into_view(),
            &right.into_view(),
            &[0],
            &[0],
            None,
            None,
        )?;
        assert_eq!(result.num_rows(), 2); // keys 2 and 3 match
        assert_eq!(result.num_columns(), 4); // left.key, left.val, right.key, right.val
        Ok(())
    }

    #[test]
    fn test_left_join() -> Result<(), Box<dyn std::error::Error>> {
        let left = make_table(vec![1, 2, 3], vec![10, 20, 30]);
        let right = make_table(vec![2, 4], vec![200, 400]);

        let result = left_join(
            &left.into_view(),
            &right.into_view(),
            &[0],
            &[0],
            None,
            None,
        )?;
        assert_eq!(result.num_rows(), 3); // all 3 left rows
        assert_eq!(result.num_columns(), 4);
        Ok(())
    }

    #[test]
    fn test_full_join() -> Result<(), Box<dyn std::error::Error>> {
        let left = make_table(vec![1, 2], vec![10, 20]);
        let right = make_table(vec![2, 3], vec![200, 300]);

        let result = full_join(
            &left.into_view(),
            &right.into_view(),
            &[0],
            &[0],
            None,
            None,
        )?;
        assert_eq!(result.num_rows(), 3); // key 2 matches, key 1 left-only, key 3 right-only
        assert_eq!(result.num_columns(), 4);
        Ok(())
    }

    #[test]
    fn test_inner_join_column_subset() -> Result<(), Box<dyn std::error::Error>> {
        let left = make_table(vec![1, 2, 3, 4], vec![10, 20, 30, 40]);
        let right = make_table(vec![2, 3, 5], vec![200, 300, 500]);

        // Select only left.val (col 1) and right.val (col 1) -> drop both key columns.
        let result = inner_join(
            &left.into_view(),
            &right.into_view(),
            &[0],
            &[0],
            Some(&[1]),
            Some(&[1]),
        )?;

        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 2); // only val from each side

        let batch = result.into_view().to_arrow_host()?;
        let left_vals = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let right_vals = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let mut pairs: Vec<(i32, i32)> = (0..left_vals.len())
            .map(|i| (left_vals.value(i), right_vals.value(i)))
            .collect();
        pairs.sort();
        assert_eq!(pairs, vec![(20, 200), (30, 300)]);
        Ok(())
    }

    #[test]
    fn test_left_join_column_subset_nulls() -> Result<(), Box<dyn std::error::Error>> {
        let left = make_table(vec![1, 2, 3], vec![10, 20, 30]);
        let right = make_table(vec![2, 4], vec![200, 400]);

        // Select only right.val (col 1) -> left side gets all columns.
        let result = left_join(
            &left.into_view(),
            &right.into_view(),
            &[0],
            &[0],
            None,
            Some(&[1]),
        )?;

        assert_eq!(result.num_rows(), 3); // all left rows preserved
        assert_eq!(result.num_columns(), 3); // left.key, left.val, right.val

        let batch = result.into_view().to_arrow_host()?;
        let right_val_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        // key=1 and key=3 have no match -> right.val must be null for those rows.
        let nulls: usize = (0..right_val_col.len())
            .filter(|&i| right_val_col.is_null(i))
            .count();
        assert_eq!(nulls, 2);
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

        let result = inner_join(
            &left.into_view(),
            &right.into_view(),
            &[0],
            &[0],
            None,
            None,
        )?;
        assert_eq!(result.num_rows(), 0);
        assert_eq!(result.num_columns(), 4);
        Ok(())
    }
}

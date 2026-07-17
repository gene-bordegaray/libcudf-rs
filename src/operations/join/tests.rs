use super::utils::int32_scalar;
use super::*;
use crate::{
    CuDFAstExpression, CuDFAstOperator, CuDFAstTableReference, CuDFError, CuDFTable, CuDFTableView,
};
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
    CuDFTable::try_from_arrow_host(batch).unwrap()
}

fn int32_values(batch: &RecordBatch, column: usize) -> Vec<i32> {
    let values = batch
        .column(column)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    (0..values.len()).map(|i| values.value(i)).collect()
}

fn int32_options(batch: &RecordBatch, column: usize) -> Vec<Option<i32>> {
    let values = batch
        .column(column)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    (0..values.len())
        .map(|i| {
            if values.is_null(i) {
                None
            } else {
                Some(values.value(i))
            }
        })
        .collect()
}

fn value_less_predicate() -> Result<CuDFAstExpression, CuDFError> {
    let mut predicate = CuDFAstExpression::new();
    let build_value = predicate.column_reference(1, CuDFAstTableReference::Left)?;
    let probe_value = predicate.column_reference(1, CuDFAstTableReference::Right)?;
    predicate.binary_operation(CuDFAstOperator::Less, build_value, probe_value)?;
    Ok(predicate)
}

fn filtered_join_args<'a>(
    probe_view: &'a CuDFTableView,
    predicate: &'a CuDFAstExpression,
    build_view: &'a CuDFTableView,
) -> CuDFFilteredHashJoinArgs<'a> {
    CuDFFilteredHashJoinArgs {
        probe: probe_view,
        probe_on: &[0],
        build_conditional: build_view,
        probe_conditional: probe_view,
        predicate,
        build_payload: build_view,
        probe_payload: probe_view,
        build_out_cols: Some(&[1]),
        probe_out_cols: Some(&[1]),
    }
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
fn test_full_join_column_subset_nulls() -> Result<(), Box<dyn std::error::Error>> {
    let left = make_table(vec![1, 2], vec![10, 20]);
    let right = make_table(vec![2, 3], vec![200, 300]);

    let result = full_join(
        &left.into_view(),
        &right.into_view(),
        &[0],
        &[0],
        Some(&[1]),
        Some(&[1]),
    )?;

    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.num_columns(), 2);

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
    assert_eq!(
        (0..left_vals.len())
            .filter(|&i| left_vals.is_null(i))
            .count(),
        1
    );
    assert_eq!(
        (0..right_vals.len())
            .filter(|&i| right_vals.is_null(i))
            .count(),
        1
    );
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
fn test_hash_join_inner_join_multiple_probes() -> Result<(), Box<dyn std::error::Error>> {
    let build = Arc::new(make_table(vec![1, 2, 3], vec![10, 20, 30]));
    let build_view = Arc::clone(&build).view();
    let join = CuDFHashJoin::try_new(&build_view, &[0], CuDFNullEquality::Unequal)?;

    let probe_a = make_table(vec![2], vec![200]);
    let probe_a_view = probe_a.into_view();
    let result_a = join.inner_join(
        &probe_a_view,
        &[0],
        &build_view,
        &probe_a_view,
        Some(&[1]),
        Some(&[1]),
    )?;

    let probe_b = make_table(vec![3], vec![300]);
    let probe_b_view = probe_b.into_view();
    let result_b = join.inner_join(
        &probe_b_view,
        &[0],
        &build_view,
        &probe_b_view,
        Some(&[1]),
        Some(&[1]),
    )?;

    assert_eq!(result_a.num_rows(), 1);
    assert_eq!(result_b.num_rows(), 1);
    assert_eq!(result_a.num_columns(), 2);
    assert_eq!(result_b.num_columns(), 2);

    let batch_a = result_a.into_view().to_arrow_host()?;
    let batch_b = result_b.into_view().to_arrow_host()?;
    let left_a = batch_a
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let right_a = batch_a
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let left_b = batch_b
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let right_b = batch_b
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!((left_a.value(0), right_a.value(0)), (20, 200));
    assert_eq!((left_b.value(0), right_b.value(0)), (30, 300));
    Ok(())
}

#[test]
fn test_gather_filtered_hash_join_indices() -> Result<(), Box<dyn std::error::Error>> {
    let build = Arc::new(make_table(vec![1, 2, 2, 3], vec![10, 20, 25, 30]));
    let build_view = Arc::clone(&build).view();
    let join = CuDFHashJoin::try_new(&build_view, &[0], CuDFNullEquality::Unequal)?;

    let probe = make_table(vec![2, 2, 3], vec![15, 30, 35]);
    let probe_view = probe.into_view();

    let mut predicate = CuDFAstExpression::new();
    let build_value = predicate.column_reference(0, CuDFAstTableReference::Left)?;
    let probe_value = predicate.column_reference(0, CuDFAstTableReference::Right)?;
    let five = predicate.literal(int32_scalar(5)?)?;
    let build_plus_five = predicate.binary_operation(CuDFAstOperator::Add, build_value, five)?;
    predicate.binary_operation(CuDFAstOperator::LessEqual, build_plus_five, probe_value)?;

    let build_values_view = CuDFTableView::try_from_column_views(vec![build_view.column(1)?])?;
    let probe_values_view = CuDFTableView::try_from_column_views(vec![probe_view.column(1)?])?;
    let result = join.inner_join_filtered(CuDFFilteredHashJoinArgs {
        probe: &probe_view,
        probe_on: &[0],
        build_conditional: &build_values_view,
        probe_conditional: &probe_values_view,
        predicate: &predicate,
        build_payload: &build_view,
        probe_payload: &probe_view,
        build_out_cols: Some(&[1]),
        probe_out_cols: Some(&[1]),
    })?;

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
    let mut pairs: Vec<_> = (0..batch.num_rows())
        .map(|i| (left_vals.value(i), right_vals.value(i)))
        .collect();
    pairs.sort();
    assert_eq!(pairs, vec![(20, 30), (25, 30), (30, 35)]);
    Ok(())
}

#[test]
fn test_hash_join_filtered_empty_predicate_returns_error() -> Result<(), Box<dyn std::error::Error>>
{
    let build = Arc::new(make_table(vec![2], vec![20]));
    let build_view = Arc::clone(&build).view();
    let join = CuDFHashJoin::try_new(&build_view, &[0], CuDFNullEquality::Unequal)?;

    let probe = make_table(vec![2], vec![25]);
    let probe_view = probe.into_view();
    let predicate = CuDFAstExpression::new();
    let result = join.inner_join_filtered(filtered_join_args(&probe_view, &predicate, &build_view));

    assert!(result
        .err()
        .is_some_and(|err| err.to_string().contains("empty AST predicate")));
    Ok(())
}

#[test]
fn test_hash_join_filtered_records_only_passing_build_rows(
) -> Result<(), Box<dyn std::error::Error>> {
    let build = Arc::new(make_table(vec![1, 2, 3], vec![10, 20, 30]));
    let build_view = Arc::clone(&build).view();
    let mut join = CuDFHashJoin::try_new(&build_view, &[0], CuDFNullEquality::Unequal)?;

    let probe = make_table(vec![2, 3], vec![25, 25]);
    let probe_view = probe.into_view();
    let predicate = value_less_predicate()?;
    let matched = join.inner_join_filtered_and_record_matches(filtered_join_args(
        &probe_view,
        &predicate,
        &build_view,
    ))?;

    let matched_batch = matched.into_view().to_arrow_host()?;
    assert_eq!(int32_values(&matched_batch, 0), vec![20]);
    assert_eq!(int32_values(&matched_batch, 1), vec![25]);

    let unmatched = join.unmatched_build_rows(&build_view, &probe_view, Some(&[1]), Some(&[1]))?;
    let unmatched_batch = unmatched.into_view().to_arrow_host()?;
    let right_vals = unmatched_batch
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(int32_values(&unmatched_batch, 0), vec![10, 30]);
    assert_eq!(right_vals.null_count(), 2);
    Ok(())
}

#[test]
fn test_hash_join_filtered_duplicate_with_passing_match_is_not_unmatched(
) -> Result<(), Box<dyn std::error::Error>> {
    let build = Arc::new(make_table(vec![2], vec![20]));
    let build_view = Arc::clone(&build).view();
    let mut join = CuDFHashJoin::try_new(&build_view, &[0], CuDFNullEquality::Unequal)?;

    let probe = make_table(vec![2, 2], vec![15, 25]);
    let probe_view = probe.into_view();
    let predicate = value_less_predicate()?;
    let matched = join.inner_join_filtered_and_record_matches(filtered_join_args(
        &probe_view,
        &predicate,
        &build_view,
    ))?;
    assert_eq!(matched.num_rows(), 1);

    let unmatched = join.unmatched_build_rows(&build_view, &probe_view, Some(&[1]), Some(&[1]))?;
    assert_eq!(unmatched.num_rows(), 0);
    Ok(())
}

#[test]
fn test_hash_join_filtered_no_predicate_matches_all_build_rows_unmatched(
) -> Result<(), Box<dyn std::error::Error>> {
    let build = Arc::new(make_table(vec![1, 2], vec![10, 20]));
    let build_view = Arc::clone(&build).view();
    let mut join = CuDFHashJoin::try_new(&build_view, &[0], CuDFNullEquality::Unequal)?;

    let probe = make_table(vec![1, 2], vec![5, 15]);
    let probe_view = probe.into_view();
    let predicate = value_less_predicate()?;
    let matched = join.inner_join_filtered_and_record_matches(filtered_join_args(
        &probe_view,
        &predicate,
        &build_view,
    ))?;
    assert_eq!(matched.num_rows(), 0);

    let unmatched = join.unmatched_build_rows(&build_view, &probe_view, Some(&[1]), Some(&[1]))?;
    let unmatched_batch = unmatched.into_view().to_arrow_host()?;
    let right_vals = unmatched_batch
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(int32_values(&unmatched_batch, 0), vec![10, 20]);
    assert_eq!(right_vals.null_count(), 2);
    Ok(())
}

#[test]
fn test_hash_join_filtered_full_probe_outputs_probe_only_rows(
) -> Result<(), Box<dyn std::error::Error>> {
    let build = Arc::new(make_table(vec![1, 2, 2, 3], vec![10, 20, 30, 40]));
    let build_view = Arc::clone(&build).view();
    let mut join = CuDFHashJoin::try_new(&build_view, &[0], CuDFNullEquality::Unequal)?;

    let probe = make_table(vec![2, 2, 4], vec![25, 35, 400]);
    let probe_view = probe.into_view();
    let predicate = value_less_predicate()?;
    let result = join.probe_left_join_filtered_and_record_matches(filtered_join_args(
        &probe_view,
        &predicate,
        &build_view,
    ))?;

    let batch = result.into_view().to_arrow_host()?;
    let mut pairs: Vec<_> = int32_options(&batch, 0)
        .into_iter()
        .zip(int32_options(&batch, 1))
        .collect();
    pairs.sort();
    assert_eq!(
        pairs,
        vec![
            (None, Some(400)),
            (Some(20), Some(25)),
            (Some(20), Some(35)),
            (Some(30), Some(35)),
        ]
    );

    let unmatched = join.unmatched_build_rows(&build_view, &probe_view, Some(&[1]), Some(&[1]))?;
    let unmatched_batch = unmatched.into_view().to_arrow_host()?;
    let right_vals = unmatched_batch
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(int32_values(&unmatched_batch, 0), vec![10, 40]);
    assert_eq!(right_vals.null_count(), 2);
    Ok(())
}

#[test]
fn test_hash_join_records_unmatched_build_rows() -> Result<(), Box<dyn std::error::Error>> {
    let build = Arc::new(make_table(vec![1, 2, 3], vec![10, 20, 30]));
    let build_view = Arc::clone(&build).view();
    let mut join = CuDFHashJoin::try_new(&build_view, &[0], CuDFNullEquality::Unequal)?;

    let probe_a = make_table(vec![2], vec![200]);
    let probe_a_view = probe_a.into_view();
    let inner = join.inner_join_and_record_matches(
        &probe_a_view,
        &[0],
        &build_view,
        &probe_a_view,
        Some(&[1]),
        Some(&[1]),
    )?;
    let inner_batch = inner.into_view().to_arrow_host()?;
    assert_eq!(int32_values(&inner_batch, 0), vec![20]);
    assert_eq!(int32_values(&inner_batch, 1), vec![200]);

    let probe_b = make_table(vec![3, 4], vec![300, 400]);
    let probe_b_view = probe_b.into_view();
    let probe_left = join.probe_left_join_and_record_matches(
        &probe_b_view,
        &[0],
        &build_view,
        &probe_b_view,
        Some(&[1]),
        Some(&[1]),
    )?;
    assert_eq!(probe_left.num_rows(), 2);
    assert_eq!(probe_left.num_columns(), 2);

    let probe_left_batch = probe_left.into_view().to_arrow_host()?;
    let build_vals = probe_left_batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(build_vals.null_count(), 1);
    assert_eq!(int32_values(&probe_left_batch, 1), vec![300, 400]);

    let non_null_build_vals: Vec<_> = (0..build_vals.len())
        .filter(|&i| build_vals.is_valid(i))
        .map(|i| build_vals.value(i))
        .collect();
    assert_eq!(non_null_build_vals, vec![30]);

    let unmatched =
        join.unmatched_build_rows(&build_view, &probe_b_view, Some(&[1]), Some(&[1]))?;
    assert_eq!(unmatched.num_rows(), 1);
    assert_eq!(unmatched.num_columns(), 2);

    let unmatched_batch = unmatched.into_view().to_arrow_host()?;
    let unmatched_probe_vals = unmatched_batch
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(int32_values(&unmatched_batch, 0), vec![10]);
    assert_eq!(unmatched_probe_vals.null_count(), 1);
    Ok(())
}

#[test]
fn test_hash_join_match_mask_handles_duplicate_and_unmatched_probe_rows(
) -> Result<(), Box<dyn std::error::Error>> {
    let build = Arc::new(make_table(vec![1, 2, 3, 4], vec![10, 20, 30, 40]));
    let build_view = Arc::clone(&build).view();
    let mut join = CuDFHashJoin::try_new(&build_view, &[0], CuDFNullEquality::Unequal)?;

    let probe_a = make_table(vec![2, 2], vec![200, 201]);
    let probe_a_view = probe_a.into_view();
    let inner = join.inner_join_and_record_matches(
        &probe_a_view,
        &[0],
        &build_view,
        &probe_a_view,
        Some(&[1]),
        Some(&[1]),
    )?;
    assert_eq!(inner.num_rows(), 2);

    let probe_b = make_table(vec![2, 3, 3, 5], vec![202, 300, 301, 500]);
    let probe_b_view = probe_b.into_view();
    let probe_left = join.probe_left_join_and_record_matches(
        &probe_b_view,
        &[0],
        &build_view,
        &probe_b_view,
        Some(&[1]),
        Some(&[1]),
    )?;
    assert_eq!(probe_left.num_rows(), 4);

    let probe_left_batch = probe_left.into_view().to_arrow_host()?;
    let build_vals = probe_left_batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(build_vals.null_count(), 1);

    let unmatched =
        join.unmatched_build_rows(&build_view, &probe_b_view, Some(&[1]), Some(&[1]))?;
    assert_eq!(unmatched.num_rows(), 2);
    assert_eq!(unmatched.num_columns(), 2);

    let unmatched_batch = unmatched.into_view().to_arrow_host()?;
    let unmatched_probe_vals = unmatched_batch
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(int32_values(&unmatched_batch, 0), vec![10, 40]);
    assert_eq!(unmatched_probe_vals.null_count(), 2);
    Ok(())
}

#[test]
fn test_hash_join_match_mask_handles_all_build_rows_matched(
) -> Result<(), Box<dyn std::error::Error>> {
    let build = Arc::new(make_table(vec![1, 2], vec![10, 20]));
    let build_view = Arc::clone(&build).view();
    let mut join = CuDFHashJoin::try_new(&build_view, &[0], CuDFNullEquality::Unequal)?;

    let probe_a = make_table(vec![1, 1], vec![100, 101]);
    let probe_a_view = probe_a.into_view();
    join.inner_join_and_record_matches(
        &probe_a_view,
        &[0],
        &build_view,
        &probe_a_view,
        Some(&[1]),
        Some(&[1]),
    )?;

    let probe_b = make_table(vec![2, 2], vec![200, 201]);
    let probe_b_view = probe_b.into_view();
    join.inner_join_and_record_matches(
        &probe_b_view,
        &[0],
        &build_view,
        &probe_b_view,
        Some(&[1]),
        Some(&[1]),
    )?;

    let unmatched =
        join.unmatched_build_rows(&build_view, &probe_b_view, Some(&[1]), Some(&[1]))?;
    assert_eq!(unmatched.num_rows(), 0);
    assert_eq!(unmatched.num_columns(), 2);
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

    let batch = result.into_view().to_arrow_host()?;
    let mut keys = int32_values(&batch, 0);
    keys.sort();
    assert_eq!(keys, vec![2, 3]);
    Ok(())
}

#[test]
fn test_left_anti_join() -> Result<(), Box<dyn std::error::Error>> {
    let left = make_table(vec![1, 2, 3, 4], vec![10, 20, 30, 40]);
    let right = make_table(vec![2, 3, 5], vec![200, 300, 500]);

    let result = left_anti_join(&left.into_view(), &right.into_view(), &[0], &[0])?;
    assert_eq!(result.num_rows(), 2); // keys 1 and 4 have no match
    assert_eq!(result.num_columns(), 2); // only left columns

    let batch = result.into_view().to_arrow_host()?;
    let mut keys = int32_values(&batch, 0);
    keys.sort();
    assert_eq!(keys, vec![1, 4]);
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

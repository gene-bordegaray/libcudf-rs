use super::utils::{gather_join_indices, select_cols};
use super::JoinIndexVector;
use crate::device_resource::resource_ref;
use crate::stream::stream_ref;
use crate::{CuDFError, CuDFTable, CuDFTableView};
use libcudf_sys::{ffi, NullEquality, OutOfBoundsPolicy, SetAsBuildTable};
use std::sync::Arc;

/// Perform an inner join on two tables using the specified key columns.
///
/// Columns from both tables are concatenated as `[left_cols | right_cols]`.
pub fn inner_join(
    left: &CuDFTableView,
    right: &CuDFTableView,
    left_on: &[usize],
    right_on: &[usize],
    left_out_cols: Option<&[usize]>,
    right_out_cols: Option<&[usize]>,
) -> Result<CuDFTable, CuDFError> {
    let left_keys = select_cols(left, left_on)?;
    let right_keys = select_cols(right, right_on)?;
    let left_payload = left_out_cols.map(|c| select_cols(left, c)).transpose()?;
    let right_payload = right_out_cols.map(|c| select_cols(right, c)).transpose()?;
    let stream = ffi::get_default_stream();
    let resource = ffi::get_current_device_resource_ref();
    let indices = ffi::inner_join_indices(
        &left_keys,
        &right_keys,
        NullEquality::Equal as i32,
        stream_ref(&stream)?,
        resource_ref(&resource)?,
    )?;
    gather_join_indices(
        indices,
        left_payload.as_ref().unwrap_or_else(|| left.inner()),
        right_payload.as_ref().unwrap_or_else(|| right.inner()),
        OutOfBoundsPolicy::DontCheck,
        OutOfBoundsPolicy::DontCheck,
    )
}

/// Perform a left outer join on two tables.
pub fn left_join(
    left: &CuDFTableView,
    right: &CuDFTableView,
    left_on: &[usize],
    right_on: &[usize],
    left_out_cols: Option<&[usize]>,
    right_out_cols: Option<&[usize]>,
) -> Result<CuDFTable, CuDFError> {
    let left_keys = select_cols(left, left_on)?;
    let right_keys = select_cols(right, right_on)?;
    let left_payload = left_out_cols.map(|c| select_cols(left, c)).transpose()?;
    let right_payload = right_out_cols.map(|c| select_cols(right, c)).transpose()?;
    let stream = ffi::get_default_stream();
    let resource = ffi::get_current_device_resource_ref();
    let indices = ffi::left_join_indices(
        &left_keys,
        &right_keys,
        NullEquality::Equal as i32,
        stream_ref(&stream)?,
        resource_ref(&resource)?,
    )?;
    gather_join_indices(
        indices,
        left_payload.as_ref().unwrap_or_else(|| left.inner()),
        right_payload.as_ref().unwrap_or_else(|| right.inner()),
        OutOfBoundsPolicy::DontCheck,
        OutOfBoundsPolicy::Nullify,
    )
}

/// Perform a full outer join on two tables.
pub fn full_join(
    left: &CuDFTableView,
    right: &CuDFTableView,
    left_on: &[usize],
    right_on: &[usize],
    left_out_cols: Option<&[usize]>,
    right_out_cols: Option<&[usize]>,
) -> Result<CuDFTable, CuDFError> {
    let left_keys = select_cols(left, left_on)?;
    let right_keys = select_cols(right, right_on)?;
    let left_payload = left_out_cols.map(|c| select_cols(left, c)).transpose()?;
    let right_payload = right_out_cols.map(|c| select_cols(right, c)).transpose()?;
    let stream = ffi::get_default_stream();
    let resource = ffi::get_current_device_resource_ref();
    let indices = ffi::full_join_indices(
        &left_keys,
        &right_keys,
        NullEquality::Equal as i32,
        stream_ref(&stream)?,
        resource_ref(&resource)?,
    )?;
    gather_join_indices(
        indices,
        left_payload.as_ref().unwrap_or_else(|| left.inner()),
        right_payload.as_ref().unwrap_or_else(|| right.inner()),
        OutOfBoundsPolicy::Nullify,
        OutOfBoundsPolicy::Nullify,
    )
}

/// Perform a left semi join, returning only matching left rows.
pub fn left_semi_join(
    left: &CuDFTableView,
    right: &CuDFTableView,
    left_on: &[usize],
    right_on: &[usize],
) -> Result<CuDFTable, CuDFError> {
    filtered_left_join(left, right, left_on, right_on, false)
}

/// Perform a left anti join, returning only non-matching left rows.
pub fn left_anti_join(
    left: &CuDFTableView,
    right: &CuDFTableView,
    left_on: &[usize],
    right_on: &[usize],
) -> Result<CuDFTable, CuDFError> {
    filtered_left_join(left, right, left_on, right_on, true)
}

fn filtered_left_join(
    left: &CuDFTableView,
    right: &CuDFTableView,
    left_on: &[usize],
    right_on: &[usize],
    anti: bool,
) -> Result<CuDFTable, CuDFError> {
    let left_keys = select_cols(left, left_on)?;
    let right_keys = select_cols(right, right_on)?;
    let stream = ffi::get_default_stream();
    let resource = ffi::get_current_device_resource_ref();
    let join = ffi::filtered_join_create(
        &right_keys,
        NullEquality::Equal as i32,
        SetAsBuildTable::Right as i32,
        stream_ref(&stream)?,
    )?;
    let inner = if anti {
        ffi::filtered_join_anti_join(
            &join,
            &left_keys,
            stream_ref(&stream)?,
            resource_ref(&resource)?,
        )?
    } else {
        ffi::filtered_join_semi_join(
            &join,
            &left_keys,
            stream_ref(&stream)?,
            resource_ref(&resource)?,
        )?
    };
    let indices = Arc::new(JoinIndexVector::try_from_inner(inner)?);
    let indices_view = indices.view()?;
    CuDFTable::try_from_inner(ffi::gather(
        left.inner(),
        indices_view.inner(),
        OutOfBoundsPolicy::DontCheck as i32,
        stream_ref(&stream)?,
        resource_ref(&resource)?,
    )?)
}

/// Perform a Cartesian product of two tables.
pub fn cross_join(left: &CuDFTableView, right: &CuDFTableView) -> Result<CuDFTable, CuDFError> {
    let stream = ffi::get_default_stream();
    let resource = ffi::get_current_device_resource_ref();
    CuDFTable::try_from_inner(ffi::cross_join(
        left.inner(),
        right.inner(),
        stream_ref(&stream)?,
        resource_ref(&resource)?,
    )?)
}

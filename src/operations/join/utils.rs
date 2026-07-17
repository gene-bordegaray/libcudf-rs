use super::JoinIndexVector;
use crate::data_type::arrow_type_to_cudf_data_type;
use crate::device_resource::resource_ref;
use crate::stream::stream_ref;
use crate::{CuDFColumn, CuDFColumnView, CuDFError, CuDFScalar, CuDFTable, CuDFTableView};
use arrow::array::{Array, BooleanArray, Int32Array, Scalar};
use arrow_schema::DataType;
use cxx::UniquePtr;
use libcudf_sys::{
    ffi, BinaryOperator, DuplicateKeepOption, NanEquality, NullEquality, OutOfBoundsPolicy,
};
use std::sync::Arc;

pub(super) fn null_gather_index() -> i32 {
    ffi::join_no_match()
}

pub(super) fn select_cols(
    view: &CuDFTableView,
    cols: &[usize],
) -> Result<cxx::UniquePtr<ffi::TableView>, CuDFError> {
    let indices = cols
        .iter()
        .map(|&index| {
            i32::try_from(index).map_err(|_| {
                CuDFError::ArrowError(arrow_schema::ArrowError::InvalidArgumentError(format!(
                    "column index {index} exceeds cuDF's column-index range"
                )))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(view.inner().select(&indices)?)
}

pub(super) fn gather_join_output(
    left_payload: &ffi::TableView,
    right_payload: &ffi::TableView,
    left_indices: &ffi::ColumnView,
    right_indices: &ffi::ColumnView,
    left_policy: OutOfBoundsPolicy,
    right_policy: OutOfBoundsPolicy,
) -> Result<CuDFTable, CuDFError> {
    let stream = ffi::get_default_stream();
    let resource = ffi::get_current_device_resource_ref();
    let left = CuDFTable::try_from_inner(ffi::gather(
        left_payload,
        left_indices,
        left_policy as i32,
        stream_ref(&stream)?,
        resource_ref(&resource)?,
    )?)?;
    let right = CuDFTable::try_from_inner(ffi::gather(
        right_payload,
        right_indices,
        right_policy as i32,
        stream_ref(&stream)?,
        resource_ref(&resource)?,
    )?)?;
    let mut columns = left.into_columns()?;
    columns.extend(right.into_columns()?);
    CuDFTable::try_from_columns(columns)
}

pub(super) fn gather_join_indices(
    mut indices: UniquePtr<ffi::JoinIndices>,
    left_payload: &ffi::TableView,
    right_payload: &ffi::TableView,
    left_policy: OutOfBoundsPolicy,
    right_policy: OutOfBoundsPolicy,
) -> Result<CuDFTable, CuDFError> {
    let left_indices = Arc::new(JoinIndexVector::try_from_inner(
        indices.pin_mut().release_left(),
    )?);
    let right_indices = Arc::new(JoinIndexVector::try_from_inner(
        indices.pin_mut().release_right(),
    )?);
    let left_indices_view = Arc::clone(&left_indices).view()?;
    let right_indices_view = Arc::clone(&right_indices).view()?;
    gather_join_output(
        left_payload,
        right_payload,
        left_indices_view.inner(),
        right_indices_view.inner(),
        left_policy,
        right_policy,
    )
}

pub(super) fn gather_hash_join_indices(
    mut indices: UniquePtr<ffi::HashJoinIndices>,
    build_payload: &ffi::TableView,
    probe_payload: &ffi::TableView,
    build_policy: OutOfBoundsPolicy,
    probe_policy: OutOfBoundsPolicy,
) -> Result<(CuDFTable, Arc<JoinIndexVector>), CuDFError> {
    let probe_indices = Arc::new(JoinIndexVector::try_from_inner(
        indices.pin_mut().release_probe(),
    )?);
    let build_indices = Arc::new(JoinIndexVector::try_from_inner(
        indices.pin_mut().release_build(),
    )?);
    let probe_indices_view = Arc::clone(&probe_indices).view()?;
    let build_indices_view = Arc::clone(&build_indices).view()?;
    let result = gather_join_output(
        build_payload,
        probe_payload,
        build_indices_view.inner(),
        probe_indices_view.inner(),
        build_policy,
        probe_policy,
    )?;
    Ok((result, build_indices))
}

pub(super) fn concat_join_outputs(
    first: CuDFTable,
    second: CuDFTable,
) -> Result<CuDFTable, CuDFError> {
    if first.num_rows() == 0 {
        return Ok(second);
    }
    if second.num_rows() == 0 {
        return Ok(first);
    }
    CuDFTable::concat(vec![first.into_view(), second.into_view()])
}

pub(super) fn take_only_column(
    table: CuDFTable,
    context: &'static str,
) -> Result<CuDFColumn, CuDFError> {
    let mut columns = table.into_columns()?.into_iter();
    let column = columns.next().ok_or_else(|| {
        arrow_schema::ArrowError::ComputeError(format!("cuDF returned no column for {context}"))
    })?;
    if columns.next().is_some() {
        return Err(arrow_schema::ArrowError::ComputeError(format!(
            "cuDF returned multiple columns for {context}"
        ))
        .into());
    }
    Ok(column)
}

pub(super) fn distinct_valid_indices(
    indices: CuDFColumnView,
) -> Result<Option<Arc<CuDFColumn>>, CuDFError> {
    if indices.len() == 0 {
        return Ok(None);
    }

    let stream = ffi::get_default_stream();
    let resource = ffi::get_current_device_resource_ref();
    let zero = int32_scalar(0)?;
    let bool_type = bool_data_type()?;
    let valid_mask = Arc::new(CuDFColumn::try_from_inner(
        ffi::binary_operation_column_scalar(
            indices.inner(),
            zero.inner(),
            BinaryOperator::GreaterEqual as i32,
            &bool_type,
            stream_ref(&stream)?,
            resource_ref(&resource)?,
        )?,
    )?);
    let indices_table = CuDFTableView::try_from_column_views(vec![indices])?;
    let valid_indices_table = CuDFTable::try_from_inner(ffi::apply_boolean_mask(
        indices_table.inner(),
        Arc::clone(&valid_mask).view().inner(),
        stream_ref(&stream)?,
        resource_ref(&resource)?,
    )?)?;
    if valid_indices_table.num_rows() == 0 {
        return Ok(None);
    }

    let valid_indices_view = valid_indices_table.into_view();
    let distinct_indices = CuDFTable::try_from_inner(ffi::distinct(
        valid_indices_view.inner(),
        &[0],
        DuplicateKeepOption::KeepAny as i32,
        NullEquality::Equal as i32,
        NanEquality::AllEqual as i32,
        stream_ref(&stream)?,
        resource_ref(&resource)?,
    )?)?;
    if distinct_indices.num_rows() == 0 {
        return Ok(None);
    }

    Ok(Some(Arc::new(take_only_column(
        distinct_indices,
        "distinct join indices",
    )?)))
}

pub(super) fn join_index_sequence(
    size: usize,
    init: i32,
    step: i32,
) -> Result<Arc<CuDFColumn>, CuDFError> {
    let stream = ffi::get_default_stream();
    let resource = ffi::get_current_device_resource_ref();
    let init_array = Int32Array::from(vec![init]);
    let step_array = Int32Array::from(vec![step]);
    let init_scalar = CuDFScalar::try_from_arrow_host(Scalar::new(&init_array))?;
    let step_scalar = CuDFScalar::try_from_arrow_host(Scalar::new(&step_array))?;
    Ok(Arc::new(CuDFColumn::try_from_inner(
        ffi::sequence_with_step(
            crate::errors::usize_to_cudf_size(size, "join index sequence length")?,
            init_scalar.inner(),
            step_scalar.inner(),
            stream_ref(&stream)?,
            resource_ref(&resource)?,
        )?,
    )?))
}

pub(super) fn int32_scalar(value: i32) -> Result<CuDFScalar, CuDFError> {
    let array = Int32Array::from(vec![value]);
    CuDFScalar::try_from_arrow_host(Scalar::new(&array))
}

pub(super) fn bool_scalar(value: bool) -> Result<CuDFScalar, CuDFError> {
    let array = BooleanArray::from(vec![value]);
    CuDFScalar::try_from_arrow_host(Scalar::new(&array))
}

pub(super) fn bool_data_type() -> Result<cxx::UniquePtr<ffi::DataType>, CuDFError> {
    arrow_type_to_cudf_data_type(&DataType::Boolean).ok_or_else(|| {
        arrow_schema::ArrowError::ComputeError("Boolean is not supported by cuDF".into()).into()
    })
}

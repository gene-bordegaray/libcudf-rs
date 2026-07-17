use super::utils::{
    bool_data_type, bool_scalar, distinct_valid_indices, gather_hash_join_indices,
    gather_join_output, join_index_sequence, null_gather_index, select_cols, take_only_column,
};
use super::JoinIndexVector;
use crate::device_resource::resource_ref;
use crate::storage::table_view::TableOwner;
use crate::stream::stream_ref;
use crate::{CuDFColumn, CuDFError, CuDFTable, CuDFTableView};
use cxx::UniquePtr;
use libcudf_sys::{ffi, BinaryOperator, NullEquality, OutOfBoundsPolicy};
use std::sync::Arc;

/// Reusable hash join built from a fixed build-side key table.
///
/// The build-side table is kept alive for the lifetime of this object.
pub struct CuDFHashJoin {
    inner: UniquePtr<ffi::HashJoin>,
    build_rows: usize,
    matched_build_mask: Option<Arc<CuDFColumn>>,
    _build_keepalive: TableOwner,
}

/// Controls whether null join-key values compare equal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CuDFNullEquality {
    /// Null join-key values match other null join-key values.
    Equal,
    /// Null join-key values do not match anything.
    Unequal,
}

impl CuDFNullEquality {
    fn into_sys(self) -> NullEquality {
        match self {
            Self::Equal => NullEquality::Equal,
            Self::Unequal => NullEquality::Unequal,
        }
    }
}

impl CuDFHashJoin {
    /// Build a reusable hash join from the selected build-side key columns.
    pub fn try_new(
        build: &CuDFTableView,
        build_on: &[usize],
        null_equality: CuDFNullEquality,
    ) -> Result<Self, CuDFError> {
        let build_keys = select_cols(build, build_on)?;
        let stream = ffi::get_default_stream();
        let inner = ffi::hash_join_create(
            &build_keys,
            null_equality.into_sys() as i32,
            stream_ref(&stream)?,
        )?;
        Ok(Self {
            inner,
            build_rows: build.num_rows(),
            matched_build_mask: None,
            _build_keepalive: build.owner().clone(),
        })
    }

    /// Probe this hash join and gather matching payload rows.
    ///
    /// Output columns are concatenated as `[build_cols | probe_cols]`.
    pub fn inner_join(
        &self,
        probe: &CuDFTableView,
        probe_on: &[usize],
        build_payload: &CuDFTableView,
        probe_payload: &CuDFTableView,
        build_out_cols: Option<&[usize]>,
        probe_out_cols: Option<&[usize]>,
    ) -> Result<CuDFTable, CuDFError> {
        let probe_keys = select_cols(probe, probe_on)?;
        let selected_build_payload = build_out_cols
            .map(|c| select_cols(build_payload, c))
            .transpose()?;
        let selected_probe_payload = probe_out_cols
            .map(|c| select_cols(probe_payload, c))
            .transpose()?;
        let indices = self.inner_join_indices(&probe_keys)?;
        let (result, _) = gather_hash_join_indices(
            indices,
            selected_build_payload
                .as_ref()
                .unwrap_or_else(|| build_payload.inner()),
            selected_probe_payload
                .as_ref()
                .unwrap_or_else(|| probe_payload.inner()),
            OutOfBoundsPolicy::DontCheck,
            OutOfBoundsPolicy::DontCheck,
        )?;
        Ok(result)
    }

    /// Probe this hash join, record matched build rows, and emit inner-join rows.
    pub fn inner_join_and_record_matches(
        &mut self,
        probe: &CuDFTableView,
        probe_on: &[usize],
        build_payload: &CuDFTableView,
        probe_payload: &CuDFTableView,
        build_out_cols: Option<&[usize]>,
        probe_out_cols: Option<&[usize]>,
    ) -> Result<CuDFTable, CuDFError> {
        let probe_keys = select_cols(probe, probe_on)?;
        let selected_build_payload = build_out_cols
            .map(|c| select_cols(build_payload, c))
            .transpose()?;
        let selected_probe_payload = probe_out_cols
            .map(|c| select_cols(probe_payload, c))
            .transpose()?;
        let indices = self.inner_join_indices(&probe_keys)?;
        let (result, build_indices) = gather_hash_join_indices(
            indices,
            selected_build_payload
                .as_ref()
                .unwrap_or_else(|| build_payload.inner()),
            selected_probe_payload
                .as_ref()
                .unwrap_or_else(|| probe_payload.inner()),
            OutOfBoundsPolicy::DontCheck,
            OutOfBoundsPolicy::DontCheck,
        )?;
        self.record_matched_build_indices(build_indices)?;
        Ok(result)
    }

    /// Probe this hash join preserving probe rows and record matched build rows.
    ///
    /// Output columns are concatenated as `[build_cols | probe_cols]`.
    pub fn probe_left_join_and_record_matches(
        &mut self,
        probe: &CuDFTableView,
        probe_on: &[usize],
        build_payload: &CuDFTableView,
        probe_payload: &CuDFTableView,
        build_out_cols: Option<&[usize]>,
        probe_out_cols: Option<&[usize]>,
    ) -> Result<CuDFTable, CuDFError> {
        let probe_keys = select_cols(probe, probe_on)?;
        let selected_build_payload = build_out_cols
            .map(|c| select_cols(build_payload, c))
            .transpose()?;
        let selected_probe_payload = probe_out_cols
            .map(|c| select_cols(probe_payload, c))
            .transpose()?;
        let indices = self.left_join_indices(&probe_keys)?;
        let (result, build_indices) = gather_hash_join_indices(
            indices,
            selected_build_payload
                .as_ref()
                .unwrap_or_else(|| build_payload.inner()),
            selected_probe_payload
                .as_ref()
                .unwrap_or_else(|| probe_payload.inner()),
            OutOfBoundsPolicy::Nullify,
            OutOfBoundsPolicy::DontCheck,
        )?;
        self.record_matched_build_indices(build_indices)?;
        Ok(result)
    }

    /// Gather build rows not matched by previous recorded probes.
    pub fn unmatched_build_rows(
        &self,
        build_payload: &CuDFTableView,
        probe_payload: &CuDFTableView,
        build_out_cols: Option<&[usize]>,
        probe_out_cols: Option<&[usize]>,
    ) -> Result<CuDFTable, CuDFError> {
        let selected_build_payload = build_out_cols
            .map(|c| select_cols(build_payload, c))
            .transpose()?;
        let selected_probe_payload = probe_out_cols
            .map(|c| select_cols(probe_payload, c))
            .transpose()?;
        let unmatched_build_indices = self.unmatched_build_indices()?;
        let null_probe_indices =
            join_index_sequence(unmatched_build_indices.len(), null_gather_index(), 0)?;
        let unmatched_build_indices_view = Arc::clone(&unmatched_build_indices).view();
        let null_probe_indices_view = Arc::clone(&null_probe_indices).view();

        gather_join_output(
            selected_build_payload
                .as_ref()
                .unwrap_or_else(|| build_payload.inner()),
            selected_probe_payload
                .as_ref()
                .unwrap_or_else(|| probe_payload.inner()),
            unmatched_build_indices_view.inner(),
            null_probe_indices_view.inner(),
            OutOfBoundsPolicy::DontCheck,
            OutOfBoundsPolicy::Nullify,
        )
    }

    pub(super) fn inner_join_indices(
        &self,
        probe_keys: &ffi::TableView,
    ) -> Result<UniquePtr<ffi::HashJoinIndices>, CuDFError> {
        let stream = ffi::get_default_stream();
        let resource = ffi::get_current_device_resource_ref();
        Ok(ffi::hash_join_inner_join_indices(
            &self.inner,
            probe_keys,
            stream_ref(&stream)?,
            resource_ref(&resource)?,
        )?)
    }

    fn left_join_indices(
        &self,
        probe_keys: &ffi::TableView,
    ) -> Result<UniquePtr<ffi::HashJoinIndices>, CuDFError> {
        let stream = ffi::get_default_stream();
        let resource = ffi::get_current_device_resource_ref();
        Ok(ffi::hash_join_left_join_indices(
            &self.inner,
            probe_keys,
            stream_ref(&stream)?,
            resource_ref(&resource)?,
        )?)
    }

    pub(super) fn record_matched_build_indices(
        &mut self,
        build_indices: Arc<JoinIndexVector>,
    ) -> Result<(), CuDFError> {
        if self.build_rows == 0 || build_indices.len() == 0 {
            return Ok(());
        }

        let Some(distinct_indices) = distinct_valid_indices(Arc::clone(&build_indices).view()?)?
        else {
            return Ok(());
        };

        let stream = ffi::get_default_stream();
        let resource = ffi::get_current_device_resource_ref();
        let matched_build_mask = match &self.matched_build_mask {
            Some(mask) => Arc::clone(mask),
            None => {
                let false_scalar = bool_scalar(false)?;
                let mask = Arc::new(CuDFColumn::try_from_inner(ffi::make_column_from_scalar(
                    false_scalar.inner(),
                    crate::errors::usize_to_cudf_size(self.build_rows, "join build row count")?,
                    stream_ref(&stream)?,
                    resource_ref(&resource)?,
                )?)?);
                self.matched_build_mask = Some(Arc::clone(&mask));
                mask
            }
        };

        let scatter_indices_view = Arc::clone(&distinct_indices).view();
        let true_scalar = bool_scalar(true)?;
        let source = [true_scalar.inner().as_ptr()];
        let target = CuDFTableView::try_from_column_views(vec![matched_build_mask.view()])?;
        let updated = CuDFTable::try_from_inner(ffi::scatter_scalars(
            &source,
            scatter_indices_view.inner(),
            target.inner(),
            stream_ref(&stream)?,
            resource_ref(&resource)?,
        )?)?;
        self.matched_build_mask = Some(Arc::new(take_only_column(
            updated,
            "updated matched-build mask",
        )?));
        Ok(())
    }

    fn unmatched_build_indices(&self) -> Result<Arc<CuDFColumn>, CuDFError> {
        let all_build_indices = join_index_sequence(self.build_rows, 0, 1)?;
        let Some(matched_build_mask) = &self.matched_build_mask else {
            return Ok(all_build_indices);
        };

        let stream = ffi::get_default_stream();
        let resource = ffi::get_current_device_resource_ref();
        let false_scalar = bool_scalar(false)?;
        let bool_type = bool_data_type()?;
        let unmatched_mask = Arc::new(CuDFColumn::try_from_inner(
            ffi::binary_operation_column_scalar(
                Arc::clone(matched_build_mask).view().inner(),
                false_scalar.inner(),
                BinaryOperator::Equal as i32,
                &bool_type,
                stream_ref(&stream)?,
                resource_ref(&resource)?,
            )?,
        )?);
        let all_build_table =
            CuDFTableView::try_from_column_views(vec![Arc::clone(&all_build_indices).view()])?;
        let unmatched_table = CuDFTable::try_from_inner(ffi::apply_boolean_mask(
            all_build_table.inner(),
            Arc::clone(&unmatched_mask).view().inner(),
            stream_ref(&stream)?,
            resource_ref(&resource)?,
        )?)?;
        Ok(Arc::new(take_only_column(
            unmatched_table,
            "unmatched build indices",
        )?))
    }
}

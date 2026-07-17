use super::utils::{
    bool_data_type, bool_scalar, concat_join_outputs, distinct_valid_indices, gather_join_output,
    join_index_sequence, null_gather_index, select_cols, take_only_column,
};
use super::{CuDFFilteredHashJoinArgs, CuDFHashJoin, JoinIndexVector};
use crate::device_resource::resource_ref;
use crate::stream::stream_ref;
use crate::{CuDFAstExpression, CuDFColumn, CuDFError, CuDFTable, CuDFTableView};
use cxx::UniquePtr;
use libcudf_sys::{ffi, BinaryOperator, JoinKind, OutOfBoundsPolicy};
use std::sync::Arc;

struct FilteredHashJoinIndicesArgs<'a> {
    build_conditional: &'a ffi::TableView,
    probe_conditional: &'a ffi::TableView,
    predicate: &'a CuDFAstExpression,
    join_kind: JoinKind,
    build_payload: &'a ffi::TableView,
    probe_payload: &'a ffi::TableView,
    build_policy: OutOfBoundsPolicy,
    probe_policy: OutOfBoundsPolicy,
}

fn gather_filtered_hash_join_indices(
    mut indices: UniquePtr<ffi::HashJoinIndices>,
    args: FilteredHashJoinIndicesArgs<'_>,
) -> Result<(CuDFTable, Arc<JoinIndexVector>, Arc<JoinIndexVector>), CuDFError> {
    // Hash join returns probe/build maps. cuDF filter_join_indices expects
    // left/right maps, so pass build as left and probe as right to preserve
    // the public `[build_cols | probe_cols]` output order.
    let probe_indices = Arc::new(JoinIndexVector::try_from_inner(
        indices.pin_mut().release_probe(),
    )?);
    let build_indices = Arc::new(JoinIndexVector::try_from_inner(
        indices.pin_mut().release_build(),
    )?);
    let stream = ffi::get_default_stream();
    let resource = ffi::get_current_device_resource_ref();
    let mut filtered_indices = ffi::filter_join_indices(
        args.build_conditional,
        args.probe_conditional,
        build_indices.as_sys()?,
        probe_indices.as_sys()?,
        args.predicate.inner(),
        args.join_kind as i32,
        stream_ref(&stream)?,
        resource_ref(&resource)?,
    )?;
    let filtered_build_indices = Arc::new(JoinIndexVector::try_from_inner(
        filtered_indices.pin_mut().release_left(),
    )?);
    let filtered_probe_indices = Arc::new(JoinIndexVector::try_from_inner(
        filtered_indices.pin_mut().release_right(),
    )?);
    let filtered_build_indices_view = Arc::clone(&filtered_build_indices).view()?;
    let filtered_probe_indices_view = Arc::clone(&filtered_probe_indices).view()?;
    let result = gather_join_output(
        args.build_payload,
        args.probe_payload,
        filtered_build_indices_view.inner(),
        filtered_probe_indices_view.inner(),
        args.build_policy,
        args.probe_policy,
    )?;
    Ok((result, filtered_build_indices, filtered_probe_indices))
}

fn matched_row_mask(
    row_count: usize,
    matched_indices: Arc<JoinIndexVector>,
) -> Result<Arc<CuDFColumn>, CuDFError> {
    let stream = ffi::get_default_stream();
    let resource = ffi::get_current_device_resource_ref();
    let false_scalar = bool_scalar(false)?;
    let mask = Arc::new(CuDFColumn::try_from_inner(ffi::make_column_from_scalar(
        false_scalar.inner(),
        crate::errors::usize_to_cudf_size(row_count, "join row count")?,
        stream_ref(&stream)?,
        resource_ref(&resource)?,
    )?)?);
    let Some(distinct_indices) = distinct_valid_indices(Arc::clone(&matched_indices).view()?)?
    else {
        return Ok(mask);
    };

    let distinct_indices_view = Arc::clone(&distinct_indices).view();
    let true_scalar = bool_scalar(true)?;
    let source = [true_scalar.inner().as_ptr()];
    let target = CuDFTableView::try_from_column_views(vec![Arc::clone(&mask).view()])?;
    let updated = CuDFTable::try_from_inner(ffi::scatter_scalars(
        &source,
        distinct_indices_view.inner(),
        target.inner(),
        stream_ref(&stream)?,
        resource_ref(&resource)?,
    )?)?;
    Ok(Arc::new(take_only_column(updated, "matched-row mask")?))
}

fn unmatched_indices_from_matches(
    row_count: usize,
    matched_indices: Arc<JoinIndexVector>,
) -> Result<Arc<CuDFColumn>, CuDFError> {
    let stream = ffi::get_default_stream();
    let resource = ffi::get_current_device_resource_ref();
    let all_indices = join_index_sequence(row_count, 0, 1)?;
    let matched_mask = matched_row_mask(row_count, matched_indices)?;
    let false_scalar = bool_scalar(false)?;
    let bool_type = bool_data_type()?;
    let unmatched_mask = Arc::new(CuDFColumn::try_from_inner(
        ffi::binary_operation_column_scalar(
            Arc::clone(&matched_mask).view().inner(),
            false_scalar.inner(),
            BinaryOperator::Equal as i32,
            &bool_type,
            stream_ref(&stream)?,
            resource_ref(&resource)?,
        )?,
    )?);
    let all_indices_table =
        CuDFTableView::try_from_column_views(vec![Arc::clone(&all_indices).view()])?;
    let unmatched_table = CuDFTable::try_from_inner(ffi::apply_boolean_mask(
        all_indices_table.inner(),
        Arc::clone(&unmatched_mask).view().inner(),
        stream_ref(&stream)?,
        resource_ref(&resource)?,
    )?)?;
    Ok(Arc::new(take_only_column(
        unmatched_table,
        "unmatched join indices",
    )?))
}

impl CuDFHashJoin {
    /// Probe this hash join, filter equality matches with an AST predicate, and gather payload rows.
    ///
    /// Output columns are concatenated as `[build_cols | probe_cols]`.
    ///
    /// `build_conditional` and `probe_conditional` are the tables referenced by
    /// `predicate`; AST `Left` column references read from `build_conditional`
    /// and AST `Right` column references read from `probe_conditional`.
    ///
    /// `build_payload` and `probe_payload` are the tables gathered into the
    /// output. Use `build_out_cols` and `probe_out_cols` to gather only selected
    /// payload columns.
    ///
    /// # Errors
    ///
    /// Returns an error if probing, predicate filtering, or payload gathering
    /// fails in cuDF.
    pub fn inner_join_filtered(
        &self,
        args: CuDFFilteredHashJoinArgs<'_>,
    ) -> Result<CuDFTable, CuDFError> {
        let probe_keys = select_cols(args.probe, args.probe_on)?;
        let selected_build_payload = args
            .build_out_cols
            .map(|c| select_cols(args.build_payload, c))
            .transpose()?;
        let selected_probe_payload = args
            .probe_out_cols
            .map(|c| select_cols(args.probe_payload, c))
            .transpose()?;
        let indices = self.inner_join_indices(&probe_keys)?;
        let (result, _, _) = gather_filtered_hash_join_indices(
            indices,
            FilteredHashJoinIndicesArgs {
                build_conditional: args.build_conditional.inner(),
                probe_conditional: args.probe_conditional.inner(),
                predicate: args.predicate,
                join_kind: JoinKind::Inner,
                build_payload: selected_build_payload
                    .as_ref()
                    .unwrap_or_else(|| args.build_payload.inner()),
                probe_payload: selected_probe_payload
                    .as_ref()
                    .unwrap_or_else(|| args.probe_payload.inner()),
                build_policy: OutOfBoundsPolicy::DontCheck,
                probe_policy: OutOfBoundsPolicy::DontCheck,
            },
        )?;
        Ok(result)
    }

    /// Probe this hash join, filter equality matches with an AST predicate,
    /// record matching build rows, and emit passing inner-join rows.
    ///
    /// This is the filtered equivalent of [`CuDFHashJoin::inner_join_and_record_matches`].
    /// Only rows with a predicate-passing partner are recorded as matched.
    ///
    /// # Errors
    ///
    /// Returns an error if probing, predicate filtering, payload gathering, or
    /// match-mask updates fail in cuDF.
    pub fn inner_join_filtered_and_record_matches(
        &mut self,
        args: CuDFFilteredHashJoinArgs<'_>,
    ) -> Result<CuDFTable, CuDFError> {
        let probe_keys = select_cols(args.probe, args.probe_on)?;
        let selected_build_payload = args
            .build_out_cols
            .map(|c| select_cols(args.build_payload, c))
            .transpose()?;
        let selected_probe_payload = args
            .probe_out_cols
            .map(|c| select_cols(args.probe_payload, c))
            .transpose()?;
        let indices = self.inner_join_indices(&probe_keys)?;
        let (result, build_indices, _) = gather_filtered_hash_join_indices(
            indices,
            FilteredHashJoinIndicesArgs {
                build_conditional: args.build_conditional.inner(),
                probe_conditional: args.probe_conditional.inner(),
                predicate: args.predicate,
                join_kind: JoinKind::Inner,
                build_payload: selected_build_payload
                    .as_ref()
                    .unwrap_or_else(|| args.build_payload.inner()),
                probe_payload: selected_probe_payload
                    .as_ref()
                    .unwrap_or_else(|| args.probe_payload.inner()),
                build_policy: OutOfBoundsPolicy::DontCheck,
                probe_policy: OutOfBoundsPolicy::DontCheck,
            },
        )?;
        self.record_matched_build_indices(build_indices)?;
        Ok(result)
    }

    /// Probe this hash join for filtered full-join streaming.
    ///
    /// Emits predicate-passing matched rows plus probe rows that have no
    /// predicate-passing build partner in this probe batch. Call
    /// [`CuDFHashJoin::unmatched_build_rows`] after all probe batches to emit
    /// build rows with no passing partner.
    ///
    /// # Errors
    ///
    /// Returns an error if probing, predicate filtering, payload gathering, or
    /// match-mask updates fail in cuDF.
    pub fn probe_left_join_filtered_and_record_matches(
        &mut self,
        args: CuDFFilteredHashJoinArgs<'_>,
    ) -> Result<CuDFTable, CuDFError> {
        let probe_keys = select_cols(args.probe, args.probe_on)?;
        let selected_build_payload = args
            .build_out_cols
            .map(|c| select_cols(args.build_payload, c))
            .transpose()?;
        let selected_probe_payload = args
            .probe_out_cols
            .map(|c| select_cols(args.probe_payload, c))
            .transpose()?;
        let build_payload = selected_build_payload
            .as_ref()
            .unwrap_or_else(|| args.build_payload.inner());
        let probe_payload = selected_probe_payload
            .as_ref()
            .unwrap_or_else(|| args.probe_payload.inner());
        let indices = self.inner_join_indices(&probe_keys)?;
        let (matched, build_indices, probe_indices) = gather_filtered_hash_join_indices(
            indices,
            FilteredHashJoinIndicesArgs {
                build_conditional: args.build_conditional.inner(),
                probe_conditional: args.probe_conditional.inner(),
                predicate: args.predicate,
                join_kind: JoinKind::Inner,
                build_payload,
                probe_payload,
                build_policy: OutOfBoundsPolicy::DontCheck,
                probe_policy: OutOfBoundsPolicy::DontCheck,
            },
        )?;
        self.record_matched_build_indices(build_indices)?;

        let unmatched_probe_indices =
            unmatched_indices_from_matches(args.probe.num_rows(), probe_indices)?;
        let null_build_indices =
            join_index_sequence(unmatched_probe_indices.len(), null_gather_index(), 0)?;
        let null_build_indices_view = Arc::clone(&null_build_indices).view();
        let unmatched_probe_indices_view = Arc::clone(&unmatched_probe_indices).view();
        let probe_only = gather_join_output(
            build_payload,
            probe_payload,
            null_build_indices_view.inner(),
            unmatched_probe_indices_view.inner(),
            OutOfBoundsPolicy::Nullify,
            OutOfBoundsPolicy::DontCheck,
        )?;

        concat_join_outputs(matched, probe_only)
    }
}

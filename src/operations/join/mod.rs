use crate::{CuDFAstExpression, CuDFTableView};

mod filtered;
mod hash;
mod indices;
mod one_shot;
mod utils;

pub use hash::{CuDFHashJoin, CuDFNullEquality};
pub(crate) use indices::JoinIndexVector;
pub use one_shot::{cross_join, full_join, inner_join, left_anti_join, left_join, left_semi_join};

/// Arguments for probing a reusable hash join with an AST predicate.
///
/// The output columns are gathered from `build_payload` and `probe_payload`.
/// AST `Left` references read from `build_conditional`; AST `Right` references
/// read from `probe_conditional`.
#[derive(Clone, Copy)]
pub struct CuDFFilteredHashJoinArgs<'a> {
    /// Probe-side table used for equality matching.
    pub probe: &'a CuDFTableView,
    /// Probe-side equality key columns.
    pub probe_on: &'a [usize],
    /// Build-side table referenced by AST `Left` columns.
    pub build_conditional: &'a CuDFTableView,
    /// Probe-side table referenced by AST `Right` columns.
    pub probe_conditional: &'a CuDFTableView,
    /// Predicate evaluated on equality-match pairs.
    pub predicate: &'a CuDFAstExpression,
    /// Build-side table gathered into the output.
    pub build_payload: &'a CuDFTableView,
    /// Probe-side table gathered into the output.
    pub probe_payload: &'a CuDFTableView,
    /// Optional build payload columns to gather.
    pub build_out_cols: Option<&'a [usize]>,
    /// Optional probe payload columns to gather.
    pub probe_out_cols: Option<&'a [usize]>,
}

#[cfg(test)]
mod tests;

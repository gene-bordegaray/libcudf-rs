use crate::deferred_operation::operation_impl::CuDFOperationImpl;
use crate::execution_policy;
use crate::keep_alive::CuDFKeepAlive;
use crate::stream_readiness::CuDFStreamDependency;
use crate::table_view::CuDFTableView;
use crate::{
    CuDFColumn, CuDFColumnView, CuDFError, CuDFExecutionContext, CuDFOperation, CuDFTable,
};
use arrow::array::Array;
use arrow_schema::ArrowError;
use cxx::UniquePtr;
use libcudf_sys::ffi::{
    self, aggregation_request_create, aggregation_requests_create, make_count_aggregation_groupby,
    make_max_aggregation_groupby, make_mean_aggregation_groupby, make_median_aggregation_groupby,
    make_min_aggregation_groupby, make_nunique_aggregation_groupby, make_std_aggregation_groupby,
    make_sum_aggregation_groupby, make_variance_aggregation_groupby,
};
use libcudf_sys::{NullPolicy, Sorted};
use std::sync::{Arc, Mutex};

/// A reusable cuDF group-by built from key columns.
///
/// Create this from a [`CuDFTableView`] and execute one or more grouped
/// aggregations with [`aggregate`](Self::aggregate).
#[derive(Clone)]
pub struct CuDFGroupBy {
    inner: Arc<CuDFGroupByInner>,
}

struct CuDFGroupByInner {
    // cuDF's groupby is built from `keys`; drop it before the key view/storage.
    inner: UniquePtr<ffi::GroupBy>,
    // Retain the last aggregate stream until the groupby handle is destroyed.
    destruction_dependency: Mutex<Option<CuDFStreamDependency>>,
    keys: CuDFTableView,
}

impl CuDFGroupBy {
    /// Create a group-by from key columns.
    ///
    /// Rows with equal key values are placed in the same group.
    pub fn from_table_view(view: CuDFTableView) -> Self {
        let column_order: &[i32] = &[];
        let null_precedence: &[i32] = &[];
        let inner = ffi::groupby_create(
            view.inner(),
            NullPolicy::Exclude as i32,
            Sorted::No as i32,
            column_order,
            null_precedence,
        );
        Self {
            inner: Arc::new(CuDFGroupByInner {
                inner,
                destruction_dependency: Mutex::new(None),
                keys: view,
            }),
        }
    }

    /// Create a deferred grouped aggregation.
    ///
    /// Each request supplies one value column and one or more aggregations to
    /// compute for that column.
    ///
    /// # Errors
    ///
    /// Execution returns an error if there are no requests, a request has no
    /// aggregations, a value column length does not match the key row count, or
    /// cuDF cannot compute the grouped result.
    pub fn aggregate<I>(&self, requests: I) -> GroupByAggregate<'_>
    where
        I: IntoIterator<Item = GroupByRequest>,
    {
        GroupByAggregate {
            group_by: self,
            requests: requests.into_iter().collect(),
        }
    }
}

/// Deferred grouped aggregation.
///
/// Created by [`CuDFGroupBy::aggregate`]. Execution waits for the group keys
/// and all requested value columns, then returns unique group keys and
/// request-ordered aggregation columns.
///
/// # Errors
///
/// Execution returns an error if request validation fails or cuDF cannot
/// compute the grouped result.
pub struct GroupByAggregate<'a> {
    group_by: &'a CuDFGroupBy,
    requests: Vec<GroupByRequest>,
}

impl CuDFOperation for GroupByAggregate<'_> {
    type Output = CuDFGroupByResult;
}

impl CuDFOperationImpl for GroupByAggregate<'_> {
    fn execute_on_context(
        self,
        ctx: &CuDFExecutionContext,
    ) -> crate::Result<<Self as CuDFOperation>::Output> {
        validate_group_by_requests(&self.group_by.inner.keys, &self.requests)?;

        let mut launch = execution_policy::launch(ctx)?;
        launch.wait_table(&self.group_by.inner.keys)?;
        launch.keep_alive(CuDFKeepAlive::GroupBy {
            _group_by: CuDFGroupBy::clone(self.group_by),
        });

        let mut requests_inner = aggregation_requests_create();
        for request in self.requests {
            launch.wait_column(&request.values)?;
            let mut inner = aggregation_request_create(request.values.inner());
            for aggregation in request.aggregations {
                inner.pin_mut().add(aggregation.into_ffi());
            }
            requests_inner.pin_mut().add(inner);
        }

        let mut gby_result = self.group_by.inner.inner.aggregate(
            &requests_inner,
            launch.stream()?,
            launch.resource(),
        )?;
        let keys = gby_result.pin_mut().release_keys();

        let mut released_columns = Vec::with_capacity(gby_result.len());
        for i in 0..gby_result.len() {
            let mut released_result = gby_result.pin_mut().release_result(i);
            let mut request_columns = Vec::with_capacity(released_result.len());
            for j in 0..released_result.len() {
                request_columns.push(released_result.pin_mut().release(j));
            }

            released_columns.push(request_columns)
        }

        let groupby_dependency = launch.record_stream_dependency(Vec::new())?;
        *self
            .group_by
            .inner
            .destruction_dependency
            .lock()
            .map_err(|_| invalid_argument("group-by dependency lock poisoned"))? =
            Some(groupby_dependency);

        let dependency = launch.into_stream_dependency()?;
        let keys = CuDFTable::from_inner(keys).with_stream_readiness(dependency.clone());
        let mut columns = Vec::with_capacity(released_columns.len());
        for released_result in released_columns {
            let mut request_columns = Vec::with_capacity(released_result.len());
            for col in released_result {
                request_columns
                    .push(CuDFColumn::from_inner(col).with_stream_readiness(dependency.clone()));
            }
            columns.push(request_columns);
        }
        Ok(CuDFGroupByResult { keys, columns })
    }
}

/// Result of a grouped aggregation.
pub struct CuDFGroupByResult {
    keys: CuDFTable,
    columns: Vec<Vec<CuDFColumn>>,
}

impl CuDFGroupByResult {
    /// Unique group keys.
    pub fn keys(&self) -> &CuDFTable {
        &self.keys
    }

    /// Aggregation columns grouped by request, then by aggregation order.
    pub fn columns(&self) -> &[Vec<CuDFColumn>] {
        &self.columns
    }

    /// Aggregation columns for one request, in aggregation order.
    pub fn columns_for_request(&self, request: usize) -> Option<&[CuDFColumn]> {
        self.columns.get(request).map(Vec::as_slice)
    }

    /// Return one aggregation column by request index and aggregation index.
    pub fn column(&self, request: usize, aggregation: usize) -> Option<&CuDFColumn> {
        self.columns
            .get(request)
            .and_then(|columns| columns.get(aggregation))
    }

    /// Consume this result into its key table and grouped aggregation columns.
    pub fn into_parts(self) -> (CuDFTable, Vec<Vec<CuDFColumn>>) {
        (self.keys, self.columns)
    }

    /// Consume this result into its key table and request-ordered columns.
    pub fn into_flat_columns(self) -> (CuDFTable, Vec<CuDFColumn>) {
        let columns = self.columns.into_iter().flatten().collect();
        (self.keys, columns)
    }
}

/// A value column and aggregations for a group-by operation.
///
/// Multiple aggregations can be added to compute several results for the same
/// value column.
#[derive(Clone)]
pub struct GroupByRequest {
    values: CuDFColumnView,
    aggregations: Vec<Aggregation>,
}

impl GroupByRequest {
    /// Create a request for a value column.
    ///
    /// Each value is grouped by the key row at the same index.
    pub fn new(values: CuDFColumnView) -> Self {
        Self {
            values,
            aggregations: Vec::new(),
        }
    }

    /// Add an aggregation and return the request.
    ///
    /// Use this for builder-style request construction.
    #[must_use]
    pub fn with(mut self, aggregation: Aggregation) -> Self {
        self.aggregations.push(aggregation);
        self
    }
}

/// Aggregation to apply to grouped values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Aggregation {
    /// Sum.
    Sum,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
    /// Mean.
    Mean,
    /// Count non-null values.
    Count,
    /// Count all values, including nulls.
    CountAll,
    /// Variance with delta degrees of freedom.
    ///
    /// - `ddof = 0`: Population variance
    /// - `ddof = 1`: Sample variance
    Variance { ddof: i32 },
    /// Standard deviation with delta degrees of freedom.
    ///
    /// - `ddof = 0`: Population std dev
    /// - `ddof = 1`: Sample std dev
    Std { ddof: i32 },
    /// Count distinct non-null values.
    NUnique,
    /// Count distinct values, including nulls.
    NUniqueAll,
    /// Median.
    Median,
}

impl Aggregation {
    fn into_ffi(self) -> UniquePtr<ffi::GroupByAggregation> {
        use Aggregation::*;
        match self {
            Sum => make_sum_aggregation_groupby(),
            Min => make_min_aggregation_groupby(),
            Max => make_max_aggregation_groupby(),
            Mean => make_mean_aggregation_groupby(),
            Count => make_count_aggregation_groupby(NullPolicy::Exclude as i32),
            CountAll => make_count_aggregation_groupby(NullPolicy::Include as i32),
            Variance { ddof } => make_variance_aggregation_groupby(ddof),
            Std { ddof } => make_std_aggregation_groupby(ddof),
            NUnique => make_nunique_aggregation_groupby(NullPolicy::Exclude as i32),
            NUniqueAll => make_nunique_aggregation_groupby(NullPolicy::Include as i32),
            Median => make_median_aggregation_groupby(),
        }
    }
}

fn validate_group_by_requests(
    keys: &CuDFTableView,
    requests: &[GroupByRequest],
) -> crate::Result<()> {
    if requests.is_empty() {
        return Err(invalid_argument(
            "group-by aggregate requires at least one request",
        ));
    }

    for request in requests {
        if request.aggregations.is_empty() {
            return Err(invalid_argument(
                "group-by request must contain at least one aggregation",
            ));
        }
        if request.values.len() != keys.num_rows() {
            return Err(invalid_argument(format!(
                "group-by request value length {} does not match key row count {}",
                request.values.len(),
                keys.num_rows()
            )));
        }
    }
    Ok(())
}

fn invalid_argument(message: impl Into<String>) -> CuDFError {
    CuDFError::ArrowError(ArrowError::InvalidArgumentError(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::Result;
    use arrow::array::{make_array, Array, Float64Array, Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    #[test]
    fn test_sum_integers() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // SUM([1, 2, 3, 4, 5]) = 15
        let values = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let result = run_single_group_agg::<Int64Array>(&values, Aggregation::Sum)?;
        assert_eq!(result.value(0), 15);
        Ok(())
    }

    #[test]
    fn test_sum_with_nulls() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // SUM([1, NULL, 3, NULL, 5]) = 9
        let values = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5)]);
        let result = run_single_group_agg::<Int64Array>(&values, Aggregation::Sum)?;
        assert_eq!(result.value(0), 9);
        Ok(())
    }

    #[test]
    fn test_min() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // MIN([5, 2, 8, 1, 9]) = 1
        let values = Int32Array::from(vec![5, 2, 8, 1, 9]);
        let result = run_single_group_agg::<Int32Array>(&values, Aggregation::Min)?;
        assert_eq!(result.value(0), 1);
        Ok(())
    }

    #[test]
    fn test_max() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // MAX([5, 2, 8, 1, 9]) = 9
        let values = Int32Array::from(vec![5, 2, 8, 1, 9]);
        let result = run_single_group_agg::<Int32Array>(&values, Aggregation::Max)?;
        assert_eq!(result.value(0), 9);
        Ok(())
    }

    #[test]
    fn test_mean() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // MEAN([10, 20, 30, 40, 50]) = 30
        let values = Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let result = run_single_group_agg::<Float64Array>(&values, Aggregation::Mean)?;
        assert!((result.value(0) - 30.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_count() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // COUNT([1, 2, 3, 4, 5]) = 5
        let values = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let result = run_single_group_agg::<Int32Array>(&values, Aggregation::Count)?;
        assert_eq!(result.value(0), 5);
        Ok(())
    }

    #[test]
    fn test_count_excludes_nulls() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // COUNT([1, NULL, 3, NULL, 5]) = 3
        let values = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5)]);
        let result = run_single_group_agg::<Int32Array>(&values, Aggregation::Count)?;
        assert_eq!(result.value(0), 3);
        Ok(())
    }

    #[test]
    fn test_variance_sample() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // VARIANCE([1, 2, 3, 4, 5]) with ddof=1 = 2.5
        let values = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result =
            run_single_group_agg::<Float64Array>(&values, Aggregation::Variance { ddof: 1 })?;
        assert!((result.value(0) - 2.5).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_std_sample() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // STD([1, 2, 3, 4, 5]) with ddof=1 = sqrt(2.5)
        let values = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = run_single_group_agg::<Float64Array>(&values, Aggregation::Std { ddof: 1 })?;
        let expected = (2.5_f64).sqrt();
        assert!((result.value(0) - expected).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_nunique() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // NUNIQUE([1, 2, 2, 3, 3, 3]) = 3
        let values = Int32Array::from(vec![1, 2, 2, 3, 3, 3]);
        let result = run_single_group_agg::<Int32Array>(&values, Aggregation::NUnique)?;
        assert_eq!(result.value(0), 3);
        Ok(())
    }

    #[test]
    fn test_nunique_all_includes_nulls() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let values = Int32Array::from(vec![Some(1), None, Some(1)]);
        let result = run_single_group_agg::<Int32Array>(&values, Aggregation::NUniqueAll)?;
        assert_eq!(result.value(0), 2);
        Ok(())
    }

    #[test]
    fn test_median() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // MEDIAN([1, 2, 3, 4, 5]) = 3
        let values = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let result = run_single_group_agg::<Float64Array>(&values, Aggregation::Median)?;
        assert!((result.value(0) - 3.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_multiple_groups() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // Group 1: [1, 2, 3] sums to 6.
        // Group 2: [10, 20] sums to 30.
        let values = Int32Array::from(vec![1, 2, 3, 10, 20]);
        let keys = Int32Array::from(vec![1, 1, 1, 2, 2]);

        let values_col = crate::execute_cudf(CuDFColumn::from_arrow_host(&values))?;
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
        let keys_batch = RecordBatch::try_new(schema, vec![Arc::new(keys)])?;
        let keys_table = crate::execute_cudf(CuDFTable::from_arrow_host(keys_batch))?;
        let groupby = keys_table.into_view().group_by_all();

        let result = crate::execute_cudf(
            groupby.aggregate([GroupByRequest::new(values_col.into_view()).with(Aggregation::Sum)]),
        )?;
        let (result_keys, results) = result.into_parts();

        assert_eq!(result_keys.num_rows(), 2);

        let sum_col = results
            .into_iter()
            .next()
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let sum_array = crate::execute_cudf(sum_col.into_view().to_arrow_host())?;
        let sum_values = sum_array.as_any().downcast_ref::<Int64Array>().unwrap();

        // Should have sums [6, 30] in some order
        let sum1 = sum_values.value(0);
        let sum2 = sum_values.value(1);
        assert!(
            (sum1 == 6 && sum2 == 30) || (sum1 == 30 && sum2 == 6),
            "Expected sums [6, 30], got [{}, {}]",
            sum1,
            sum2
        );

        Ok(())
    }

    #[test]
    fn test_group_by_selected_key_columns() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let values = Int32Array::from(vec![1, 2, 3, 10, 20]);
        let keys = Int32Array::from(vec![1, 1, 1, 2, 2]);
        let ignored = Int32Array::from(vec![9, 8, 7, 6, 5]);

        let values_col = crate::execute_cudf(CuDFColumn::from_arrow_host(&values))?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("ignored", DataType::Int32, false),
        ]));
        let keys_batch = RecordBatch::try_new(schema, vec![Arc::new(keys), Arc::new(ignored)])?;
        let keys_table = crate::execute_cudf(CuDFTable::from_arrow_host(keys_batch))?;
        let groupby = keys_table.into_view().group_by(&[0])?;

        let result = crate::execute_cudf(
            groupby.aggregate([GroupByRequest::new(values_col.into_view()).with(Aggregation::Sum)]),
        )?;

        assert_eq!(result.keys().num_rows(), 2);
        assert_eq!(result.keys().num_columns(), 1);
        Ok(())
    }

    #[test]
    fn test_empty_request_list_is_rejected() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let keys = Int32Array::from(vec![1, 1, 1]);
        let keys_table = make_keys_table(&keys)?;
        let groupby = keys_table.into_view().group_by_all();

        let result = crate::execute_cudf(groupby.aggregate([]));

        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_empty_request_aggregation_list_is_rejected(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let values = Int32Array::from(vec![1, 2, 3]);
        let keys = Int32Array::from(vec![1, 1, 1]);

        let values_col = crate::execute_cudf(CuDFColumn::from_arrow_host(&values))?;
        let keys_table = make_keys_table(&keys)?;
        let groupby = keys_table.into_view().group_by_all();

        let result =
            crate::execute_cudf(groupby.aggregate([GroupByRequest::new(values_col.into_view())]));

        assert!(result.is_err());
        Ok(())
    }

    fn make_keys_table(keys: &dyn Array) -> Result<CuDFTable> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "key",
            keys.data_type().clone(),
            false,
        )]));
        let keys_data = keys.to_data();
        let batch = RecordBatch::try_new(schema, vec![make_array(keys_data)])?;
        crate::execute_cudf(CuDFTable::from_arrow_host(batch))
    }

    fn run_single_group_agg<T: Array + Clone + 'static>(
        values: &dyn Array,
        aggregation: Aggregation,
    ) -> Result<Arc<T>> {
        let n = values.len();
        let keys = Int32Array::from(vec![1; n]);

        let values_col = crate::execute_cudf(CuDFColumn::from_arrow_host(values))?;
        let keys_table = make_keys_table(&keys)?;
        let groupby = keys_table.into_view().group_by_all();

        let (_result_keys, results) = crate::execute_cudf(
            groupby.aggregate([GroupByRequest::new(values_col.into_view()).with(aggregation)]),
        )?
        .into_parts();
        let result_col = results
            .into_iter()
            .next()
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let result_array = crate::execute_cudf(result_col.into_view().to_arrow_host())?;

        Ok(Arc::new(
            (*result_array.as_any().downcast_ref::<T>().unwrap()).clone(),
        ))
    }
}

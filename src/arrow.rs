use crate::config::ensure_pools_configured;
use crate::data_type::is_arrow_type_supported_by_cudf;
use crate::deferred_operation::deferred;
use crate::execution_policy;
use crate::{
    CuDFColumn, CuDFColumnView, CuDFError, CuDFExecutionContext, CuDFOperation, CuDFScalar,
    CuDFTable, CuDFTableView,
};
use ::arrow::array::{
    make_array, Array, ArrayData, ArrayRef, RecordBatch, RecordBatchOptions, Scalar, StructArray,
};
use ::arrow::ffi::{from_ffi, from_ffi_and_data_type, FFI_ArrowArray};
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::{ArrowError, Schema, SchemaRef};
use libcudf_sys::{ffi, ArrowDeviceArray};
use std::sync::Arc;

/// Check whether an Arrow array is backed by cuDF GPU storage.
///
/// Returns `true` for [`CuDFColumnView`] and [`CuDFScalar`] arrays, and `false`
/// for regular host Arrow arrays.
///
/// # Examples
///
/// ```no_run
/// use arrow::array::{Array, Int32Array};
/// use libcudf_rs::{is_cudf_array, CuDFColumn, CuDFExecutionContext};
///
/// let host_array = Int32Array::from(vec![1, 2, 3]);
/// assert!(!is_cudf_array(&host_array));
///
/// let gpu_array = CuDFExecutionContext::try_new_non_blocking()?
///     .execute(CuDFColumn::from_arrow_host(&host_array))?
///     .into_view();
/// assert!(is_cudf_array(&gpu_array));
/// # Ok::<(), libcudf_rs::CuDFError>(())
/// ```
pub fn is_cudf_array(arr: &dyn Array) -> bool {
    let any = arr.as_any();
    any.is::<CuDFColumnView>() || any.is::<CuDFScalar>()
}

/// Check whether all columns in a RecordBatch are backed by cuDF GPU storage.
///
/// # Examples
///
/// ```no_run
/// # use arrow::array::{Int32Array, RecordBatch};
/// # use arrow::datatypes::{DataType, Field, Schema};
/// # use libcudf_rs::{is_cudf_record_batch, CuDFColumn, CuDFExecutionContext};
/// # use std::sync::Arc;
/// let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
/// let host_array = Int32Array::from(vec![1, 2, 3]);
/// let host_batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(host_array)])?;
/// assert!(!is_cudf_record_batch(&host_batch));
///
/// let values = Int32Array::from(vec![1, 2, 3]);
/// let gpu_array = CuDFExecutionContext::try_new_non_blocking()?
///     .execute(CuDFColumn::from_arrow_host(&values))?
///     .into_view();
/// let gpu_batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(gpu_array)])?;
/// assert!(is_cudf_record_batch(&gpu_batch));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn is_cudf_record_batch(batch: &RecordBatch) -> bool {
    batch.columns().iter().all(|col| is_cudf_array(col))
}

impl CuDFColumn {
    /// Create a deferred operation that copies an Arrow array into a cuDF column.
    ///
    /// The copy does not start until the returned operation is passed to
    /// [`CuDFExecutionContext::execute`]. The input array must stay alive until
    /// `execute` returns; the operation keeps Arrow FFI buffers alive until the
    /// submitted cuDF work is ready on the target stream.
    ///
    /// # Errors
    ///
    /// Execution returns an error if:
    /// - The Arrow array cannot be converted to cuDF format
    /// - There is insufficient GPU memory
    ///
    /// # Example
    ///
    /// ```no_run
    /// use arrow::array::{Array, Int32Array};
    /// use libcudf_rs::{CuDFColumn, CuDFExecutionContext};
    ///
    /// let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    /// let column = CuDFExecutionContext::try_new_non_blocking()?
    ///     .execute(CuDFColumn::from_arrow_host(&array))?;
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn from_arrow_host(array: &dyn Array) -> impl CuDFOperation<Output = Self> + '_ {
        deferred(move |ctx| Self::from_arrow_host_on_context(ctx, array))
    }

    pub(crate) fn from_arrow_host_on_context(
        ctx: &CuDFExecutionContext,
        array: &dyn Array,
    ) -> Result<Self, CuDFError> {
        ensure_pools_configured();
        if !is_arrow_type_supported_by_cudf(array.data_type()) {
            return Err(CuDFError::ArrowError(ArrowError::NotYetImplemented(
                format!("Arrow type {} not supported in CuDF", array.data_type()),
            )));
        };

        let array_data = Arc::new(array.to_data());
        let ffi_array = FFI_ArrowArray::new(&array_data);
        let ffi_schema = FFI_ArrowSchema::try_from(array.data_type())?;

        let schema_ptr = &ffi_schema as *const FFI_ArrowSchema as *const u8;
        let array_ptr = &ffi_array as *const FFI_ArrowArray as *const u8;

        let mut launch = execution_policy::launch(ctx)?;
        let inner = unsafe {
            ffi::column_from_arrow(schema_ptr, array_ptr, launch.stream()?, launch.resource())
        }?;
        launch.keep_arrow_array_data(array_data);
        launch.ready_column(Self::from_inner(inner))
    }
}

impl CuDFColumnView {
    /// Create a deferred operation that copies this cuDF column view to a host Arrow array.
    ///
    /// The copy runs on the stream in the execution context used to execute the
    /// returned operation. If this view was produced by work on another stream,
    /// execution first waits for that stream dependency.
    ///
    /// # Errors
    ///
    /// Execution returns an error if:
    /// - The cuDF column cannot be converted to Arrow format
    /// - There is an error copying data from GPU to host
    ///
    /// # Example
    ///
    /// ```no_run
    /// use arrow::array::Int32Array;
    /// use libcudf_rs::{CuDFColumn, CuDFExecutionContext};
    ///
    /// let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let column = ctx.execute(CuDFColumn::from_arrow_host(&array))?.into_view();
    /// let result = ctx.execute(column.to_arrow_host())?;
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn to_arrow_host(&self) -> impl CuDFOperation<Output = ArrayRef> + '_ {
        deferred(move |ctx| {
            let mut device_array = ArrowDeviceArray::new_cpu();
            let ffi_schema = FFI_ArrowSchema::try_from(self.data_type())?;

            let mut launch = execution_policy::launch(ctx)?;
            launch.wait_column(self)?;
            unsafe {
                let device_array_ptr = &mut device_array as *mut ArrowDeviceArray as *mut u8;
                self.inner()
                    .to_arrow_array(device_array_ptr, launch.stream()?, launch.resource());
            }
            launch.stream()?.synchronize()?;

            let array_data = unsafe { from_ffi(device_array.array, &ffi_schema)? };
            Ok(make_array(array_data))
        })
    }
}

impl CuDFScalar {
    /// Create a deferred operation that copies this scalar to a single-element Arrow array.
    ///
    /// The copy runs on the stream in the execution context used to execute the
    /// returned operation. If the scalar was produced on another stream, execution
    /// first waits for that recorded dependency.
    ///
    /// # Errors
    ///
    /// Execution returns an error if:
    /// - The cuDF scalar cannot be converted to Arrow format
    /// - There is an error copying data from GPU to host
    /// - There is insufficient memory for the conversion
    ///
    /// # Example
    ///
    /// ```no_run
    /// use arrow::array::{Array, Int32Array, Scalar};
    /// use libcudf_rs::{CuDFExecutionContext, CuDFScalar};
    ///
    /// let array = Int32Array::from(vec![42]);
    /// let scalar = Scalar::new(&array);
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let cudf_scalar = ctx.execute(CuDFScalar::from_arrow_host(scalar))?;
    /// let arrow_array = ctx.execute(cudf_scalar.to_arrow_host())?;
    ///
    /// assert_eq!(arrow_array.len(), 1);
    /// assert_eq!(arrow_array.null_count(), 0);
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn to_arrow_host(&self) -> impl CuDFOperation<Output = ArrayRef> + '_ {
        deferred(move |ctx| {
            let mut device_array = ArrowDeviceArray::new_cpu();
            let mut launch = execution_policy::launch(ctx)?;
            launch.wait_scalar(self)?;
            unsafe {
                let device_array_ptr = &mut device_array as *mut ArrowDeviceArray as *mut u8;
                self.inner()
                    .to_arrow_array(device_array_ptr, launch.stream()?, launch.resource());
            }
            launch.stream()?.synchronize()?;

            let array_data =
                unsafe { from_ffi_and_data_type(device_array.array, self.data_type().clone())? };
            Ok(make_array(array_data))
        })
    }

    /// Create a deferred operation that copies an Arrow scalar into a cuDF scalar.
    ///
    /// This creates a single-element column from the scalar, transfers it to GPU,
    /// and extracts the scalar from that column.
    ///
    /// # Errors
    ///
    /// Execution returns an error if:
    /// - The Arrow scalar cannot be converted to cuDF format
    /// - There is insufficient GPU memory
    ///
    /// # Example
    ///
    /// ```no_run
    /// use arrow::array::{Int32Array, Scalar};
    /// use libcudf_rs::{CuDFExecutionContext, CuDFScalar};
    ///
    /// let array = Int32Array::from(vec![42]);
    /// let scalar = Scalar::new(&array);
    /// let cudf_scalar = CuDFExecutionContext::try_new_non_blocking()?
    ///     .execute(CuDFScalar::from_arrow_host(scalar))?;
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn from_arrow_host<T: Array>(scalar: Scalar<T>) -> impl CuDFOperation<Output = Self> {
        deferred(move |ctx| {
            let array = scalar.into_inner();
            let column = CuDFColumn::from_arrow_host_on_context(ctx, &array)?.into_view();

            let mut launch = execution_policy::launch(ctx)?;
            launch.wait_column(&column)?;
            let cudf_scalar =
                ffi::get_element(column.inner(), 0, launch.stream()?, launch.resource());

            launch.ready_scalar(Self::new(cudf_scalar))
        })
    }
}

impl CuDFTable {
    /// Create a deferred operation that copies an Arrow `RecordBatch` into a cuDF table.
    ///
    /// The copy runs when the returned operation is executed by a
    /// [`CuDFExecutionContext`]. The returned table records the stream it was
    /// produced on, so later operations on other contexts wait for the import
    /// before using the table.
    ///
    /// # Arguments
    ///
    /// * `batch` - An Arrow RecordBatch
    ///
    /// # Errors
    ///
    /// Execution returns an error if:
    /// - The Arrow data cannot be converted to cuDF format
    /// - The Arrow RecordBatch contains columns that are already in cuDF
    /// - There is insufficient GPU memory
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow::record_batch::RecordBatch;
    /// use libcudf_rs::{CuDFExecutionContext, CuDFTable};
    ///
    /// # let batch: RecordBatch = todo!();
    /// let table = CuDFExecutionContext::try_new_non_blocking()?
    ///     .execute(CuDFTable::from_arrow_host(batch))?;
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn from_arrow_host(batch: RecordBatch) -> impl CuDFOperation<Output = Self> {
        deferred(move |ctx| {
            ensure_pools_configured();
            for col in batch.columns() {
                if is_cudf_array(col) {
                    return Err(ArrowError::InvalidArgumentError("Tried to move a RecordBatch from the host to CuDF, but a column was already in CuDF".to_string()))?;
                }
            }
            let schema = batch.schema().as_ref().clone();
            let struct_array = StructArray::from(batch);
            let array_data: Arc<ArrayData> = Arc::new(struct_array.into_data());

            let ffi_array = FFI_ArrowArray::new(&array_data);
            let ffi_schema = FFI_ArrowSchema::try_from(schema)?;

            let device_array = ArrowDeviceArray::new_cpu().with_array(ffi_array);

            let schema_ptr = &ffi_schema as *const FFI_ArrowSchema as *const u8;
            let device_array_ptr = &device_array as *const ArrowDeviceArray as *const u8;
            let mut launch = execution_policy::launch(ctx)?;
            let inner = unsafe {
                ffi::table_from_arrow_host(
                    schema_ptr,
                    device_array_ptr,
                    launch.stream()?,
                    launch.resource(),
                )
            }?;

            launch.keep_arrow_array_data(array_data);
            launch.ready_table(Self::from_inner(inner))
        })
    }
}

impl CuDFTableView {
    /// Create a deferred operation that copies this table view to a host Arrow `RecordBatch`.
    ///
    /// The copy runs on the stream in the execution context used to execute the
    /// returned operation. If any table columns were produced on other streams,
    /// execution waits for their recorded dependencies first.
    ///
    /// # Errors
    ///
    /// Execution returns an error if:
    /// - The cuDF data cannot be converted to Arrow format
    /// - There is insufficient memory
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use libcudf_rs::{CuDFExecutionContext, CuDFTable};
    ///
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let table = ctx.execute(CuDFTable::read_parquet("data.parquet"))?;
    /// let batch = ctx.execute(table.into_view().to_arrow_host())?;
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn to_arrow_host(&self) -> impl CuDFOperation<Output = RecordBatch> + '_ {
        deferred(move |ctx| {
            let mut ffi_schema = FFI_ArrowSchema::empty();
            let mut ffi_array = FFI_ArrowArray::empty();

            let mut launch = execution_policy::launch(ctx)?;
            launch.wait_table(self)?;
            unsafe {
                self.inner()
                    .to_arrow_schema(&mut ffi_schema as *mut FFI_ArrowSchema as *mut u8);
                self.inner().to_arrow_array(
                    &mut ffi_array as *mut FFI_ArrowArray as *mut u8,
                    launch.stream()?,
                    launch.resource(),
                );
            }
            launch.stream()?.synchronize()?;

            let schema = Arc::new(Schema::try_from(&ffi_schema)?);
            let array_data = unsafe { from_ffi(ffi_array, &ffi_schema)? };
            let struct_array = StructArray::from(array_data);

            // Carry the row count explicitly so zero-column batches don't trip
            // Arrow's "must either specify a row count or at least one column" check.
            let options = RecordBatchOptions::new().with_row_count(Some(struct_array.len()));
            let batch = RecordBatch::try_new_with_options(
                schema,
                struct_array.columns().to_vec(),
                &options,
            )?;

            Ok(batch)
        })
    }

    /// Get this table view's Arrow schema.
    ///
    /// This reads schema metadata only. It does not copy column data from GPU
    /// memory to host memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the cuDF schema cannot be represented as an Arrow
    /// schema.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow::array::{Int32Array, RecordBatch};
    /// use arrow_schema::{DataType, Field, Schema};
    /// use libcudf_rs::{CuDFExecutionContext, CuDFTable};
    /// use std::sync::Arc;
    ///
    /// let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    /// let batch = RecordBatch::try_new(
    ///     Arc::new(schema),
    ///     vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    /// )?;
    ///
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let view = ctx.execute(CuDFTable::from_arrow_host(batch))?.into_view();
    ///
    /// assert_eq!(view.schema()?.fields().len(), 1);
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn schema(&self) -> Result<Schema, CuDFError> {
        let mut ffi_schema = FFI_ArrowSchema::empty();
        unsafe {
            self.inner()
                .to_arrow_schema(&mut ffi_schema as *mut FFI_ArrowSchema as *mut u8);
        }
        Ok(Schema::try_from(&ffi_schema)?)
    }

    /// Create an Arrow `RecordBatch` that keeps this table's columns on GPU.
    ///
    /// Each output column is a [`CuDFColumnView`] wrapped as an Arrow array.
    /// Unlike [`to_arrow_host`](Self::to_arrow_host), this does not copy data
    /// to host memory.
    ///
    /// # Errors
    ///
    /// Returns an error if a column view cannot be created or if Arrow rejects
    /// the resulting `RecordBatch`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow::array::{Int32Array, RecordBatch};
    /// use arrow_schema::{DataType, Field, Schema};
    /// use libcudf_rs::{CuDFExecutionContext, CuDFTable};
    /// use std::sync::Arc;
    ///
    /// let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    /// let batch = RecordBatch::try_new(
    ///     Arc::new(schema),
    ///     vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    /// )?;
    ///
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let view = ctx.execute(CuDFTable::from_arrow_host(batch))?.into_view();
    /// let gpu_batch = view.to_record_batch()?;
    ///
    /// assert_eq!(gpu_batch.num_rows(), 3);
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn to_record_batch(&self) -> Result<RecordBatch, CuDFError> {
        let columns = (0..self.num_columns())
            .map(|i| self.column(i).map(|col| Arc::new(col) as ArrayRef))
            .collect::<Result<Vec<_>, _>>()?;

        let options = RecordBatchOptions::new().with_row_count(Some(self.num_rows()));
        Ok(RecordBatch::try_new_with_options(
            Arc::new(self.schema()?),
            columns,
            &options,
        )?)
    }

    /// Wrap GPU columns in a `RecordBatch` whose types match `schema`.
    ///
    /// Delegates to [`record_batch_with_schema`]. Use this when the columns
    /// are still in a `CuDFTableView`.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of table columns does not match the
    /// number of schema fields, if a column view cannot be created, or if Arrow
    /// rejects the resulting `RecordBatch`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow::array::{Int32Array, RecordBatch};
    /// use arrow_schema::{DataType, Field, Schema};
    /// use libcudf_rs::{CuDFExecutionContext, CuDFTable};
    /// use std::sync::Arc;
    ///
    /// let schema = Arc::new(Schema::new(vec![Field::new(
    ///     "renamed",
    ///     DataType::Int32,
    ///     false,
    /// )]));
    /// let batch = RecordBatch::try_new(
    ///     Arc::clone(&schema),
    ///     vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    /// )?;
    ///
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let view = ctx.execute(CuDFTable::from_arrow_host(batch))?.into_view();
    /// let gpu_batch = view.to_record_batch_with_schema(&schema)?;
    ///
    /// assert_eq!(gpu_batch.schema().field(0).name(), "renamed");
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn to_record_batch_with_schema(
        &self,
        schema: &SchemaRef,
    ) -> Result<RecordBatch, CuDFError> {
        if self.num_columns() != schema.fields().len() {
            return Err(CuDFError::ArrowError(ArrowError::InvalidArgumentError(
                format!(
                    "to_record_batch_with_schema: table has {} columns but schema has {} fields",
                    self.num_columns(),
                    schema.fields().len()
                ),
            )));
        }
        let columns = (0..self.num_columns())
            .map(|i| self.column(i).map(|col| Arc::new(col) as ArrayRef))
            .collect::<Result<Vec<_>, _>>()?;
        record_batch_with_schema(columns, schema, self.num_rows()).map_err(CuDFError::ArrowError)
    }

    /// Create a table view from a `RecordBatch` containing cuDF GPU arrays.
    ///
    /// Every column in `batch` must be a [`CuDFColumnView`]. The returned table
    /// view keeps those column views alive.
    ///
    /// # Errors
    ///
    /// Returns an error if any column is not a [`CuDFColumnView`] or if the
    /// table view cannot be created.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow::array::{Int32Array, RecordBatch};
    /// use arrow_schema::{DataType, Field, Schema};
    /// use libcudf_rs::{CuDFExecutionContext, CuDFTable, CuDFTableView};
    /// use std::sync::Arc;
    ///
    /// let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    /// let batch = RecordBatch::try_new(
    ///     Arc::new(schema),
    ///     vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    /// )?;
    ///
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let gpu_batch = ctx
    ///     .execute(CuDFTable::from_arrow_host(batch))?
    ///     .into_view()
    ///     .to_record_batch()?;
    /// let view = CuDFTableView::from_record_batch(&gpu_batch)?;
    ///
    /// assert_eq!(view.num_columns(), 1);
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn from_record_batch(batch: &RecordBatch) -> Result<Self, CuDFError> {
        let column_views: Result<Vec<_>, _> = batch
            .columns()
            .iter()
            .map(|col| {
                let Some(col) = col.as_any().downcast_ref::<CuDFColumnView>() else {
                    return Err(CuDFError::ArrowError(ArrowError::InvalidArgumentError(
                        "Expected all Arrays in RecordBatch to be CuDFColumnView".to_string(),
                    )));
                };
                Ok(col.clone())
            })
            .collect();

        Self::from_column_views(column_views?)
    }
}

/// Build a `RecordBatch`, relabeling any `CuDFColumnView` whose type differs from
/// the corresponding `schema` field.
///
/// cuDF normalises decimal precision to the storage maximum (e.g. 38 for Decimal128).
/// This function restores the declared precision so `RecordBatch::try_new` accepts it.
/// All GPU `RecordBatch` creation should go through this function instead of calling
/// `RecordBatch::try_new` directly.
///
/// `num_rows` is required so zero-column batches (e.g. produced by `FilterExec`
/// with `projection=[]` for `COUNT(*) WHERE ...` plans) carry their row count.
///
/// # Errors
///
/// Returns an error if `columns` do not match `schema`, if a relabeled GPU
/// column is still incompatible with its schema field, or if Arrow rejects the
/// requested row count.
///
/// # Examples
///
/// ```no_run
/// use arrow_schema::Schema;
/// use libcudf_rs::record_batch_with_schema;
/// use std::sync::Arc;
///
/// let schema = Arc::new(Schema::empty());
/// let batch = record_batch_with_schema(vec![], &schema, 3)?;
///
/// assert_eq!(batch.num_columns(), 0);
/// assert_eq!(batch.num_rows(), 3);
/// # Ok::<(), arrow_schema::ArrowError>(())
/// ```
pub fn record_batch_with_schema(
    columns: Vec<ArrayRef>,
    schema: &SchemaRef,
    num_rows: usize,
) -> Result<RecordBatch, ArrowError> {
    if columns.len() != schema.fields().len() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "record_batch_with_schema: received {} columns but schema has {} fields",
            columns.len(),
            schema.fields().len()
        )));
    }

    let relabeled: Vec<ArrayRef> = columns
        .into_iter()
        .zip(schema.fields())
        .map(|(col, field)| {
            if col.data_type() != field.data_type() {
                if let Some(v) = col.as_any().downcast_ref::<CuDFColumnView>() {
                    return Arc::new(v.clone().with_data_type(field.data_type().clone())) as _;
                }
            }
            col
        })
        .collect();
    let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
    RecordBatch::try_new_with_options(Arc::clone(schema), relabeled, &options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::arrow::array::*;
    use ::arrow::datatypes::*;
    use std::sync::Arc;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn column_view_from_arrow(array: &dyn Array) -> Result<CuDFColumnView, CuDFError> {
        Ok(crate::execute_cudf(CuDFColumn::from_arrow_host(array))?.into_view())
    }

    fn column_to_arrow_host(array: &dyn Array) -> Result<ArrayRef, CuDFError> {
        crate::execute_cudf(column_view_from_arrow(array)?.to_arrow_host())
    }

    fn scalar_from_arrow<T: Array>(array: &T) -> Result<CuDFScalar, CuDFError> {
        crate::execute_cudf(CuDFScalar::from_arrow_host(Scalar::new(array)))
    }

    fn scalar_to_arrow_host<T: Array>(array: &T) -> Result<ArrayRef, CuDFError> {
        crate::execute_cudf(scalar_from_arrow(array)?.to_arrow_host())
    }

    fn assert_scalar_type<T: Array>(array: &T, expected: DataType) -> TestResult {
        let scalar = scalar_from_arrow(array)?;
        assert_eq!(scalar.data_type(), &expected);
        Ok(())
    }

    fn table_to_arrow_host(batch: RecordBatch) -> Result<RecordBatch, CuDFError> {
        let table = crate::execute_cudf(CuDFTable::from_arrow_host(batch))?;
        crate::execute_cudf(table.into_view().to_arrow_host())
    }

    fn assert_record_batch_equal(expected: &RecordBatch, actual: &RecordBatch) {
        assert_eq!(actual.num_rows(), expected.num_rows());
        assert_eq!(actual.num_columns(), expected.num_columns());

        for (i, (expected_field, actual_field)) in expected
            .schema()
            .fields()
            .iter()
            .zip(actual.schema().fields().iter())
            .enumerate()
        {
            assert_eq!(
                actual_field.data_type(),
                expected_field.data_type(),
                "data type mismatch for column {i}"
            );
        }

        for col_idx in 0..expected.num_columns() {
            assert_eq!(
                expected.column(col_idx),
                actual.column(col_idx),
                "data mismatch for column {col_idx}"
            );
        }
    }

    #[test]
    fn column_from_arrow_preserves_shape() -> TestResult {
        let int_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let string_array = StringArray::from(vec!["hello", "world", "test"]);
        let nullable_array = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5)]);

        let int_column = column_view_from_arrow(&int_array)?;
        let string_column = column_view_from_arrow(&string_array)?;
        let nullable_column = column_view_from_arrow(&nullable_array)?;

        assert_eq!(int_column.len(), 5);
        assert!(!int_column.is_empty());
        assert_eq!(string_column.len(), 3);
        assert!(!string_column.is_empty());
        assert_eq!(nullable_column.len(), 5);
        assert!(!nullable_column.is_empty());
        Ok(())
    }

    #[test]
    fn column_to_arrow_host_int64() -> TestResult {
        let original = Int64Array::from(vec![100, 200, 300, 400, 500]);
        let result = column_to_arrow_host(&original)?;

        assert_eq!(result.len(), 5);
        let result = result.as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..5 {
            assert_eq!(result.value(i), original.value(i));
        }
        Ok(())
    }

    #[test]
    fn column_to_arrow_host_float64() -> TestResult {
        let original = Float64Array::from(vec![1.5, 2.5, 3.5, 4.5, 5.5]);
        let result = column_to_arrow_host(&original)?;

        assert_eq!(result.len(), 5);
        let result = result.as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..5 {
            assert_eq!(result.value(i), original.value(i));
        }
        Ok(())
    }

    #[test]
    fn column_to_arrow_host_string() -> TestResult {
        let original = StringArray::from(vec!["hello", "world", "test", "cudf", "rust"]);
        let result = column_to_arrow_host(&original)?;

        assert_eq!(result.len(), 5);
        let result = result.as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..5 {
            assert_eq!(result.value(i), original.value(i));
        }
        Ok(())
    }

    #[test]
    fn column_to_arrow_host_with_nulls() -> TestResult {
        let original = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5)]);
        let result = column_to_arrow_host(&original)?;

        assert_eq!(result.len(), 5);
        let result = result.as_any().downcast_ref::<Int32Array>().unwrap();
        assert!(result.is_valid(0));
        assert_eq!(result.value(0), 1);
        assert!(result.is_null(1));
        assert!(result.is_valid(2));
        assert_eq!(result.value(2), 3);
        assert!(result.is_null(3));
        assert!(result.is_valid(4));
        assert_eq!(result.value(4), 5);
        Ok(())
    }

    #[test]
    fn column_to_arrow_host_empty() -> TestResult {
        let original = Int32Array::from(Vec::<i32>::new());
        let result = column_to_arrow_host(&original)?;

        assert_eq!(result.len(), 0);
        assert!(result.is_empty());
        Ok(())
    }

    #[test]
    fn column_to_arrow_host_roundtrip_preserves_data() -> TestResult {
        let original = Int32Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
        let result = column_to_arrow_host(&original)?;
        let result = result.as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(result.len(), original.len());
        for i in 0..original.len() {
            assert_eq!(result.value(i), original.value(i));
        }
        Ok(())
    }

    #[test]
    fn scalar_to_arrow_host_int32() -> TestResult {
        let array = Int32Array::from(vec![42]);
        let result = scalar_to_arrow_host(&array)?;

        assert_eq!(result.len(), 1);
        assert_eq!(result.data_type(), &DataType::Int32);
        let result = result.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(result.value(0), 42);
        Ok(())
    }

    #[test]
    fn scalar_to_arrow_host_int64() -> TestResult {
        let array = Int64Array::from(vec![12345_i64]);
        let result = scalar_to_arrow_host(&array)?;

        assert_eq!(result.len(), 1);
        assert_eq!(result.data_type(), &DataType::Int64);
        let result = result.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(result.value(0), 12345);
        Ok(())
    }

    #[test]
    fn scalar_to_arrow_host_float64() -> TestResult {
        let array = Float64Array::from(vec![std::f64::consts::PI]);
        let result = scalar_to_arrow_host(&array)?;

        assert_eq!(result.len(), 1);
        assert_eq!(result.data_type(), &DataType::Float64);
        let result = result.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(result.value(0), std::f64::consts::PI);
        Ok(())
    }

    #[test]
    fn scalar_to_arrow_host_boolean() -> TestResult {
        let array = BooleanArray::from(vec![true]);
        let result = scalar_to_arrow_host(&array)?;

        assert_eq!(result.len(), 1);
        assert_eq!(result.data_type(), &DataType::Boolean);
        let result = result.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(result.value(0));
        Ok(())
    }

    #[test]
    fn scalar_to_arrow_host_string() -> TestResult {
        let array = StringArray::from(vec!["hello world"]);
        let result = scalar_to_arrow_host(&array)?;

        assert_eq!(result.len(), 1);
        assert_eq!(result.data_type(), &DataType::Utf8);
        let result = result.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(result.value(0), "hello world");
        Ok(())
    }

    #[test]
    fn scalar_to_arrow_host_null() -> TestResult {
        let result = scalar_to_arrow_host(&Int32Array::from(vec![None]))?;

        assert_eq!(result.len(), 1);
        assert_eq!(result.data_type(), &DataType::Int32);
        assert_eq!(result.null_count(), 1);
        Ok(())
    }

    #[test]
    fn scalar_from_arrow_host_types() -> TestResult {
        assert_scalar_type(&Int32Array::from(vec![42]), DataType::Int32)?;
        assert_scalar_type(&Int64Array::from(vec![12345]), DataType::Int64)?;
        assert_scalar_type(
            &Float64Array::from(vec![std::f64::consts::PI]),
            DataType::Float64,
        )?;
        assert_scalar_type(&StringArray::from(vec!["hello"]), DataType::Utf8)?;
        Ok(())
    }

    #[test]
    fn scalar_from_arrow_host_null() -> TestResult {
        let scalar = scalar_from_arrow(&Int32Array::from(vec![None]))?;

        assert_eq!(scalar.data_type(), &DataType::Int32);
        assert!(!scalar.inner().is_valid());
        Ok(())
    }

    #[test]
    fn table_arrow_roundtrip_simple() -> TestResult {
        let schema = Schema::new(vec![
            Field::new("int8", DataType::Int8, false),
            Field::new("int16", DataType::Int16, false),
            Field::new("int32", DataType::Int32, false),
            Field::new("int64", DataType::Int64, false),
            Field::new("uint8", DataType::UInt8, false),
            Field::new("uint16", DataType::UInt16, false),
            Field::new("uint32", DataType::UInt32, false),
            Field::new("uint64", DataType::UInt64, false),
            Field::new("float32", DataType::Float32, false),
            Field::new("float64", DataType::Float64, false),
            Field::new("bool", DataType::Boolean, false),
            Field::new("string", DataType::Utf8, false),
            Field::new("date32", DataType::Date32, false),
            Field::new(
                "timestamp_ms",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                false,
            ),
        ]);

        let arrays: Vec<ArrayRef> = vec![
            Arc::new(Int8Array::from(vec![1i8, 2, 3, 4, 5])),
            Arc::new(Int16Array::from(vec![10i16, 20, 30, 40, 50])),
            Arc::new(Int32Array::from(vec![100i32, 200, 300, 400, 500])),
            Arc::new(Int64Array::from(vec![1000i64, 2000, 3000, 4000, 5000])),
            Arc::new(UInt8Array::from(vec![1u8, 2, 3, 4, 5])),
            Arc::new(UInt16Array::from(vec![10u16, 20, 30, 40, 50])),
            Arc::new(UInt32Array::from(vec![100u32, 200, 300, 400, 500])),
            Arc::new(UInt64Array::from(vec![1000u64, 2000, 3000, 4000, 5000])),
            Arc::new(Float32Array::from(vec![1.5f32, 2.5, 3.5, 4.5, 5.5])),
            Arc::new(Float64Array::from(vec![10.5f64, 20.5, 30.5, 40.5, 50.5])),
            Arc::new(BooleanArray::from(vec![true, false, true, false, true])),
            Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
            Arc::new(Date32Array::from(vec![18000, 18001, 18002, 18003, 18004])),
            Arc::new(TimestampMillisecondArray::from(vec![
                1609459200000i64,
                1609545600000,
                1609632000000,
                1609718400000,
                1609804800000,
            ])),
        ];

        let batch = RecordBatch::try_new(Arc::new(schema), arrays)?;
        let result_batch = table_to_arrow_host(batch.clone())?;

        assert_record_batch_equal(&batch, &result_batch);
        Ok(())
    }

    #[test]
    fn table_arrow_empty_roundtrip() -> TestResult {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float64, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(Float64Array::from(Vec::<f64>::new())),
            ],
        )?;

        let result_batch = table_to_arrow_host(batch)?;
        assert_eq!(result_batch.num_rows(), 0);
        assert_eq!(result_batch.num_columns(), 2);
        Ok(())
    }

    #[test]
    fn table_arrow_parquet_roundtrip() -> TestResult {
        let table = crate::execute_cudf(CuDFTable::read_parquet(
            "testdata/weather/result-000000.parquet",
        ))?;
        let batch = crate::execute_cudf(table.into_view().to_arrow_host())?;

        let original_rows = batch.num_rows();
        let original_cols = batch.num_columns();
        let table = crate::execute_cudf(CuDFTable::from_arrow_host(batch))?;

        assert_eq!(table.num_rows(), original_rows);
        assert_eq!(table.num_columns(), original_cols);
        Ok(())
    }
}

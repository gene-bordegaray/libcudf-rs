use cxx::UniquePtr;
use libcudf_sys::ffi;
use std::sync::Arc;

use crate::deferred_operation::{execute_on_context, CuDFOperation};
use crate::{CuDFError, CuDFStream, CuDFStreamFlags, Result};

/// Execution context for cuDF work submitted to a CUDA stream.
///
/// A context captures the current CUDA device, stream policy, and RMM memory
/// resource used when executing [`CuDFOperation`] values.
///
/// ```no_run
/// use arrow::array::Int32Array;
/// use libcudf_rs::{CuDFColumn, CuDFExecutionContext};
/// use arrow_schema::DataType;
///
/// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
/// let input = Int32Array::from(vec![1, 2, 3]);
///
/// let column = ctx.execute(CuDFColumn::from_arrow_host(&input))?.into_view();
/// let casted = ctx.execute(column.cast(&DataType::Int64))?;
/// let host = ctx.execute(casted.into_view().to_arrow_host())?;
/// assert_eq!(host.len(), 3);
/// # Ok::<(), libcudf_rs::CuDFError>(())
/// ```
///
/// GPU outputs record stream-readiness metadata. Later operations on another
/// context wait on that metadata before reading the value.
pub struct CuDFExecutionContext {
    stream: CuDFExecutionStream,
    resource: CuDFDeviceResource,
    device_id: i32,
}

enum CuDFExecutionStream {
    Default,
    Owned(Arc<CuDFStream>),
}

/// Device memory resource captured for a cuDF execution context.
struct CuDFDeviceResource {
    inner: UniquePtr<ffi::DeviceAsyncResourceRef>,
}

impl CuDFDeviceResource {
    /// Captures the current cuDF/RMM device resource.
    fn current() -> Self {
        Self {
            inner: ffi::get_current_device_resource_ref(),
        }
    }

    /// Returns the non-null resource reference passed to sys bindings.
    fn as_ref(&self) -> Result<&ffi::DeviceAsyncResourceRef> {
        resource_ref(&self.inner)
    }
}

/// Return a non-null device resource reference from a cuDF FFI handle.
pub(crate) fn resource_ref(
    resource: &UniquePtr<ffi::DeviceAsyncResourceRef>,
) -> Result<&ffi::DeviceAsyncResourceRef> {
    resource
        .as_ref()
        .ok_or(CuDFError::NullHandle("current device resource ref"))
}

impl CuDFExecutionContext {
    /// Create a context backed by cuDF's default CUDA stream.
    ///
    /// This is useful for synchronous integrations that are not yet stream-aware.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot report the current device.
    pub fn try_default_stream() -> Result<Self> {
        Ok(Self {
            stream: CuDFExecutionStream::Default,
            resource: CuDFDeviceResource::current(),
            device_id: ffi::cuda_get_device()?,
        })
    }

    /// Create a context backed by a non-blocking owned CUDA stream.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot create a stream or report the current
    /// device.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use libcudf_rs::CuDFExecutionContext;
    ///
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// ctx.synchronize()?;
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn try_new_non_blocking() -> Result<Self> {
        Ok(Self {
            stream: CuDFExecutionStream::Owned(Arc::new(CuDFStream::try_with_flags(
                CuDFStreamFlags::NonBlocking,
            )?)),
            resource: CuDFDeviceResource::current(),
            device_id: ffi::cuda_get_device()?,
        })
    }

    /// Create a context from an existing owned stream.
    ///
    /// The stream's device becomes the context device. Construction temporarily
    /// activates that device so the captured memory resource matches the stream,
    /// then restores the previous current CUDA device before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot switch devices or report the current
    /// cuDF/RMM memory resource.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use libcudf_rs::{CuDFExecutionContext, CuDFStream, CuDFStreamFlags};
    ///
    /// let stream = CuDFStream::try_with_flags(CuDFStreamFlags::NonBlocking)?;
    /// let ctx = CuDFExecutionContext::try_from_stream(stream)?;
    /// ctx.synchronize()?;
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn try_from_stream(stream: CuDFStream) -> Result<Self> {
        let device_id = stream.device_id();
        let _device = CurrentDeviceGuard::switch_to(device_id)?;
        let resource = CuDFDeviceResource::current();
        Ok(Self {
            stream: CuDFExecutionStream::Owned(Arc::new(stream)),
            resource,
            device_id,
        })
    }

    /// Block until all work submitted to this context has completed.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot activate the context device or
    /// synchronize the stream.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use libcudf_rs::CuDFExecutionContext;
    ///
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// ctx.synchronize()?;
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn synchronize(&self) -> Result<()> {
        let _device = self.activate_device()?;
        self.stream_view()?
            .as_ref()
            .ok_or(CuDFError::NullHandle("CUDA stream view"))?
            .synchronize()?;
        Ok(())
    }

    /// Run a deferred cuDF operation on this context.
    ///
    /// # Errors
    ///
    /// Returns an error if device activation fails or the submitted operation
    /// fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow::array::Int32Array;
    /// use libcudf_rs::{CuDFColumn, CuDFExecutionContext};
    ///
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let input = Int32Array::from(vec![1, 2, 3]);
    /// let column = ctx.execute(CuDFColumn::from_arrow_host(&input))?;
    /// assert_eq!(column.len(), 3);
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn execute<O>(&self, operation: O) -> Result<O::Output>
    where
        O: CuDFOperation,
    {
        execute_on_context(operation, self)
    }

    pub(crate) fn activate_device(&self) -> Result<CurrentDeviceGuard> {
        CurrentDeviceGuard::switch_to(self.device_id)
    }

    pub(crate) fn stream_view(&self) -> Result<UniquePtr<ffi::CudaStreamView>> {
        match &self.stream {
            CuDFExecutionStream::Default => Ok(ffi::get_default_stream()),
            CuDFExecutionStream::Owned(stream) => stream.view(),
        }
    }

    pub(crate) fn stream_keepalive(&self) -> Option<Arc<CuDFStream>> {
        match &self.stream {
            CuDFExecutionStream::Default => None,
            CuDFExecutionStream::Owned(stream) => Some(Arc::clone(stream)),
        }
    }

    pub(crate) fn resource(&self) -> Result<&ffi::DeviceAsyncResourceRef> {
        self.resource.as_ref()
    }
}

pub(crate) struct CurrentDeviceGuard {
    previous_device_id: i32,
}

impl CurrentDeviceGuard {
    fn switch_to(device_id: i32) -> Result<Self> {
        let previous_device_id = ffi::cuda_get_device()?;
        ffi::cuda_set_device(device_id)?;
        Ok(Self { previous_device_id })
    }
}

impl Drop for CurrentDeviceGuard {
    fn drop(&mut self) {
        let _ = ffi::cuda_set_device(self.previous_device_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keep_alive::CuDFKeepAlive;
    use crate::stream_readiness::CuDFStreamDependency;
    use arrow::array::{Array, BooleanArray, Int32Array, Int64Array, Scalar};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    #[test]
    fn dependency_records_waits_and_keeps_stream_alive(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let producer = CuDFExecutionContext::try_new_non_blocking()?;
        let consumer = CuDFExecutionContext::try_new_non_blocking()?;
        let stream = producer
            .stream_keepalive()
            .expect("non-blocking context should own a stream");
        let stream_weak = Arc::downgrade(&stream);
        drop(stream);

        let mut keepalives: Vec<CuDFKeepAlive> = Vec::new();
        keepalives.push(CuDFKeepAlive::Stream {
            _stream: producer
                .stream_keepalive()
                .expect("non-blocking context should own a stream"),
        });
        let stream = producer.stream_view()?;
        let dependency = CuDFStreamDependency::record_on_stream(
            stream
                .as_ref()
                .expect("execution context stream should not be null"),
            keepalives,
        )?;
        drop(producer);

        assert!(stream_weak.upgrade().is_some());
        let consumer_stream = consumer.stream_view()?;
        dependency.wait_on_stream(
            consumer_stream
                .as_ref()
                .expect("consumer stream should not be null"),
        )?;
        consumer.synchronize()?;
        drop(dependency);
        assert!(stream_weak.upgrade().is_none());

        Ok(())
    }

    #[test]
    fn context_column_import_synchronizes_on_host_read(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let ctx = CuDFExecutionContext::try_new_non_blocking()?;
        let input = Int32Array::from(vec![1, 2, 3]);
        let column = ctx.execute(crate::CuDFColumn::from_arrow_host(&input))?;

        assert!(column.stream_readiness().is_some());
        assert_i32_values(&column.into_view(), &[1, 2, 3])?;
        Ok(())
    }

    #[test]
    fn context_output_can_be_consumed_by_another_context_cast(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let producer = CuDFExecutionContext::try_new_non_blocking()?;
        let consumer = CuDFExecutionContext::try_new_non_blocking()?;
        let input = Int32Array::from(vec![1, 2, 3]);
        let column = producer
            .execute(crate::CuDFColumn::from_arrow_host(&input))?
            .into_view();

        let casted = consumer.execute(column.cast(&DataType::Int64))?;

        assert!(casted.stream_readiness().is_some());
        assert_i64_values(&casted.into_view(), &[1, 2, 3])?;
        Ok(())
    }

    #[test]
    fn try_from_stream_restores_current_device_and_executes(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let original_device = ffi::cuda_get_device()?;
        let stream = CuDFStream::try_with_flags(CuDFStreamFlags::NonBlocking)?;
        let ctx = CuDFExecutionContext::try_from_stream(stream)?;

        assert_eq!(ffi::cuda_get_device()?, original_device);

        let input = Int32Array::from(vec![1, 2, 3]);
        let column = ctx.execute(crate::CuDFColumn::from_arrow_host(&input))?;
        assert_i32_values(&column.into_view(), &[1, 2, 3])?;
        assert_eq!(ffi::cuda_get_device()?, original_device);
        Ok(())
    }

    #[test]
    fn table_view_preserves_per_column_readiness(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let left_ctx = CuDFExecutionContext::try_new_non_blocking()?;
        let right_ctx = CuDFExecutionContext::try_new_non_blocking()?;
        let left_input = Int32Array::from(vec![1, 2, 3]);
        let right_input = Int32Array::from(vec![10, 20, 30]);
        let left = left_ctx
            .execute(crate::CuDFColumn::from_arrow_host(&left_input))?
            .into_view();
        let right = right_ctx
            .execute(crate::CuDFColumn::from_arrow_host(&right_input))?
            .into_view();

        let table = crate::CuDFTableView::from_column_views(vec![left, right])?;
        let projected = table.select_columns(&[1, 0])?;
        let first = projected.column(0)?;
        let second = projected.column(1)?;

        assert!(first.stream_readiness().is_some());
        assert!(second.stream_readiness().is_some());
        assert_i32_values(&first, &[10, 20, 30])?;
        assert_i32_values(&second, &[1, 2, 3])?;
        Ok(())
    }

    #[test]
    fn slice_column_preserves_readiness() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let producer = CuDFExecutionContext::try_new_non_blocking()?;
        let consumer = CuDFExecutionContext::try_new_non_blocking()?;
        let input = Int32Array::from(vec![1, 2, 3, 4]);
        let column = producer
            .execute(crate::CuDFColumn::from_arrow_host(&input))?
            .into_view();

        let sliced = consumer.execute(column.slice_view(1, 2))?;

        assert!(sliced.stream_readiness().is_some());
        assert_i32_values(&sliced, &[2, 3])?;
        Ok(())
    }

    #[test]
    fn apply_boolean_mask_waits_for_cross_stream_inputs(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let table_ctx = CuDFExecutionContext::try_new_non_blocking()?;
        let mask_ctx = CuDFExecutionContext::try_new_non_blocking()?;
        let consumer = CuDFExecutionContext::try_new_non_blocking()?;

        let table = table_ctx
            .execute(crate::CuDFTable::from_arrow_host(record_batch_i32(vec![
                1, 2, 3, 4, 5,
            ])?))?
            .into_view();
        let mask = BooleanArray::from(vec![true, false, true, false, true]);
        let mask = mask_ctx
            .execute(crate::CuDFColumn::from_arrow_host(&mask))?
            .into_view();

        let filtered = consumer.execute(table.filter(&mask))?;

        assert_table_i32_values(filtered.into_view(), &[1, 3, 5])?;
        Ok(())
    }

    #[test]
    fn concat_waits_for_cross_stream_inputs() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let left_ctx = CuDFExecutionContext::try_new_non_blocking()?;
        let right_ctx = CuDFExecutionContext::try_new_non_blocking()?;
        let consumer = CuDFExecutionContext::try_new_non_blocking()?;

        let left = left_ctx
            .execute(crate::CuDFTable::from_arrow_host(record_batch_i32(vec![
                1, 2,
            ])?))?
            .into_view();
        let right = right_ctx
            .execute(crate::CuDFTable::from_arrow_host(record_batch_i32(vec![
                3, 4,
            ])?))?
            .into_view();

        let concatenated = consumer.execute(crate::CuDFTable::concat(vec![left, right]))?;

        assert_table_i32_values(concatenated.into_view(), &[1, 2, 3, 4])?;
        Ok(())
    }

    #[test]
    fn group_by_waits_for_cross_stream_keys_and_values(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let keys_ctx = CuDFExecutionContext::try_new_non_blocking()?;
        let values_ctx = CuDFExecutionContext::try_new_non_blocking()?;
        let consumer = CuDFExecutionContext::try_new_non_blocking()?;

        let keys = keys_ctx
            .execute(crate::CuDFTable::from_arrow_host(record_batch_i32(vec![
                1, 2, 1, 2,
            ])?))?
            .into_view();
        let values = Int32Array::from(vec![10, 20, 30, 40]);
        let values = values_ctx
            .execute(crate::CuDFColumn::from_arrow_host(&values))?
            .into_view();

        let group_by = keys.group_by_all();
        let result =
            consumer
                .execute(group_by.aggregate([
                    crate::GroupByRequest::new(values).with(crate::Aggregation::Sum),
                ]))?;
        let (keys, results) = result.into_parts();
        let sums = results
            .into_iter()
            .next()
            .expect("one request")
            .into_iter()
            .next()
            .expect("one aggregation");

        let key_batch = CuDFExecutionContext::try_new_non_blocking()?
            .execute(keys.into_view().to_arrow_host())?;
        let key_values = key_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("expected Int32Array");
        let sum_values = CuDFExecutionContext::try_new_non_blocking()?
            .execute(sums.into_view().to_arrow_host())?;
        let sum_values = sum_values
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("expected Int64Array");

        let mut pairs = (0..key_values.len())
            .map(|i| (key_values.value(i), sum_values.value(i)))
            .collect::<Vec<_>>();
        pairs.sort_unstable_by_key(|(key, _)| *key);

        assert_eq!(pairs, vec![(1, 40), (2, 60)]);
        Ok(())
    }

    #[test]
    fn select_columns_rejects_invalid_indices_before_ffi(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let ctx = CuDFExecutionContext::try_new_non_blocking()?;
        let table = ctx
            .execute(crate::CuDFTable::from_arrow_host(record_batch_i32(vec![
                1, 2, 3,
            ])?))?
            .into_view();

        let Err(err) = table.column(1) else {
            panic!("index should fail");
        };
        assert!(format!("{err}").contains("out of bounds"));

        let Err(err) = table.select_columns(&[1]) else {
            panic!("index should fail");
        };
        assert!(format!("{err}").contains("out of bounds"));
        Ok(())
    }

    #[test]
    fn context_table_import_synchronizes_on_host_read(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let ctx = CuDFExecutionContext::try_new_non_blocking()?;
        let table = ctx
            .execute(crate::CuDFTable::from_arrow_host(record_batch_i32(vec![
                1, 2, 3,
            ])?))?
            .into_view();

        let result =
            CuDFExecutionContext::try_new_non_blocking()?.execute(table.to_arrow_host())?;
        let values = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("expected Int32Array");
        assert_eq!(values.values(), &[1, 2, 3]);
        Ok(())
    }

    #[test]
    fn scalar_operations_execute_on_context() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let ctx = CuDFExecutionContext::try_new_non_blocking()?;
        let input = Int32Array::from(vec![7]);
        let scalar = ctx.execute(crate::CuDFScalar::from_arrow_host(Scalar::new(&input)))?;

        let host_scalar =
            CuDFExecutionContext::try_new_non_blocking()?.execute(scalar.to_arrow_host())?;
        let values = host_scalar
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("expected Int32Array");
        assert_eq!(values.values(), &[7]);

        let column = ctx.execute(crate::CuDFColumn::from_scalar(&scalar, 3))?;
        assert_i32_values(&column.into_view(), &[7, 7, 7])?;
        Ok(())
    }

    fn record_batch_i32(
        values: Vec<i32>,
    ) -> std::result::Result<RecordBatch, Box<dyn std::error::Error>> {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        Ok(RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(values))],
        )?)
    }

    fn assert_table_i32_values(
        table: crate::CuDFTableView,
        expected: &[i32],
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let result =
            CuDFExecutionContext::try_new_non_blocking()?.execute(table.to_arrow_host())?;
        let result = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("expected Int32Array");
        assert_eq!(result.values(), expected);
        Ok(())
    }

    fn assert_i32_values(
        column: &crate::CuDFColumnView,
        expected: &[i32],
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let result =
            CuDFExecutionContext::try_new_non_blocking()?.execute(column.to_arrow_host())?;
        let result = result
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("expected Int32Array");
        assert_eq!(result.values(), expected);
        Ok(())
    }

    fn assert_i64_values(
        column: &crate::CuDFColumnView,
        expected: &[i64],
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let result =
            CuDFExecutionContext::try_new_non_blocking()?.execute(column.to_arrow_host())?;
        let result = result
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("expected Int64Array");
        assert_eq!(result.values(), expected);
        Ok(())
    }
}

use crate::data_type::{arrow_type_to_cudf_data_type, cudf_type_to_arrow};
use crate::deferred_operation::deferred;
use crate::execution_policy;
use crate::stream_readiness::{CuDFStreamDependency, CuDFStreamReady};
use crate::{CuDFColumn, CuDFError, CuDFExecutionContext, CuDFOperation, CuDFViewStorage};
use arrow::array::{Array, ArrayData, ArrayRef};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_schema::{ArrowError, DataType};
use cxx::UniquePtr;
use libcudf_sys::ffi;
use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, OnceLock};

/// A non-owning view of a GPU column.
///
/// This is a safe wrapper around cuDF's `column_view` type. A view may keep an
/// owning cuDF object alive so the referenced GPU buffers remain valid, and it
/// implements Arrow's [`Array`] trait for interoperability with Arrow APIs.
pub struct CuDFColumnView {
    // Non-owning cuDF view. Keep this before `storage` so it drops first.
    inner: UniquePtr<ffi::ColumnView>,
    // Backing owner for the buffers referenced by `inner`.
    storage: Option<CuDFViewStorage>,
    dt: DataType,
    null_buf: OnceLock<Option<NullBuffer>>,
    stream_readiness: Option<CuDFStreamDependency>,
}

impl CuDFColumnView {
    pub(crate) fn from_view(
        inner: UniquePtr<ffi::ColumnView>,
        storage: Option<CuDFViewStorage>,
        stream_readiness: Option<CuDFStreamDependency>,
    ) -> Self {
        let cudf_dtype = inner.data_type();
        let dt = cudf_type_to_arrow(&cudf_dtype);
        let dt = dt.unwrap_or(DataType::Null);
        Self {
            inner,
            storage,
            dt,
            null_buf: OnceLock::new(),
            stream_readiness,
        }
    }

    pub(crate) fn inner(&self) -> &UniquePtr<ffi::ColumnView> {
        &self.inner
    }

    pub(crate) fn into_inner(self) -> UniquePtr<ffi::ColumnView> {
        self.inner
    }

    /// Relabel this view's Arrow `DataType` without touching GPU memory.
    /// Used by [`record_batch_with_schema`](crate::record_batch_with_schema) to
    /// adjust cuDF's max-precision decimals with declared schema types.
    pub(crate) fn with_data_type(self, dt: DataType) -> Self {
        Self {
            inner: self.inner,
            storage: self.storage,
            dt,
            null_buf: self.null_buf,
            stream_readiness: self.stream_readiness,
        }
    }

    pub(crate) fn stream_readiness(&self) -> Option<&CuDFStreamDependency> {
        self.stream_readiness.as_ref()
    }

    /// Create a deferred operation that returns this view's own device-buffer memory.
    ///
    /// This matches cuDF's `column_view::get_buffer_memory_size` and does not
    /// include child buffers.
    ///
    /// # Errors
    ///
    /// Execution returns an error if this view's stream readiness cannot be
    /// waited on by the target execution context.
    pub fn buffer_memory_size(&self) -> impl CuDFOperation<Output = usize> + '_ {
        deferred(move |ctx| {
            let mut launch = execution_policy::launch(ctx)?;
            launch.wait_column(self)?;
            Ok(self.inner().get_buffer_memory_size(launch.stream()?))
        })
    }

    /// Create a deferred operation that returns this view's total device memory.
    ///
    /// This matches cuDF's `column_view::get_array_memory_size` and includes
    /// child buffers.
    ///
    /// # Errors
    ///
    /// Execution returns an error if this view's stream readiness cannot be
    /// waited on by the target execution context.
    pub fn array_memory_size(&self) -> impl CuDFOperation<Output = usize> + '_ {
        deferred(move |ctx| {
            let mut launch = execution_policy::launch(ctx)?;
            launch.wait_column(self)?;
            Ok(self.inner().get_array_memory_size(launch.stream()?))
        })
    }

    /// Cast this column to a different Arrow data type on the GPU.
    ///
    /// This creates a deferred cuDF cast operation. The cast does not run until
    /// the returned operation is passed to [`CuDFExecutionContext::execute`].
    /// Execution waits for this column's stream readiness before launching the
    /// cast on the target context.
    ///
    /// # Errors
    ///
    /// Execution returns an error if:
    /// - `target_type` is not supported by cuDF
    /// - cuDF cannot cast this column to `target_type`
    /// - the output column cannot be allocated
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow::array::Int32Array;
    /// use arrow_schema::DataType;
    /// use libcudf_rs::{CuDFColumn, CuDFExecutionContext};
    ///
    /// let input = Int32Array::from(vec![1, 2, 3]);
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let column = ctx.execute(CuDFColumn::from_arrow_host(&input))?.into_view();
    ///
    /// let casted = ctx.execute(column.cast(&DataType::Int64))?;
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn cast<'a>(
        &'a self,
        target_type: &'a DataType,
    ) -> impl CuDFOperation<Output = CuDFColumn> + 'a {
        deferred(move |ctx| {
            let cudf_dt = arrow_type_to_cudf_data_type(target_type).ok_or_else(|| {
                CuDFError::ArrowError(ArrowError::NotYetImplemented(format!(
                    "Arrow type {} not supported in cuDF cast",
                    target_type
                )))
            })?;
            let mut launch = execution_policy::launch(ctx)?;
            launch.wait_column(self)?;
            let result =
                ffi::cast_column(self.inner(), &cudf_dt, launch.stream()?, launch.resource())?;
            launch.ready_column(CuDFColumn::from_inner(result))
        })
    }

    /// Slice this column view on the GPU.
    ///
    /// The returned operation produces a new [`CuDFColumnView`] covering
    /// `offset..offset + length`. The sliced view keeps this source column
    /// alive and inherits its stream readiness. This method is named
    /// `slice_view` to avoid shadowing Arrow's [`Array::slice`](arrow::array::Array::slice)
    /// on `CuDFColumnView`.
    ///
    /// # Errors
    ///
    /// Execution returns an error if the requested range is outside the column
    /// bounds or cuDF cannot create the slice view.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow::array::{Array, Int32Array};
    /// use libcudf_rs::{CuDFColumn, CuDFExecutionContext};
    ///
    /// let input = Int32Array::from(vec![1, 2, 3, 4, 5]);
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let column = ctx.execute(CuDFColumn::from_arrow_host(&input))?.into_view();
    ///
    /// let sliced = ctx.execute(column.slice_view(1, 3))?;
    /// assert_eq!(sliced.len(), 3);
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn slice_view(
        &self,
        offset: usize,
        length: usize,
    ) -> impl CuDFOperation<Output = CuDFColumnView> + '_ {
        deferred(move |ctx| {
            let mut launch = execution_policy::launch(ctx)?;
            launch.wait_column(self)?;
            let inner = ffi::slice_column(self.inner(), offset, length, launch.stream()?)?;
            // cuDF slice returns a view over the original buffers. No new producer
            // dependency is recorded; the sliced view inherits the input readiness.
            let storage: CuDFViewStorage = Arc::new(self.clone());
            Ok(CuDFColumnView::from_view(
                inner,
                Some(storage),
                self.stream_readiness().cloned(),
            ))
        })
    }
}

impl Clone for CuDFColumnView {
    fn clone(&self) -> Self {
        let cloned_inner = self.inner.clone();
        Self {
            inner: cloned_inner,
            storage: self.storage.clone(),
            dt: self.dt.clone(),
            null_buf: self.null_buf.clone(),
            stream_readiness: self.stream_readiness.clone(),
        }
    }
}

impl CuDFStreamReady for CuDFColumnView {
    fn wait_ready_on_stream(&self, stream: &ffi::CudaStreamView) -> Result<(), CuDFError> {
        if let Some(dependency) = &self.stream_readiness {
            dependency.wait_on_stream(stream)?;
        }
        Ok(())
    }
}

impl Debug for CuDFColumnView {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CuDFColumnView: type={}, size={}",
            self.data_type(),
            self.len()
        )
    }
}

unsafe impl Array for CuDFColumnView {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn to_data(&self) -> ArrayData {
        // In debug/test builds this panics so implicit GPU->CPU copies are easily
        // caught.
        //
        // Call to_arrow_host() explicitly rather than this method to separate
        // implicit and explicit transfers.
        //
        // TODO: Avoid fallible cuDF work in Arrow's infallible Array trait.
        debug_assert!(
            false,
            "CuDFColumnView::to_data(), implicit GPU->CPU copy detected. \
             Call to_arrow_host() explicitly."
        );
        CuDFExecutionContext::try_new_non_blocking()
            .expect("failed to create cuDF execution context")
            .execute(self.to_arrow_host())
            .expect("Failed to convert GPU column to host Arrow")
            .to_data()
    }

    fn into_data(self) -> ArrayData {
        self.to_data()
    }

    fn data_type(&self) -> &DataType {
        &self.dt
    }

    fn slice(&self, offset: usize, length: usize) -> ArrayRef {
        // TODO: Avoid fallible cuDF work in Arrow's infallible Array trait.
        Arc::new(
            CuDFExecutionContext::try_new_non_blocking()
                .expect("failed to create cuDF execution context")
                .execute(self.slice_view(offset, length))
                .expect("Failed to slice column"),
        )
    }

    fn len(&self) -> usize {
        self.inner.size()
    }

    fn is_empty(&self) -> bool {
        self.inner.size() == 0
    }

    fn offset(&self) -> usize {
        self.inner.offset() as usize
    }

    fn nulls(&self) -> Option<&NullBuffer> {
        // TODO: Avoid fallible cuDF work in Arrow's infallible Array trait.
        self.null_buf
            .get_or_init(|| {
                if self.null_count() == 0 {
                    return None;
                }
                let ctx = CuDFExecutionContext::try_new_non_blocking()
                    .expect("failed to create cuDF execution context");
                let mut launch =
                    execution_policy::launch(&ctx).expect("failed to launch cuDF operation");
                launch
                    .wait_column(self)
                    .expect("failed to wait for cuDF column readiness");
                launch
                    .stream()
                    .expect("cuDF stream should not be null")
                    .synchronize()
                    .expect("failed to synchronize default cuDF stream");
                let null_bytes = self.inner().get_null_buffer();
                let offset = self.inner().offset() as usize;
                let length = self.inner().size();

                let buffer = Buffer::from_vec(null_bytes);
                let boolean_buffer = BooleanBuffer::new(buffer, offset, length);
                Some(NullBuffer::new(boolean_buffer))
            })
            .as_ref()
    }

    fn get_buffer_memory_size(&self) -> usize {
        CuDFExecutionContext::try_new_non_blocking()
            .expect("failed to create cuDF execution context")
            .execute(self.buffer_memory_size())
            .expect("failed to get cuDF column buffer memory size")
    }

    fn get_array_memory_size(&self) -> usize {
        CuDFExecutionContext::try_new_non_blocking()
            .expect("failed to create cuDF execution context")
            .execute(self.array_memory_size())
            .expect("failed to get cuDF column array memory size")
    }

    fn logical_nulls(&self) -> Option<NullBuffer> {
        // TODO: For now only primitive types are supported
        // In this case logical_nulls == physical_nulls
        self.nulls().cloned()
    }

    fn is_null(&self, index: usize) -> bool {
        match self.nulls() {
            Some(nulls) => nulls.is_null(index),
            None => false,
        }
    }

    fn is_valid(&self, index: usize) -> bool {
        !self.is_null(index)
    }

    fn null_count(&self) -> usize {
        self.inner.null_count() as usize
    }

    fn logical_null_count(&self) -> usize {
        // TODO: For now only primitive types are supported
        // In this case logical_null_count == physical_null_count
        self.null_count()
    }

    fn is_nullable(&self) -> bool {
        // All cuDF columns can potentially have nulls
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CuDFColumn;
    use arrow::array::{Int32Array, Int64Array, StringArray};

    #[test]
    fn test_column_view_clone() -> Result<(), Box<dyn std::error::Error>> {
        let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        let cloned = column.clone();

        assert_eq!(column.len(), 5);
        assert_eq!(cloned.len(), 5);

        let original_ptr = column.inner.data_ptr();
        let cloned_ptr = cloned.inner.data_ptr();
        assert_eq!(
            original_ptr, cloned_ptr,
            "Cloned view should point to the same GPU memory"
        );

        Ok(())
    }

    #[test]
    fn test_string_column_view_memory_size() -> Result<(), Box<dyn std::error::Error>> {
        let array = StringArray::from(vec!["hello", "world", ""]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))?.into_view();

        let size = crate::execute_cudf(column.array_memory_size())?;
        let min_offsets_and_chars =
            (array.len() + 1) * std::mem::size_of::<i32>() + array.value_data().len();
        assert!(size >= min_offsets_and_chars);

        Ok(())
    }

    #[test]
    fn test_multiple_clones() -> Result<(), Box<dyn std::error::Error>> {
        let array = Int32Array::from(vec![10, 20, 30]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        let clone1 = column.clone();
        let clone2 = column.clone();
        let clone3 = clone1.clone();

        let ptr = column.inner.data_ptr();
        assert_eq!(clone1.inner.data_ptr(), ptr);
        assert_eq!(clone2.inner.data_ptr(), ptr);
        assert_eq!(clone3.inner.data_ptr(), ptr);

        assert_eq!(column.len(), 3);
        assert_eq!(clone1.len(), 3);
        assert_eq!(clone2.len(), 3);
        assert_eq!(clone3.len(), 3);

        drop(column);
        assert_eq!(clone1.inner.data_ptr(), ptr);
        assert_eq!(clone1.len(), 3);

        Ok(())
    }

    #[test]
    fn test_column_data_is_on_gpu() -> Result<(), Box<dyn std::error::Error>> {
        let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        let ptr = column.inner.data_ptr();

        assert_ne!(ptr, 0, "Column data pointer should not be null");

        let mut attrs: cuda_runtime_sys::cudaPointerAttributes = unsafe { std::mem::zeroed() };
        let result = unsafe {
            cuda_runtime_sys::cudaPointerGetAttributes(
                &mut attrs as *mut _,
                ptr as *const std::ffi::c_void,
            )
        };

        assert_eq!(
            result,
            cuda_runtime_sys::cudaError_t::cudaSuccess,
            "cudaPointerGetAttributes should succeed"
        );

        assert_ne!(
            attrs.type_,
            cuda_runtime_sys::cudaMemoryType::cudaMemoryTypeHost,
            "Column data should NOT be in host memory"
        );

        match attrs.type_ {
            cuda_runtime_sys::cudaMemoryType::cudaMemoryTypeDevice => {}
            cuda_runtime_sys::cudaMemoryType::cudaMemoryTypeUnregistered => {}
            _ => {
                panic!("Unexpected memory type: {:?}", attrs.type_);
            }
        }

        Ok(())
    }

    #[test]
    fn test_get_buffer_memory_size_int32() -> Result<(), Box<dyn std::error::Error>> {
        let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        let size = crate::execute_cudf(column.buffer_memory_size())?;
        assert_eq!(size, 20, "Int32 column should be 20 bytes");

        Ok(())
    }

    #[test]
    fn test_get_buffer_memory_size_large() -> Result<(), Box<dyn std::error::Error>> {
        let data: Vec<i32> = (0..1_000_000).collect();
        let array = Int32Array::from(data);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        let size = crate::execute_cudf(column.buffer_memory_size())?;
        assert_eq!(size, 4_000_000, "1M Int32 elements should be 4MB");

        Ok(())
    }

    #[test]
    fn test_get_array_memory_size_no_nulls() -> Result<(), Box<dyn std::error::Error>> {
        let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        let buffer_size = crate::execute_cudf(column.buffer_memory_size())?;
        let array_size = crate::execute_cudf(column.array_memory_size())?;

        // With no nulls, array_size should equal buffer_size
        assert_eq!(
            array_size, buffer_size,
            "Array size should equal buffer size when no nulls"
        );

        Ok(())
    }

    #[test]
    fn test_get_array_memory_size_with_nulls() -> Result<(), Box<dyn std::error::Error>> {
        let array = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5)]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        let buffer_size = crate::execute_cudf(column.buffer_memory_size())?;
        let array_size = crate::execute_cudf(column.array_memory_size())?;

        // With nulls -> array_size = buffer_size + null_mask_size
        assert!(
            array_size > buffer_size,
            "Array size should be larger than buffer size when nulls present"
        );

        let null_overhead = array_size - buffer_size;
        assert!(
            (1..=128).contains(&null_overhead),
            "Null mask overhead should be between 2 and 128 bytes, got {}",
            null_overhead
        );

        Ok(())
    }

    #[test]
    fn test_nulls_no_nulls_returns_none() -> Result<(), Box<dyn std::error::Error>> {
        let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        let nulls = column.nulls();
        assert!(nulls.is_none(), "Column with no nulls should return None");

        Ok(())
    }

    #[test]
    fn test_nulls_with_nulls_returns_buffer() -> Result<(), Box<dyn std::error::Error>> {
        let array = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5)]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        let nulls = column.nulls();
        assert!(
            nulls.is_some(),
            "Column with nulls should return Some(NullBuffer)"
        );

        Ok(())
    }

    #[test]
    fn test_nulls_correct_bit_pattern() -> Result<(), Box<dyn std::error::Error>> {
        let array = Int32Array::from(vec![Some(10), None, Some(30), None, Some(50)]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        let nulls = column.nulls().expect("Should have null buffer");
        assert!(nulls.is_valid(0), "Element 0 should be valid (Some(10))");
        assert!(nulls.is_null(1), "Element 1 should be null");
        assert!(nulls.is_valid(2), "Element 2 should be valid (Some(30))");
        assert!(nulls.is_null(3), "Element 3 should be null");
        assert!(nulls.is_valid(4), "Element 4 should be valid (Some(50))");

        Ok(())
    }

    #[test]
    fn test_nulls_cached_same_reference() -> Result<(), Box<dyn std::error::Error>> {
        let array = Int32Array::from(vec![Some(1), None, Some(3)]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        let nulls1 = column.nulls();
        let nulls2 = column.nulls();

        // Should return references to the same cached NullBuffer
        assert!(nulls1.is_some());
        assert!(nulls2.is_some());

        let ptr1 = nulls1.unwrap() as *const _;
        let ptr2 = nulls2.unwrap() as *const _;
        assert_eq!(
            ptr1, ptr2,
            "Subsequent calls should return same cached buffer"
        );

        Ok(())
    }

    #[test]
    fn test_is_null_uses_null_buffer() -> Result<(), Box<dyn std::error::Error>> {
        let array = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5)]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        assert!(!column.is_null(0), "Index 0 should not be null");
        assert!(column.is_null(1), "Index 1 should be null");
        assert!(!column.is_null(2), "Index 2 should not be null");
        assert!(column.is_null(3), "Index 3 should be null");
        assert!(!column.is_null(4), "Index 4 should not be null");

        Ok(())
    }

    #[test]
    fn test_is_valid_inverse_of_is_null() -> Result<(), Box<dyn std::error::Error>> {
        let array = Int32Array::from(vec![Some(1), None, Some(3)]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        for i in 0..3 {
            assert_eq!(
                column.is_valid(i),
                !column.is_null(i),
                "is_valid should be inverse of is_null at index {}",
                i
            );
        }

        Ok(())
    }

    #[test]
    fn test_null_count_matches_null_buffer() -> Result<(), Box<dyn std::error::Error>> {
        let array = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5), None]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        let null_count = column.null_count();

        // Should have 3 nulls (indices 1, 3, 5)
        assert_eq!(null_count, 3, "Should have 3 nulls");

        let manual_count = (0..column.len()).filter(|&i| column.is_null(i)).count();
        assert_eq!(
            null_count, manual_count,
            "null_count() should match manual count"
        );

        Ok(())
    }

    #[test]
    fn test_nulls_with_large_column() -> Result<(), Box<dyn std::error::Error>> {
        let data: Vec<Option<i32>> = (0..10_000)
            .map(|i| if i % 2 == 0 { Some(i) } else { None })
            .collect();
        let array = Int32Array::from(data);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        let nulls = column.nulls().expect("Should have null buffer");

        // Verify pattern: even indices valid, odd indices null
        for i in 0..100 {
            if i % 2 == 0 {
                assert!(nulls.is_valid(i), "Even index {} should be valid", i);
            } else {
                assert!(nulls.is_null(i), "Odd index {} should be null", i);
            }
        }

        assert_eq!(column.null_count(), 5_000, "Should have 5,000 nulls");

        Ok(())
    }

    #[test]
    fn test_logical_nulls_equals_physical_nulls() -> Result<(), Box<dyn std::error::Error>> {
        // For primitive types, logical_nulls should equal physical nulls
        let array = Int32Array::from(vec![Some(1), None, Some(3)]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))
            .expect("Failed to convert Arrow array to column")
            .into_view();

        let physical = column.nulls();
        let logical = column.logical_nulls();

        match (physical, logical) {
            (Some(p), Some(l)) => {
                // Both should have same null pattern
                for i in 0..3 {
                    assert_eq!(
                        p.is_null(i),
                        l.is_null(i),
                        "Physical and logical nulls should match at index {}",
                        i
                    );
                }
            }
            (None, None) => {
                // Both None is valid
            }
            _ => {
                panic!("Physical and logical nulls should both be Some or both None");
            }
        }

        Ok(())
    }

    #[test]
    fn test_cast_int32_to_int64_with_nulls() -> Result<(), Box<dyn std::error::Error>> {
        let array = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5)]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))?.into_view();

        let casted = crate::execute_cudf(column.cast(&DataType::Int64))?;
        let view = casted.into_view();
        assert_eq!(view.data_type(), &DataType::Int64);

        let result = crate::execute_cudf(view.to_arrow_host())?;

        let result = result.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(result.len(), 5);
        assert!(result.is_valid(0));
        assert!(result.is_null(1));
        assert!(result.is_valid(2));
        assert!(result.is_null(3));
        assert!(result.is_valid(4));
        assert_eq!(result.value(0), 1);
        assert_eq!(result.value(2), 3);
        assert_eq!(result.value(4), 5);
        Ok(())
    }

    #[test]
    fn test_cast_unsupported_type_returns_error() -> Result<(), Box<dyn std::error::Error>> {
        let array = Int32Array::from(vec![1, 2, 3]);
        let column = crate::execute_cudf(CuDFColumn::from_arrow_host(&array))?.into_view();

        let result = crate::execute_cudf(column.cast(&DataType::Null));
        assert!(result.is_err());
        Ok(())
    }
}

use crate::cudf_reference::CuDFRef;
use crate::data_type::cudf_type_to_arrow;
use crate::{slice_column, CuDFError};
use arrow::array::{Array, ArrayData, ArrayRef};
use arrow::buffer::NullBuffer;
use arrow::compute::is_null;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_schema::DataType;
use cxx::UniquePtr;
use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, OnceLock};

/// A view into a cuDF column stored in GPU memory
///
/// This type wraps a cuDF column_view and implements Arrow's Array trait,
/// allowing cuDF columns to be used seamlessly with the Arrow ecosystem.
///
/// CuDFColumnView is a non-owning view that may optionally keep the underlying
/// column alive. It can be created from Arrow arrays (copying to GPU) or from
/// existing cuDF columns.
pub struct CuDFColumnView {
    // Keep a ref to CuDF structs so that they live as long as this view exists
    pub(crate) _ref: Option<Arc<dyn CuDFRef>>,
    inner: UniquePtr<libcudf_sys::ffi::ColumnView>,
    dt: DataType,
    null_buf: OnceLock<Option<NullBuffer>>,
}

impl CuDFColumnView {
    /// Create a [CuDFColumnView] from a column view and optional table reference
    ///
    /// This is used internally to create column views that keep the source table or column alive
    pub(crate) fn new_with_ref(
        inner: UniquePtr<libcudf_sys::ffi::ColumnView>,
        _ref: Option<Arc<dyn CuDFRef>>,
    ) -> Self {
        let cudf_dtype = inner.data_type();
        let dt = cudf_type_to_arrow(&cudf_dtype);
        let dt = dt.unwrap_or(DataType::Null);
        Self {
            _ref,
            inner,
            dt,
            null_buf: OnceLock::new(),
        }
    }

    pub(crate) fn inner(&self) -> &UniquePtr<libcudf_sys::ffi::ColumnView> {
        &self.inner
    }

    /// Consume this wrapper and return the underlying cuDF column view
    pub(crate) fn into_inner(self) -> UniquePtr<libcudf_sys::ffi::ColumnView> {
        self.inner
    }

    /// Convert the cuDF column view to an Arrow array, copying data from GPU to host
    ///
    /// This method copies the GPU data back to the CPU and creates an Arrow array.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The cuDF column cannot be converted to Arrow format
    /// - There is an error copying data from GPU to host
    ///
    /// # Example
    ///
    /// ```no_run
    /// use arrow::array::Int32Array;
    /// use libcudf_rs::CuDFColumn;
    ///
    /// let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    /// let column = CuDFColumn::from_arrow_host(&array)?.into_view();
    /// // Do some GPU processing...
    /// let result = column.to_arrow_host()?;
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn to_arrow_host(&self) -> Result<ArrayRef, CuDFError> {
        let mut device_array = libcudf_sys::ArrowDeviceArray {
            array: FFI_ArrowArray::empty(),
            device_id: -1,
            device_type: 1, // CPU
            sync_event: std::ptr::null_mut(),
            reserved: [0; 3],
        };

        // Create schema from the column's data type
        let ffi_schema = FFI_ArrowSchema::try_from(self.data_type())?;

        // Convert the column view to Arrow format (copying from GPU to host)
        // cuDF's to_arrow_array expects an ArrowDeviceArray pointer
        unsafe {
            let device_array_ptr =
                &mut device_array as *mut libcudf_sys::ArrowDeviceArray as *mut u8;
            self.inner.to_arrow_array(device_array_ptr);
        }

        // Convert from FFI structures to Arrow ArrayData
        // Extract just the ArrowArray part
        let array_data = unsafe { arrow::ffi::from_ffi(device_array.array, &ffi_schema)? };

        // Create an ArrayRef from the ArrayData
        Ok(arrow::array::make_array(array_data))
    }
}

impl Clone for CuDFColumnView {
    fn clone(&self) -> Self {
        // Clone the view using the FFI clone method
        let cloned_inner = self.inner.clone();
        Self {
            _ref: self._ref.clone(),
            inner: cloned_inner,
            dt: self.dt.clone(),
            null_buf: self.null_buf.clone(),
        }
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

impl Array for CuDFColumnView {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn to_data(&self) -> ArrayData {
        self.to_arrow_host()
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
        Arc::new(slice_column(self, offset, length).expect("Failed to slice column"))
    }

    fn len(&self) -> usize {
        self.inner.size()
    }

    fn is_empty(&self) -> bool {
        self.inner.size() == 0
    }

    fn offset(&self) -> usize {
        self.inner.offset()
    }

    fn nulls(&self) -> Option<&NullBuffer> {
        // TODO: This should use FFI to transfer ONLY the null bitmap from GPU,
        // not the entire column data.
        // Add column_view_get_null_buffer() to libcudf-sys and call it here.
        self.null_buf
            .get_or_init(|| {
                if self.null_count() == 0 {
                    return None;
                }
                let array_data = self.to_data();
                array_data.nulls().cloned()
            })
            .as_ref()
    }

    fn get_buffer_memory_size(&self) -> usize {
        // TODO: This should use FFI to query cuDF metadata directly instead of
        // transferring all data from GPU to CPU (no allocation)
        // Add column_view_get_buffer_memory_size() to libcudf-sys and call it here.
        let data = self.to_data();
        data.buffers().iter().map(|b| b.len()).sum()
    }

    fn get_array_memory_size(&self) -> usize {
        // TODO: This should use FFI to query cuDF metadata directly instead of
        // transferring all data from GPU to CPU (no allocation).
        // Add column_view_get_array_memory_size() to libcudf-sys and call it here.
        let data = self.to_data();
        let buffer_size: usize = data.buffers().iter().map(|b| b.len()).sum();
        let null_size = data.nulls().map(|n| n.buffer().len()).unwrap_or(0);
        let child_size: usize = (0..data.num_children())
            .filter_map(|i| data.child_data().get(i))
            .map(|child| child.get_array_memory_size())
            .sum();
        buffer_size + null_size + child_size
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
    use arrow::array::Int32Array;

    #[test]
    fn test_column_view_clone() {
        let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let column = CuDFColumn::from_arrow_host(&array)
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
    }

    #[test]
    fn test_multiple_clones() {
        let array = Int32Array::from(vec![10, 20, 30]);
        let column = CuDFColumn::from_arrow_host(&array)
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
    }

    #[test]
    fn test_column_data_is_on_gpu() {
        let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let column = CuDFColumn::from_arrow_host(&array)
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
    }
}

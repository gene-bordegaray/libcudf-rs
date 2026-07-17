use crate::device_resource::resource_ref;
use crate::stream::stream_ref;
use crate::{CuDFColumn, CuDFError, CuDFScalar};
use arrow::array::{Array, ArrayData, ArrayRef, Scalar};
use arrow::buffer::NullBuffer;
use libcudf_sys::ffi;
use std::any::Any;
use std::sync::Arc;

impl CuDFScalar {
    /// Convert the cuDF scalar to a single-element host Arrow array.
    ///
    /// This copies the scalar from GPU to host memory.
    pub fn to_arrow_host(&self) -> Result<ArrayRef, CuDFError> {
        let mut device_array = libcudf_sys::ArrowDeviceArray::new_cpu();

        unsafe {
            let device_array_ptr =
                &mut device_array as *mut libcudf_sys::ArrowDeviceArray as *mut u8;
            let stream = ffi::get_default_stream();
            let resource = ffi::get_current_device_resource_ref();
            let column = ffi::make_column_from_scalar(
                self.inner(),
                1,
                stream_ref(&stream)?,
                resource_ref(&resource)?,
            )?;
            let column_view = column.view()?;
            ffi::to_arrow_host_column(
                &column_view,
                device_array_ptr,
                stream_ref(&stream)?,
                resource_ref(&resource)?,
            )?;
            stream_ref(&stream)?.synchronize()?;
        }

        let array_data = unsafe {
            arrow::ffi::from_ffi_and_data_type(
                device_array.array,
                self.data_type_metadata().clone(),
            )?
        };
        Ok(arrow::array::make_array(array_data))
    }

    /// Convert a host Arrow scalar to a cuDF scalar.
    ///
    /// This uploads a single-element column and extracts its first element.
    pub fn try_from_arrow_host<T: Array>(scalar: Scalar<T>) -> Result<Self, CuDFError> {
        let column = CuDFColumn::try_from_arrow_host(&scalar.into_inner())?.into_view();
        let stream = ffi::get_default_stream();
        let resource = ffi::get_current_device_resource_ref();
        let inner = ffi::get_element(
            column.inner(),
            0,
            stream_ref(&stream)?,
            resource_ref(&resource)?,
        )?;
        Self::try_from_inner(inner)
    }

    /// Return cached host Arrow data for this scalar, materializing it if needed.
    pub fn get_scalar_data(&self) -> Result<Arc<ArrayData>, CuDFError> {
        if let Some(cached_data) = self.cached_scalar_data() {
            return Ok(cached_data);
        }

        let array_data = Arc::new(self.to_arrow_host()?.to_data());
        self.cache_scalar_data(Arc::clone(&array_data));
        Ok(array_data)
    }
}

unsafe impl Array for CuDFScalar {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn to_data(&self) -> ArrayData {
        self.to_arrow_host()
            .expect("Failed to convert GPU scalar to host Arrow")
            .to_data()
    }

    fn into_data(self) -> ArrayData {
        self.to_data()
    }

    fn data_type(&self) -> &arrow_schema::DataType {
        self.data_type_metadata()
    }

    fn slice(&self, offset: usize, length: usize) -> ArrayRef {
        if offset >= 1 || length == 0 {
            arrow::array::new_empty_array(self.data_type_metadata())
        } else {
            Arc::new(self.clone())
        }
    }

    fn len(&self) -> usize {
        1
    }

    fn is_empty(&self) -> bool {
        false
    }

    fn offset(&self) -> usize {
        0
    }

    fn nulls(&self) -> Option<&NullBuffer> {
        None
    }

    fn get_buffer_memory_size(&self) -> usize {
        self.get_scalar_data()
            .map(|data| data.get_buffer_memory_size())
            .unwrap_or(0)
    }

    fn get_array_memory_size(&self) -> usize {
        self.get_scalar_data()
            .map(|data| data.get_array_memory_size())
            .unwrap_or(0)
    }
}

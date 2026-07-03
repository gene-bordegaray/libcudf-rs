use crate::data_type::cudf_type_to_arrow;
use crate::stream_readiness::{CuDFStreamDependency, CuDFStreamReady};
use crate::{CuDFError, CuDFExecutionContext};
use arrow::array::{new_empty_array, Array, ArrayData, ArrayRef};
use arrow::buffer::NullBuffer;
use arrow_schema::DataType;
use cxx::UniquePtr;
use libcudf_sys::ffi;
use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, RwLock};

/// A single cuDF value stored in GPU memory.
///
/// `CuDFScalar` wraps a cuDF scalar and implements Arrow's [`Array`] trait with
/// length 1. That lets scalar values participate in APIs that accept either GPU
/// columns or scalar arrays.
pub struct CuDFScalar {
    inner: UniquePtr<ffi::Scalar>,
    dt: DataType,
    cached_scalar: RwLock<Option<Arc<ArrayData>>>,
    stream_readiness: Option<CuDFStreamDependency>,
}

impl CuDFScalar {
    pub(crate) fn new(inner: UniquePtr<ffi::Scalar>) -> Self {
        let cudf_dtype = inner.data_type();
        let dt = cudf_type_to_arrow(&cudf_dtype);
        let dt = dt.unwrap_or(DataType::Null);
        let cached_scalar = RwLock::new(None);
        Self {
            inner,
            dt,
            cached_scalar,
            stream_readiness: None,
        }
    }

    pub(crate) fn with_stream_readiness(mut self, dependency: CuDFStreamDependency) -> Self {
        self.stream_readiness = Some(dependency);
        self
    }

    pub(crate) fn inner(&self) -> &UniquePtr<ffi::Scalar> {
        &self.inner
    }

    fn cached_array_data(&self) -> Result<Arc<ArrayData>, CuDFError> {
        if let Ok(cache) = self.cached_scalar.read() {
            if let Some(cached_data) = cache.as_ref() {
                return Ok(Arc::clone(cached_data));
            }
        }

        let array_ref =
            CuDFExecutionContext::try_new_non_blocking()?.execute(self.to_arrow_host())?;
        let array_data = Arc::new(array_ref.to_data());
        if let Ok(mut cache) = self.cached_scalar.write() {
            *cache = Some(Arc::clone(&array_data));
        }

        Ok(array_data)
    }
}

impl Debug for CuDFScalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "CuDFScalar: type={}", self.dt)
    }
}

impl Clone for CuDFScalar {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            dt: self.dt.clone(),
            cached_scalar: RwLock::new(None),
            stream_readiness: self.stream_readiness.clone(),
        }
    }
}

impl CuDFStreamReady for CuDFScalar {
    fn wait_ready_on_stream(&self, stream: &ffi::CudaStreamView) -> Result<(), CuDFError> {
        if let Some(dependency) = &self.stream_readiness {
            dependency.wait_on_stream(stream)?;
        }
        Ok(())
    }
}

unsafe impl Array for CuDFScalar {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn to_data(&self) -> ArrayData {
        // WARNING: This performs a full GPU to CPU data transfer.
        //
        // TODO: Avoid fallible cuDF work in Arrow's infallible Array trait.
        CuDFExecutionContext::try_new_non_blocking()
            .expect("failed to create cuDF execution context")
            .execute(self.to_arrow_host())
            .expect("Failed to convert GPU scalar to host Arrow")
            .to_data()
    }

    fn into_data(self) -> ArrayData {
        self.to_data()
    }

    fn data_type(&self) -> &DataType {
        &self.dt
    }

    fn slice(&self, offset: usize, length: usize) -> ArrayRef {
        if offset >= 1 || length == 0 {
            new_empty_array(&self.dt)
        } else {
            Arc::new(Self {
                inner: self.inner.clone(),
                dt: self.dt.clone(),
                cached_scalar: RwLock::new(None),
                stream_readiness: self.stream_readiness.clone(),
            })
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
        self.cached_array_data()
            .map(|data| data.get_buffer_memory_size())
            .unwrap_or(0)
    }

    fn get_array_memory_size(&self) -> usize {
        self.cached_array_data()
            .map(|data| data.get_array_memory_size())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::*;

    /// Assert Array trait basic properties
    fn assert_array_trait_basics(scalar: &CuDFScalar, expected_type: &DataType) {
        assert_eq!(scalar.len(), 1, "Scalar length should always be 1");
        assert!(!scalar.is_empty(), "Scalar should never be empty");
        assert_eq!(scalar.offset(), 0, "Scalar offset should always be 0");
        assert_eq!(scalar.data_type(), expected_type, "Data type mismatch");
        assert!(
            scalar.nulls().is_none(),
            "nulls() should return None for scalars"
        );
    }

    /// Test comprehensive slice behavior for a scalar
    fn assert_slice_behavior(scalar: &CuDFScalar) {
        let dt = scalar.data_type();

        // slice(0, 1) -> returns self
        let sliced = scalar.slice(0, 1);
        assert_eq!(sliced.len(), 1, "slice(0, 1) should return self");
        assert_eq!(sliced.data_type(), dt, "slice(0, 1) should preserve type");

        // slice(0, 0) -> returns empty
        let empty = scalar.slice(0, 0);
        assert_eq!(empty.len(), 0, "slice(0, 0) should return empty array");
        assert_eq!(empty.data_type(), dt, "Empty slice should preserve type");

        // slice(1, 1) -> returns empty (offset beyond bounds)
        let oob = scalar.slice(1, 1);
        assert_eq!(oob.len(), 0, "slice(1, 1) should return empty array");

        // slice(5, 10) -> returns empty (way out of bounds)
        let far_oob = scalar.slice(5, 10);
        assert_eq!(far_oob.len(), 0, "slice(5, 10) should return empty array");

        // slice(0, 100) -> returns self (length clamped)
        let oversized = scalar.slice(0, 100);
        assert_eq!(
            oversized.len(),
            1,
            "slice(0, 100) should return self (clamped)"
        );
    }

    /// Verify slice clears cache
    fn assert_slice_clears_cache(scalar: &CuDFScalar) {
        // First call to populate cache
        let _ = scalar.get_buffer_memory_size();

        // Slice creates new instance
        let sliced = scalar.slice(0, 1);

        // Both scalars should still work independently
        assert!(scalar.get_buffer_memory_size() > 0);
        assert!(sliced.get_buffer_memory_size() > 0);
    }

    /// Assert memory size for a scalar
    fn assert_memory_size(scalar: &CuDFScalar, expected_buffer_size: usize, type_name: &str) {
        let buffer_size = scalar.get_buffer_memory_size();
        let array_size = scalar.get_array_memory_size();

        assert_eq!(
            buffer_size, expected_buffer_size,
            "Buffer size mismatch for {}",
            type_name
        );

        // Array size should include null bitmap overhead
        assert!(
            array_size >= buffer_size,
            "Array size ({}) should be >= buffer size ({})",
            array_size,
            buffer_size
        );
    }

    #[test]
    fn test_array_trait_slice_behavior_int32() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![42])));
        assert_slice_behavior(&scalar);

        Ok(())
    }

    #[test]
    fn test_array_trait_slice_behavior_float64() -> Result<(), Box<dyn std::error::Error>> {
        let scalar =
            scalar_from_arrow(Scalar::new(&Float64Array::from(vec![std::f64::consts::PI])));
        assert_slice_behavior(&scalar);

        Ok(())
    }

    #[test]
    fn test_array_trait_slice_behavior_string() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&StringArray::from(vec!["test"])));
        assert_slice_behavior(&scalar);

        Ok(())
    }

    #[test]
    fn test_array_trait_slice_behavior_null() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![None])));
        assert_slice_behavior(&scalar);

        Ok(())
    }

    #[test]
    fn test_array_trait_slice_clears_cache() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&Int64Array::from(vec![999])));
        assert_slice_clears_cache(&scalar);

        Ok(())
    }

    #[test]
    fn test_array_trait_basics_int32() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![42])));
        assert_array_trait_basics(&scalar, &DataType::Int32);

        Ok(())
    }

    #[test]
    fn test_array_trait_basics_string() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&StringArray::from(vec!["test"])));
        assert_array_trait_basics(&scalar, &DataType::Utf8);

        Ok(())
    }

    #[test]
    fn test_array_trait_basics_boolean() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&BooleanArray::from(vec![true])));
        assert_array_trait_basics(&scalar, &DataType::Boolean);

        Ok(())
    }

    #[test]
    fn test_array_trait_basics_null() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![None])));
        assert_array_trait_basics(&scalar, &DataType::Int32);

        Ok(())
    }

    #[test]
    fn test_as_any_downcast() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![42])));
        assert!(scalar.as_any().downcast_ref::<CuDFScalar>().is_some());

        Ok(())
    }

    #[test]
    fn test_caching_works_int32() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![42])));
        let size1 = scalar.get_buffer_memory_size();
        let size2 = scalar.get_buffer_memory_size();
        assert_eq!(size1, size2, "Cached and uncached sizes should match");

        Ok(())
    }

    #[test]
    fn test_caching_works_string() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&StringArray::from(vec!["cached"])));
        let size1 = scalar.get_buffer_memory_size();
        let size2 = scalar.get_buffer_memory_size();
        assert_eq!(size1, size2, "Cached and uncached sizes should match");

        Ok(())
    }

    #[test]
    fn test_caching_memory_sizes() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&Int64Array::from(vec![12345])));

        // Buffer size caching
        let size1 = scalar.get_buffer_memory_size();
        assert!(size1 > 0, "Buffer size should be positive");
        let size2 = scalar.get_buffer_memory_size();
        assert_eq!(size1, size2, "Cached buffer size should match");

        // Array size caching
        let array_size1 = scalar.get_array_memory_size();
        assert!(array_size1 > 0, "Array size should be positive");
        let array_size2 = scalar.get_array_memory_size();
        assert_eq!(array_size1, array_size2, "Cached array size should match");

        Ok(())
    }

    #[test]
    fn test_caching_clone_clears_cache() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![42])));

        // Populate cache
        let _ = scalar.get_buffer_memory_size();

        // Clone should have empty cache
        let cloned = scalar.clone();

        assert!(scalar.get_buffer_memory_size() > 0);
        assert!(cloned.get_buffer_memory_size() > 0);

        Ok(())
    }

    #[test]
    fn test_caching_slice_clears_cache() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![42])));

        // Populate original cache
        let original_size = scalar.get_buffer_memory_size();

        // Slice creates new instance with empty cache
        let sliced = scalar.slice(0, 1);
        let sliced_size = sliced.get_buffer_memory_size();

        // Both should have same size but independent caches
        assert_eq!(original_size, sliced_size);

        Ok(())
    }

    #[test]
    fn test_memory_size() -> Result<(), Box<dyn std::error::Error>> {
        let int32_scalar = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![42])));
        assert_memory_size(&int32_scalar, 4, "Int32");

        let int64_scalar = scalar_from_arrow(Scalar::new(&Int64Array::from(vec![12345])));
        assert_memory_size(&int64_scalar, 8, "Int64");

        let float_scalar =
            scalar_from_arrow(Scalar::new(&Float64Array::from(vec![std::f64::consts::PI])));
        assert_memory_size(&float_scalar, 8, "Float64");

        let bool_scalar = scalar_from_arrow(Scalar::new(&BooleanArray::from(vec![true])));
        let buffer_size = bool_scalar.get_buffer_memory_size();
        let array_size = bool_scalar.get_array_memory_size();
        assert!(buffer_size > 0, "Boolean buffer size should be positive");
        assert!(
            array_size >= buffer_size,
            "Array size should be >= buffer size"
        );

        let test_string = "hello world";
        let string_scalar = scalar_from_arrow(Scalar::new(&StringArray::from(vec![test_string])));
        let buffer_size = string_scalar.get_buffer_memory_size();
        let array_size = string_scalar.get_array_memory_size();
        // For strings, size should be at least the string length
        assert!(
            buffer_size >= test_string.len(),
            "Buffer size should be at least string length"
        );
        assert!(array_size >= buffer_size);

        let scalar = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![None])));
        let buffer_size = scalar.get_buffer_memory_size();
        let array_size = scalar.get_array_memory_size();
        // Null scalar should still have some memory overhead
        assert!(
            array_size >= buffer_size,
            "Array size should be >= buffer size"
        );

        Ok(())
    }

    #[test]
    fn test_null_handling() -> Result<(), Box<dyn std::error::Error>> {
        let null_scalar = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![None])));
        assert!(!null_scalar.inner().is_valid(), "Scalar should be null");

        let valid_scalar = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![42])));
        assert!(valid_scalar.inner().is_valid(), "Scalar should be valid");

        Ok(())
    }

    #[test]
    fn test_data_conversions() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![42])));

        let data = scalar.to_data();
        assert_eq!(data.len(), 1);
        assert_eq!(data.data_type(), &DataType::Int32);

        let scalar2 = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![42])));
        let data2 = scalar2.into_data();
        assert_eq!(data2.len(), 1);
        assert_eq!(data2.data_type(), &DataType::Int32);

        Ok(())
    }

    #[test]
    fn test_data_type_preserved_int32() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![42])));
        assert_eq!(scalar.data_type(), &DataType::Int32);

        Ok(())
    }

    #[test]
    fn test_data_type_preserved_float64() -> Result<(), Box<dyn std::error::Error>> {
        let scalar =
            scalar_from_arrow(Scalar::new(&Float64Array::from(vec![std::f64::consts::PI])));
        assert_eq!(scalar.data_type(), &DataType::Float64);

        Ok(())
    }

    #[test]
    fn test_data_type_preserved_string() -> Result<(), Box<dyn std::error::Error>> {
        let scalar = scalar_from_arrow(Scalar::new(&StringArray::from(vec!["test"])));
        assert_eq!(scalar.data_type(), &DataType::Utf8);

        Ok(())
    }

    #[test]
    fn test_scalar_clone() -> Result<(), Box<dyn std::error::Error>> {
        let original = scalar_from_arrow(Scalar::new(&Int32Array::from(vec![999])));
        let cloned = original.clone();
        assert_eq!(cloned.data_type(), original.data_type());

        Ok(())
    }

    fn scalar_from_arrow<T: Array>(scalar: Scalar<T>) -> CuDFScalar {
        crate::execute_cudf(CuDFScalar::from_arrow_host(scalar)).expect("Failed to create scalar")
    }
}

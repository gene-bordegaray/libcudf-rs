use crate::cudf_array::is_cudf_array;
use crate::device_resource::resource_ref;
use crate::stream::stream_ref;
use crate::{CuDFError, CuDFTable};
use arrow::array::{Array, ArrayData, StructArray};
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow::record_batch::RecordBatch;
use arrow_schema::ArrowError;
use libcudf_sys::{ffi, ArrowDeviceArray};

impl CuDFTable {
    /// Create a table from a host Arrow `RecordBatch`.
    ///
    /// The Arrow buffers are copied to GPU memory.
    pub fn try_from_arrow_host(batch: RecordBatch) -> Result<Self, CuDFError> {
        crate::config::ensure_pools_configured()?;
        if batch.columns().iter().any(|column| is_cudf_array(column)) {
            return Err(ArrowError::InvalidArgumentError(
                "cannot upload a RecordBatch containing a cuDF array".into(),
            )
            .into());
        }

        let schema = batch.schema().as_ref().clone();
        let array_data: ArrayData = StructArray::from(batch).into_data();
        let ffi_array = FFI_ArrowArray::new(&array_data);
        let ffi_schema = FFI_ArrowSchema::try_from(schema)?;
        let device_array = ArrowDeviceArray::new_cpu().with_array(ffi_array);
        let stream = ffi::get_default_stream();
        let resource = ffi::get_current_device_resource_ref();

        let inner = unsafe {
            ffi::from_arrow_host(
                &ffi_schema as *const FFI_ArrowSchema as *const u8,
                &device_array as *const ArrowDeviceArray as *const u8,
                stream_ref(&stream)?,
                resource_ref(&resource)?,
            )
        }?;
        Self::try_from_inner(inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::*;
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use std::sync::Arc;

    #[test]
    fn arrow_roundtrip_preserves_supported_types() -> Result<(), Box<dyn std::error::Error>> {
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
        let arrays: Vec<Arc<dyn Array>> = vec![
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

        let result = CuDFTable::try_from_arrow_host(batch.clone())?
            .into_view()
            .to_arrow_host()?;

        assert_eq!(result.num_rows(), batch.num_rows());
        assert_eq!(result.num_columns(), batch.num_columns());
        for (expected, actual) in batch.columns().iter().zip(result.columns()) {
            assert_eq!(expected, actual);
        }
        Ok(())
    }

    #[test]
    fn empty_arrow_batch_roundtrips() -> Result<(), Box<dyn std::error::Error>> {
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

        let result = CuDFTable::try_from_arrow_host(batch)?
            .into_view()
            .to_arrow_host()?;
        assert_eq!(result.num_rows(), 0);
        assert_eq!(result.num_columns(), 2);
        Ok(())
    }
}

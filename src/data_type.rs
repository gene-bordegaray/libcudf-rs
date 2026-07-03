use arrow_schema::{DataType, TimeUnit};
use cxx::UniquePtr;
use libcudf_sys::ffi;

// Numeric values mirror cudf::type_id in the local cuDF header.
const TYPE_EMPTY: i32 = 0;
const TYPE_INT8: i32 = 1;
const TYPE_INT16: i32 = 2;
const TYPE_INT32: i32 = 3;
const TYPE_INT64: i32 = 4;
const TYPE_UINT8: i32 = 5;
const TYPE_UINT16: i32 = 6;
const TYPE_UINT32: i32 = 7;
const TYPE_UINT64: i32 = 8;
const TYPE_FLOAT32: i32 = 9;
const TYPE_FLOAT64: i32 = 10;
const TYPE_BOOL8: i32 = 11;
const TYPE_TIMESTAMP_DAYS: i32 = 12;
const TYPE_TIMESTAMP_SECONDS: i32 = 13;
const TYPE_TIMESTAMP_MILLISECONDS: i32 = 14;
const TYPE_TIMESTAMP_MICROSECONDS: i32 = 15;
const TYPE_TIMESTAMP_NANOSECONDS: i32 = 16;
const TYPE_DURATION_DAYS: i32 = 17;
const TYPE_DURATION_SECONDS: i32 = 18;
const TYPE_DURATION_MILLISECONDS: i32 = 19;
const TYPE_DURATION_MICROSECONDS: i32 = 20;
const TYPE_DURATION_NANOSECONDS: i32 = 21;
const TYPE_DICTIONARY32: i32 = 22;
const TYPE_STRING: i32 = 23;
const TYPE_LIST: i32 = 24;
const TYPE_DECIMAL32: i32 = 25;
const TYPE_DECIMAL64: i32 = 26;
const TYPE_DECIMAL128: i32 = 27;
const TYPE_STRUCT: i32 = 28;

/// Converts an Arrow type to a cuDF data type, including decimal scale metadata.
pub(crate) fn arrow_type_to_cudf_data_type(
    arrow_type: &DataType,
) -> Option<UniquePtr<ffi::DataType>> {
    match arrow_type {
        // Arrow and cuDF fixed-point scales use opposite signs.
        DataType::Decimal32(_, scale) => Some(ffi::new_data_type_with_scale(
            TYPE_DECIMAL32,
            -i32::from(*scale),
        )),
        DataType::Decimal64(_, scale) => Some(ffi::new_data_type_with_scale(
            TYPE_DECIMAL64,
            -i32::from(*scale),
        )),
        DataType::Decimal128(_, scale) => Some(ffi::new_data_type_with_scale(
            TYPE_DECIMAL128,
            -i32::from(*scale),
        )),
        _ => arrow_type_to_cudf_type_id(arrow_type).map(ffi::new_data_type),
    }
}

/// Returns true if the Arrow type has a direct cuDF type mapping.
pub(crate) fn is_arrow_type_supported_by_cudf(dtype: &DataType) -> bool {
    arrow_type_to_cudf_type_id(dtype).is_some()
}

fn arrow_type_to_cudf_type_id(dtype: &DataType) -> Option<i32> {
    match dtype {
        DataType::Int8 => Some(TYPE_INT8),
        DataType::Int16 => Some(TYPE_INT16),
        DataType::Int32 => Some(TYPE_INT32),
        DataType::Int64 => Some(TYPE_INT64),
        DataType::UInt8 => Some(TYPE_UINT8),
        DataType::UInt16 => Some(TYPE_UINT16),
        DataType::UInt32 => Some(TYPE_UINT32),
        DataType::UInt64 => Some(TYPE_UINT64),
        DataType::Float32 => Some(TYPE_FLOAT32),
        DataType::Float64 => Some(TYPE_FLOAT64),
        DataType::Boolean => Some(TYPE_BOOL8),
        DataType::Utf8 => Some(TYPE_STRING),
        DataType::LargeUtf8 => Some(TYPE_STRING),
        DataType::Date32 => Some(TYPE_TIMESTAMP_DAYS),
        DataType::Timestamp(TimeUnit::Second, _) => Some(TYPE_TIMESTAMP_SECONDS),
        DataType::Timestamp(TimeUnit::Millisecond, _) => Some(TYPE_TIMESTAMP_MILLISECONDS),
        DataType::Timestamp(TimeUnit::Microsecond, _) => Some(TYPE_TIMESTAMP_MICROSECONDS),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => Some(TYPE_TIMESTAMP_NANOSECONDS),
        DataType::Duration(TimeUnit::Second) => Some(TYPE_DURATION_SECONDS),
        DataType::Duration(TimeUnit::Millisecond) => Some(TYPE_DURATION_MILLISECONDS),
        DataType::Duration(TimeUnit::Microsecond) => Some(TYPE_DURATION_MICROSECONDS),
        DataType::Duration(TimeUnit::Nanosecond) => Some(TYPE_DURATION_NANOSECONDS),
        DataType::List(_) => Some(TYPE_LIST),
        DataType::LargeList(_) => Some(TYPE_LIST),
        DataType::Struct(_) => Some(TYPE_STRUCT),
        DataType::Dictionary(_, _) => Some(TYPE_DICTIONARY32),
        DataType::Decimal32(_, _) => Some(TYPE_DECIMAL32),
        DataType::Decimal64(_, _) => Some(TYPE_DECIMAL64),
        DataType::Decimal128(_, _) => Some(TYPE_DECIMAL128),
        _ => None,
    }
}

/// Converts a cuDF data type to Arrow when the Arrow type can be reconstructed locally.
pub(crate) fn cudf_type_to_arrow(typ: &UniquePtr<ffi::DataType>) -> Option<DataType> {
    match typ.id() {
        TYPE_EMPTY => None,
        TYPE_INT8 => Some(DataType::Int8),
        TYPE_INT16 => Some(DataType::Int16),
        TYPE_INT32 => Some(DataType::Int32),
        TYPE_INT64 => Some(DataType::Int64),
        TYPE_UINT8 => Some(DataType::UInt8),
        TYPE_UINT16 => Some(DataType::UInt16),
        TYPE_UINT32 => Some(DataType::UInt32),
        TYPE_UINT64 => Some(DataType::UInt64),
        TYPE_FLOAT32 => Some(DataType::Float32),
        TYPE_FLOAT64 => Some(DataType::Float64),
        TYPE_BOOL8 => Some(DataType::Boolean),
        TYPE_TIMESTAMP_DAYS => Some(DataType::Date32),
        TYPE_TIMESTAMP_SECONDS => Some(DataType::Timestamp(TimeUnit::Second, None)),
        TYPE_TIMESTAMP_MILLISECONDS => Some(DataType::Timestamp(TimeUnit::Millisecond, None)),
        TYPE_TIMESTAMP_MICROSECONDS => Some(DataType::Timestamp(TimeUnit::Microsecond, None)),
        TYPE_TIMESTAMP_NANOSECONDS => Some(DataType::Timestamp(TimeUnit::Nanosecond, None)),
        TYPE_DURATION_DAYS => None,
        TYPE_DURATION_SECONDS => Some(DataType::Duration(TimeUnit::Second)),
        TYPE_DURATION_MILLISECONDS => Some(DataType::Duration(TimeUnit::Millisecond)),
        TYPE_DURATION_MICROSECONDS => Some(DataType::Duration(TimeUnit::Microsecond)),
        TYPE_DURATION_NANOSECONDS => Some(DataType::Duration(TimeUnit::Nanosecond)),
        TYPE_DICTIONARY32 => None,
        TYPE_STRING => Some(DataType::Utf8),
        TYPE_LIST => None,
        // Arrow and cuDF fixed-point scales use opposite signs.
        TYPE_DECIMAL32 => Some(DataType::Decimal32(9, (-typ.scale()) as i8)),
        TYPE_DECIMAL64 => Some(DataType::Decimal64(18, (-typ.scale()) as i8)),
        TYPE_DECIMAL128 => Some(DataType::Decimal128(38, (-typ.scale()) as i8)),
        TYPE_STRUCT => None,
        _ => None,
    }
}

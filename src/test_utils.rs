use crate::{CuDFError, CuDFTable};
use arrow::array::{make_array, Array};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

/// Create a single-column CuDFTable from an Arrow array
///
/// The column will be named "col"
pub fn make_single_column_table(array: &dyn Array) -> Result<CuDFTable, CuDFError> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "col",
        array.data_type().clone(),
        true,
    )]));
    let array_data = array.to_data();
    let batch = RecordBatch::try_new(schema, vec![make_array(array_data)])?;
    CuDFTable::from_arrow_host(batch)
}

/// Create a CuDFTable from multiple Arrow arrays with custom column names
pub fn make_table(arrays: Vec<&dyn Array>, names: Vec<&str>) -> Result<CuDFTable, CuDFError> {
    assert_eq!(
        arrays.len(),
        names.len(),
        "Number of arrays must match number of names"
    );

    let fields: Vec<Field> = arrays
        .iter()
        .zip(names.iter())
        .map(|(array, name)| Field::new(*name, array.data_type().clone(), true))
        .collect();

    let schema = Arc::new(Schema::new(fields));
    let columns: Vec<Arc<dyn Array>> = arrays
        .iter()
        .map(|array| make_array(array.to_data()))
        .collect();

    let batch = RecordBatch::try_new(schema, columns)?;
    CuDFTable::from_arrow_host(batch)
}

/// Create a keys table for groupby operations from an array
///
/// Helper for creating the keys table needed for CuDFGroupBy
pub fn make_keys_table(keys: &dyn Array) -> Result<CuDFTable, CuDFError> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "key",
        keys.data_type().clone(),
        false,
    )]));
    let keys_data = keys.to_data();
    let batch = RecordBatch::try_new(schema, vec![make_array(keys_data)])?;
    CuDFTable::from_arrow_host(batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;

    #[test]
    fn test_make_single_column_table() {
        let array = Int32Array::from(vec![1, 2, 3]);
        let table = make_single_column_table(&array).expect("Failed to create table");
        assert_eq!(table.num_rows(), 3);
        assert_eq!(table.num_columns(), 1);
    }

    #[test]
    fn test_make_table() {
        let array1 = Int32Array::from(vec![1, 2, 3]);
        let array2 = Int32Array::from(vec![10, 20, 30]);
        let table =
            make_table(vec![&array1, &array2], vec!["a", "b"]).expect("Failed to create table");
        assert_eq!(table.num_rows(), 3);
        assert_eq!(table.num_columns(), 2);
    }

    #[test]
    fn test_make_keys_table() {
        let keys = Int32Array::from(vec![1, 1, 2, 2]);
        let table = make_keys_table(&keys).expect("Failed to create keys table");
        assert_eq!(table.num_rows(), 4);
        assert_eq!(table.num_columns(), 1);
    }
}

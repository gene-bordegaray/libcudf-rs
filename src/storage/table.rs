use crate::device_resource::resource_ref;
use crate::stream::stream_ref;
use crate::{CuDFColumn, CuDFError};
use arrow_schema::ArrowError;
use cxx::UniquePtr;
use libcudf_sys::ffi;
use std::sync::Arc;

use super::table_view::{CuDFTableView, TableOwner};

pub(crate) struct TableStorage {
    view: Arc<UniquePtr<ffi::TableView>>,
    inner: UniquePtr<ffi::Table>,
    num_rows: usize,
    column_allocation_sizes: Vec<usize>,
    device_memory_size: usize,
}

impl TableStorage {
    pub(super) fn device_memory_size(&self) -> usize {
        self.device_memory_size
    }
}

/// A GPU-accelerated table (similar to a DataFrame).
///
/// This is a safe wrapper around cuDF's owning table type.
pub struct CuDFTable {
    storage: Arc<TableStorage>,
}

impl CuDFTable {
    /// Create a `CuDFTable` from a raw FFI table.
    pub(crate) fn try_from_inner(inner: UniquePtr<ffi::Table>) -> Result<Self, CuDFError> {
        if inner.is_null() {
            return Err(CuDFError::NullHandle("table"));
        }
        let num_rows = crate::errors::cudf_size_to_usize(inner.num_rows()?, "table row count")?;
        let column_allocation_sizes = table_column_allocation_sizes(&inner)?;
        let device_memory_size = column_allocation_sizes.iter().sum();
        // The view field is dropped before the owning table field.
        let view = unsafe { inner.view() }?;
        Ok(Self {
            storage: Arc::new(TableStorage {
                view: Arc::new(view),
                inner,
                num_rows,
                column_allocation_sizes,
                device_memory_size,
            }),
        })
    }

    /// Create an empty table.
    ///
    /// # Examples
    ///
    /// ```
    /// use libcudf_rs::CuDFTable;
    ///
    /// let table = CuDFTable::try_empty()?;
    /// assert_eq!(table.num_rows(), 0);
    /// assert_eq!(table.num_columns(), 0);
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn try_empty() -> Result<Self, CuDFError> {
        Self::try_from_inner(ffi::create_empty_table()?)
    }

    pub(crate) fn try_from_columns(columns: Vec<CuDFColumn>) -> Result<Self, CuDFError> {
        let columns = columns
            .into_iter()
            .map(CuDFColumn::try_into_inner)
            .collect::<Result<Vec<_>, _>>()?;
        let ptrs: Vec<_> = columns.iter().map(UniquePtr::as_mut_ptr).collect();
        let inner = unsafe { ffi::create_table_from_columns_move(&ptrs) }?;
        Self::try_from_inner(inner)
    }

    pub(crate) fn view_inner(&self) -> &UniquePtr<ffi::TableView> {
        &self.storage.view
    }

    /// Get the number of rows in the table.
    pub fn num_rows(&self) -> usize {
        self.storage.num_rows
    }

    /// Get the number of columns in the table.
    pub fn num_columns(&self) -> usize {
        self.storage.column_allocation_sizes.len()
    }

    /// Return whether the table has no rows.
    pub fn is_empty(&self) -> bool {
        self.num_rows() == 0
    }

    /// Return the bytes allocated for this table in device memory.
    pub fn device_memory_size(&self) -> usize {
        self.storage.device_memory_size()
    }

    /// Return a non-owning view that retains this table's storage.
    pub fn view(self: Arc<Self>) -> CuDFTableView {
        self.view_from_storage()
    }

    fn view_from_storage(&self) -> CuDFTableView {
        let storage = Arc::clone(&self.storage);
        CuDFTableView::from_shared_view(
            Arc::clone(&storage.view),
            TableOwner::Table(storage),
            self.storage.num_rows,
            self.storage.column_allocation_sizes.clone(),
        )
    }

    /// Consume this table and return a non-owning view that retains its storage.
    pub fn into_view(self) -> CuDFTableView {
        self.view_from_storage()
    }

    /// Consume this table and transfer ownership of its columns.
    pub fn into_columns(self) -> Result<Vec<CuDFColumn>, CuDFError> {
        let mut storage = Arc::try_unwrap(self.storage).map_err(|_| {
            ArrowError::ComputeError(
                "cannot consume a cuDF table while a view still owns it".into(),
            )
        })?;
        let mut columns = storage.inner.pin_mut().release()?;
        let mut result = Vec::with_capacity(columns.len());
        for index in 0..columns.len() {
            result.push(CuDFColumn::try_from_inner(
                columns.pin_mut().release(index),
            )?);
        }
        Ok(result)
    }

    /// Concatenate table views into one owning table.
    pub fn concat(views: Vec<CuDFTableView>) -> Result<Self, CuDFError> {
        let inner_views = views
            .iter()
            .map(CuDFTableView::clone_inner)
            .collect::<Result<Vec<_>, _>>()?;
        let stream = ffi::get_default_stream();
        let resource = ffi::get_current_device_resource_ref();
        Self::try_from_inner(ffi::concatenate_tables(
            &inner_views,
            stream_ref(&stream)?,
            resource_ref(&resource)?,
        )?)
    }
}

fn table_column_allocation_sizes(table: &UniquePtr<ffi::Table>) -> Result<Vec<usize>, CuDFError> {
    let num_columns =
        crate::errors::cudf_size_to_usize(table.num_columns()?, "table column count")?;
    (0..num_columns)
        .map(|index| {
            let index = i32::try_from(index).map_err(|_| {
                ArrowError::ComputeError("cuDF table column index overflowed i32".into())
            })?;
            let column = unsafe { table.get_column(index) }?;
            Ok(column.alloc_size()?)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_table_has_no_storage() -> Result<(), Box<dyn std::error::Error>> {
        let table = CuDFTable::try_empty()?;
        assert_eq!(table.num_rows(), 0);
        assert_eq!(table.num_columns(), 0);
        assert_eq!(table.device_memory_size(), 0);
        assert!(table.is_empty());
        Ok(())
    }
}

use crate::storage::column_view::ColumnOwner;
use crate::{CuDFColumnView, CuDFError};
use cxx::UniquePtr;
use libcudf_sys::ffi;
use std::sync::Arc;

pub(crate) struct JoinIndexVector {
    inner: UniquePtr<ffi::DeviceIndexVector>,
}

impl JoinIndexVector {
    pub(super) fn try_from_inner(
        inner: UniquePtr<ffi::DeviceIndexVector>,
    ) -> Result<Self, CuDFError> {
        if inner.is_null() {
            return Err(CuDFError::NullHandle("device index vector"));
        }
        Ok(Self { inner })
    }

    pub(super) fn len(&self) -> usize {
        self.inner.size()
    }

    pub(super) fn as_sys(&self) -> Result<&ffi::DeviceIndexVector, CuDFError> {
        self.inner
            .as_ref()
            .ok_or(CuDFError::NullHandle("device index vector"))
    }

    pub(super) fn view(self: Arc<Self>) -> Result<CuDFColumnView, CuDFError> {
        let view = unsafe { self.inner.view() };
        CuDFColumnView::try_from_inner(view, ColumnOwner::JoinIndices(self))
    }
}

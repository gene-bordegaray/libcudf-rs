use std::sync::Arc;

use cxx::UniquePtr;
use libcudf_sys::ffi;

use crate::keep_alive::CuDFKeepAlive;
use crate::{CuDFError, Result};

const CUDA_EVENT_FLAG_DEFAULT: u32 = 0;

/// A value that may need to wait for producer-stream work.
pub(crate) trait CuDFStreamReady {
    /// Waits on `stream` until this value's recorded producer work is visible.
    fn wait_ready_on_stream(&self, stream: &ffi::CudaStreamView) -> Result<()>;
}

/// CUDA event used to transfer readiness from one stream to another.
///
/// Consumers wait on this event instead of synchronizing the device or host.
struct CuDFEvent {
    inner: UniquePtr<ffi::CudaEvent>,
}

impl CuDFEvent {
    /// Records a CUDA event on `stream`.
    fn try_recorded_on(stream: &ffi::CudaStreamView) -> Result<Self> {
        let stream_ref = ffi::cuda_stream_ref_from_view(stream);
        Ok(Self {
            inner: Self::stream_ref(&stream_ref)?.record_event(CUDA_EVENT_FLAG_DEFAULT)?,
        })
    }

    /// Makes `stream` wait until this event has completed.
    fn wait_on(&self, stream: &ffi::CudaStreamView) -> Result<()> {
        let stream_ref = ffi::cuda_stream_ref_from_view(stream);
        Self::stream_ref(&stream_ref)?.wait_event(self.inner()?)?;
        Ok(())
    }

    fn inner(&self) -> Result<&ffi::CudaEvent> {
        self.inner
            .as_ref()
            .ok_or(CuDFError::NullHandle("CUDA event"))
    }

    fn stream_ref(stream: &UniquePtr<ffi::CudaStreamRef>) -> Result<&ffi::CudaStreamRef> {
        stream
            .as_ref()
            .ok_or(CuDFError::NullHandle("CUDA stream ref"))
    }
}

/// Producer event plus values retained with it.
struct CuDFStreamDependencyInner {
    event: CuDFEvent,
    _keepalives: Vec<CuDFKeepAlive>,
}

/// Readiness dependency recorded after launching cuDF work on a stream.
#[derive(Clone)]
pub(crate) struct CuDFStreamDependency {
    inner: Arc<CuDFStreamDependencyInner>,
}

impl CuDFStreamDependency {
    /// Records a producer event and its keepalives.
    pub(crate) fn record_on_stream(
        stream: &ffi::CudaStreamView,
        keepalives: Vec<CuDFKeepAlive>,
    ) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(CuDFStreamDependencyInner {
                event: CuDFEvent::try_recorded_on(stream)?,
                _keepalives: keepalives,
            }),
        })
    }

    /// Makes `stream` wait for this dependency's producer event.
    pub(crate) fn wait_on_stream(&self, stream: &ffi::CudaStreamView) -> Result<()> {
        self.inner.event.wait_on(stream)
    }
}

/// Readiness metadata for table-like values.
///
/// `Whole` is used for outputs produced by one table operation. `Columns` is
/// used when a table view is assembled from independent column views and should
/// preserve each column's original dependency.
#[derive(Clone, Default)]
pub(crate) enum CuDFTableReadiness {
    /// No stream dependency is associated with the table.
    #[default]
    None,
    /// One dependency covers every column in the table.
    Whole(CuDFStreamDependency),
    /// Dependencies are tracked per column.
    Columns(Vec<Option<CuDFStreamDependency>>),
}

impl CuDFTableReadiness {
    /// Creates readiness for a table produced by one operation.
    pub(crate) fn whole(dependency: CuDFStreamDependency) -> Self {
        Self::Whole(dependency)
    }

    /// Creates per-column readiness metadata.
    pub(crate) fn columns(
        mut dependencies: Vec<Option<CuDFStreamDependency>>,
        num_columns: usize,
    ) -> Self {
        dependencies.truncate(num_columns);
        dependencies.resize_with(num_columns, || None);
        if dependencies.iter().any(Option::is_some) {
            Self::Columns(dependencies)
        } else {
            Self::None
        }
    }

    /// Returns the readiness dependency for a specific column, if any.
    pub(crate) fn column(&self, index: usize) -> Option<CuDFStreamDependency> {
        match self {
            Self::None => None,
            Self::Whole(dependency) => Some(dependency.clone()),
            Self::Columns(dependencies) => dependencies.get(index).cloned().flatten(),
        }
    }

    /// Waits for all dependencies needed to read this table on `stream`.
    pub(crate) fn wait_on_stream(&self, stream: &ffi::CudaStreamView) -> Result<()> {
        match self {
            Self::None => Ok(()),
            Self::Whole(dependency) => dependency.wait_on_stream(stream),
            Self::Columns(dependencies) => wait_unique_on_stream(dependencies.iter(), stream),
        }
    }
}

fn wait_unique_on_stream<'a>(
    dependencies: impl IntoIterator<Item = &'a Option<CuDFStreamDependency>>,
    stream: &ffi::CudaStreamView,
) -> Result<()> {
    let mut waited = Vec::new();
    for dependency in dependencies.into_iter().flatten() {
        if waited
            .iter()
            .any(|seen: &&CuDFStreamDependency| Arc::ptr_eq(&dependency.inner, &seen.inner))
        {
            continue;
        }
        dependency.wait_on_stream(stream)?;
        waited.push(dependency);
    }
    Ok(())
}

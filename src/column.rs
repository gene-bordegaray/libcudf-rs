use crate::config::ensure_pools_configured;
use crate::deferred_operation::deferred;
use crate::execution_policy;
use crate::stream_readiness::{CuDFStreamDependency, CuDFStreamReady};
use crate::{CuDFColumnView, CuDFError, CuDFOperation, CuDFScalar, CuDFViewStorage};
use cxx::UniquePtr;
use libcudf_sys::ffi;
use std::sync::Arc;

/// An owning cuDF column stored in GPU memory.
///
/// `CuDFColumn` owns the underlying cuDF column. Use [`CuDFColumnView`] for
/// non-owning operations and Arrow interoperability.
pub struct CuDFColumn {
    inner: UniquePtr<ffi::Column>,
    stream_readiness: Option<CuDFStreamDependency>,
}

impl CuDFColumn {
    pub(crate) fn from_inner(inner: UniquePtr<ffi::Column>) -> Self {
        Self {
            inner,
            stream_readiness: None,
        }
    }

    pub(crate) fn inner_mut(&mut self) -> &mut UniquePtr<ffi::Column> {
        &mut self.inner
    }

    pub(crate) fn with_stream_readiness(mut self, dependency: CuDFStreamDependency) -> Self {
        self.stream_readiness = Some(dependency);
        self
    }

    pub(crate) fn stream_readiness(&self) -> Option<&CuDFStreamDependency> {
        self.stream_readiness.as_ref()
    }

    /// Returns the number of rows in this column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow::array::Int32Array;
    /// use libcudf_rs::{CuDFColumn, CuDFExecutionContext};
    ///
    /// let input = Int32Array::from(vec![1, 2, 3]);
    /// let column = CuDFExecutionContext::try_new_non_blocking()?
    ///     .execute(CuDFColumn::from_arrow_host(&input))?;
    /// assert_eq!(column.len(), 3);
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn len(&self) -> usize {
        self.inner.size()
    }

    /// Returns true when this column has no rows.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow::array::Int32Array;
    /// use libcudf_rs::{CuDFColumn, CuDFExecutionContext};
    ///
    /// let input = Int32Array::from(Vec::<i32>::new());
    /// let column = CuDFExecutionContext::try_new_non_blocking()?
    ///     .execute(CuDFColumn::from_arrow_host(&input))?;
    /// assert!(column.is_empty());
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a deferred operation that builds a column by repeating a scalar value.
    ///
    /// The scalar may have been produced on a different stream. Executing this
    /// operation waits on that scalar's readiness before launching the repeat
    /// operation on the target context.
    ///
    /// # Errors
    ///
    /// Execution returns an error if the column cannot be allocated on the GPU.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow::array::{Int32Array, Scalar};
    /// use libcudf_rs::{CuDFColumn, CuDFExecutionContext, CuDFScalar};
    ///
    /// let array = Int32Array::from(vec![7]);
    /// let scalar = Scalar::new(&array);
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let scalar = ctx.execute(CuDFScalar::from_arrow_host(scalar))?;
    /// let column = ctx.execute(CuDFColumn::from_scalar(&scalar, 3))?;
    ///
    /// assert_eq!(column.len(), 3);
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn from_scalar(scalar: &CuDFScalar, len: usize) -> impl CuDFOperation<Output = Self> + '_ {
        deferred(move |ctx| {
            ensure_pools_configured();
            let mut launch = execution_policy::launch(ctx)?;
            launch.wait_scalar(scalar)?;
            let column = Self::from_inner(ffi::make_column_from_scalar(
                scalar.inner(),
                len,
                launch.stream()?,
                launch.resource(),
            )?);
            launch.ready_column(column)
        })
    }

    /// Returns a non-owning view of this column.
    ///
    /// The returned view keeps this column alive and preserves any stream
    /// readiness recorded when the column was produced.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow::array::{Array, Int32Array};
    /// use libcudf_rs::{CuDFColumn, CuDFExecutionContext};
    /// use std::sync::Arc;
    ///
    /// let input = Int32Array::from(vec![1, 2, 3]);
    /// let column = Arc::new(
    ///     CuDFExecutionContext::try_new_non_blocking()?
    ///         .execute(CuDFColumn::from_arrow_host(&input))?,
    /// );
    /// let view = column.view();
    /// assert_eq!(view.len(), 3);
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn view(self: Arc<Self>) -> CuDFColumnView {
        let view = self.inner.view();
        let dependency = self.stream_readiness.clone();
        let storage: CuDFViewStorage = self;
        CuDFColumnView::from_view(view, Some(storage), dependency)
    }

    /// Converts this column into a non-owning view that keeps the column alive.
    ///
    /// This is a convenience wrapper around [`CuDFColumn::view`] for callers
    /// that do not already store the column in an [`Arc`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow::array::{Array, Int32Array};
    /// use libcudf_rs::{CuDFColumn, CuDFExecutionContext};
    ///
    /// let input = Int32Array::from(vec![1, 2, 3]);
    /// let column = CuDFExecutionContext::try_new_non_blocking()?
    ///     .execute(CuDFColumn::from_arrow_host(&input))?;
    /// let view = column.into_view();
    ///
    /// assert_eq!(view.len(), 3);
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn into_view(self) -> CuDFColumnView {
        Arc::new(self).view()
    }

    /// Create a deferred operation that concatenates column views into one column.
    ///
    /// Execution waits for each input column view before launching cuDF concat on
    /// the target context.
    ///
    /// # Errors
    ///
    /// Execution returns an error if the inputs have incompatible types or the
    /// output column cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use arrow::array::Int32Array;
    /// use libcudf_rs::{CuDFColumn, CuDFExecutionContext};
    ///
    /// let ctx = CuDFExecutionContext::try_new_non_blocking()?;
    /// let first = ctx
    ///     .execute(CuDFColumn::from_arrow_host(&Int32Array::from(vec![1, 2])))?
    ///     .into_view();
    /// let second = ctx
    ///     .execute(CuDFColumn::from_arrow_host(&Int32Array::from(vec![3, 4])))?
    ///     .into_view();
    ///
    /// let concatenated = ctx.execute(CuDFColumn::concat(vec![first, second]))?;
    /// assert_eq!(concatenated.len(), 4);
    /// # Ok::<(), libcudf_rs::CuDFError>(())
    /// ```
    pub fn concat(views: Vec<CuDFColumnView>) -> impl CuDFOperation<Output = Self> {
        deferred(move |ctx| {
            let mut launch = execution_policy::launch(ctx)?;
            let mut inner_views = Vec::with_capacity(views.len());
            for view in views {
                launch.wait_column(&view)?;
                inner_views.push(view.into_inner());
            }
            let inner =
                ffi::concat_column_views(&inner_views, launch.stream()?, launch.resource())?;
            launch.ready_column(Self::from_inner(inner))
        })
    }
}

impl CuDFStreamReady for CuDFColumn {
    fn wait_ready_on_stream(&self, stream: &ffi::CudaStreamView) -> Result<(), CuDFError> {
        if let Some(dependency) = &self.stream_readiness {
            dependency.wait_on_stream(stream)?;
        }
        Ok(())
    }
}

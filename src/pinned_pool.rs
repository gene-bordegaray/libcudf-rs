/// Configuration for cuDF's pinned host-memory pool.
///
/// A fixed-size pool of page-locked (pinned) host RAM reserved once at startup.
/// cuDF draws from this pool for its internal host allocations, intermediate buffers,
/// and output staging on the download path (GPU -> host), instead of calling
/// `cudaMallocHost` per operation. Pinned memory can be DMA'd directly by the GPU
/// without the CUDA runtime needing to stage through a temporary buffer first.
///
/// The pool primarily benefits the download path where cuDF owns the destination
/// allocation. On the upload path, the Arrow source buffer is the user's pageable
/// memory and is unaffected by this pool.
///
/// ## Without pool
///
/// For each column in a batch, on a single CUDA stream:
///
/// 1. cuDF calls `cudaMemcpyAsync(gpu_dst, arrow_ptr, size, Default, stream)`
/// 2. CUDA detects the source is pageable and cannot DMA it directly
/// 3. CUDA runtime stages the data internally through its own pinned staging buffer:
///    - CPU copies data to the staging buffer (CPU blocks)
///    - DMA enqueued for that chunk on stream
///    - CPU waits for DMA before reusing the staging buffer
///    - Repeats until the full column is transferred
/// 4. Steps 1–3 repeat for every column, serially
///
/// ```text
/// CPU  [--stage--][----wait----][--stage--][----wait----] col 0 done [--stage--]...
/// PCIe            [DMA transfer]           [DMA transfer]
/// ```
///
/// ## With pool
///
/// cuDF's internal host allocations come from the pre-locked slab. With a pinned
/// destination, CUDA issues a single DMA per column with no staging:
///
/// 1. cuDF allocates destination buffer from the pinned slab (pointer bump, no OS call)
/// 2. `cudaMemcpyAsync` sees a pinned destination and enqueues one DMA
/// 3. CPU returns from the call instantly (no staging, no blocking)
///
/// ```text
/// CPU  [alloc][enqueue] [alloc][enqueue] [alloc][enqueue]
/// PCIe [----- DMA col0 -----][----- DMA col1 -----][----- DMA col2 -----]
/// ```
///
/// # When to use
///
/// Enable for workloads where batches exceed ~1 MB per column as the staging
/// overhead is paid per column per batch.
///
/// Disable for workloads with small batches where the pool reservation cost
/// exceeds the savings.
///
/// # Usage
///
/// Call [`apply`] once before running any GPU operations. See also
/// [`configure_default_pools`](crate::configure_default_pools) for a
/// single-call shorthand using the default values.
///
/// ```rust,no_run
/// use libcudf_rs::PinnedPoolConfig;
///
/// PinnedPoolConfig {
///     pool_size: 512 * 1024 * 1024,
///     threshold:   1 * 1024 * 1024,
/// }.apply();
/// ```
pub struct PinnedPoolConfig {
    /// Bytes of page-locked host RAM to reserve upfront.
    ///
    /// The pool is fixed-size, both the initial and maximum capacity are set to this
    /// value. If exhausted, cuDF falls back to a non-pooled pinned allocation rather
    /// than growing or erroring.
    ///
    /// Paid once at [`apply`] time. All subsequent cuDF internal host allocations
    /// draw from this slab with no OS calls and no per-allocation page locking.
    ///
    /// Pinned memory is page-locked and cannot be swapped, so it reduces the
    /// pageable RAM available to the rest of the system. Size conservatively on
    /// shared or memory-constrained hosts.
    pub pool_size: usize,

    /// Allocations up to this size (bytes) use the pinned pool. Larger ones stay pageable.
    ///
    /// cuDF's built-in default is 0, meaning all host allocations are pageable unless
    /// this is set. 1 MB is a good starting point for analytics workloads; lower it
    /// if your columns are typically smaller than 1 MB per batch.
    pub threshold: usize,
}

impl Default for PinnedPoolConfig {
    /// Conservative defaults suitable for GPU analytics workloads.
    ///
    /// - `pool_size`: 512 MB
    /// - `threshold`: 1 MB
    ///
    /// Override `pool_size` upward if your batches regularly exceed 256 MB per column,
    /// or downward on memory-constrained hosts.
    fn default() -> Self {
        Self {
            pool_size: 512 * 1024 * 1024, // 512 MB
            threshold: 1 * 1024 * 1024,   //   1 MB - smaller allocs stay pageable
        }
    }
}

impl PinnedPoolConfig {
    /// Apply this configuration globally for the current process.
    ///
    /// Must be called before any cuDF operations run.
    ///
    /// Returns `true` if the pool was configured, `false` if already set by
    /// a previous call (the threshold is still updated).
    pub fn apply(&self) -> bool {
        let configured = libcudf_sys::ffi::config_pinned_memory_resource(self.pool_size);
        libcudf_sys::ffi::set_host_pinned_threshold(self.threshold);
        configured
    }
}

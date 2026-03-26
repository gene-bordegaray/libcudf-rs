/// Configuration for the RMM GPU device-memory pool.
///
/// A pre-reserved slab of GPU VRAM allocated once at startup. Every cuDF column
/// buffer is backed by an `rmm::device_buffer` which draws from this slab. Freed
/// buffers are returned immediately and reused by the next batch with no driver
/// involvement.
///
/// ## Without pool
///
/// For each column in a batch, before any data transfer begins:
///
/// 1. cuDF calls `cudaMalloc(size)` to back the `rmm::device_buffer`
/// 2. GPU driver acquires a global mutex
/// 3. Driver walks the virtual address tree to find a free region
/// 4. Driver updates page tables and returns a pointer
/// 5. Steps 1–4 repeat for every column, serially
/// 6. On batch teardown, `cudaFree` is called for every column (another driver round-trip each)
///
/// ```text
/// col 0:  [cudaMalloc] [DMA col0 ----------]
/// col 1:               [cudaMalloc] [DMA col1 ----------]
/// col 2:                            [cudaMalloc] [DMA col2 ----------]
/// ```
///
/// ## With pool
///
/// The slab is reserved once at [`apply`] time. For each column in a batch:
///
/// 1. cuDF calls `pool.alloc(size)` - pointer bump into the slab (no driver call)
/// 2. DMA transfer begins immediately
/// 3. On teardown, `pool.free()` returns the slot to the slab (no driver call)
/// 4. The next batch reuses the same slots
///
/// ```text
/// startup:  [cudaMalloc once -----------------------------------------]
///
/// batch N:  [alloc x cols] [DMA col0 --][DMA col1 --][DMA col2 --]
/// batch N+1:[alloc x cols] [DMA col0 --][DMA col1 --][DMA col2 --]
/// ```
///
/// # When to use
///
/// Enable for any workload that processes many batches. The `cudaMalloc` cost is
/// paid per column per batch on both allocation and free.
///
/// For one-shot queries the benefit is marginal. On shared GPU servers
/// always set `max_size` to leave headroom. Without a cap the pool can grow to
/// exhaust all available VRAM.
///
/// # Usage
///
/// Call [`apply`] once before running any GPU operations. See also
/// [`configure_default_pools`](crate::configure_default_pools) for a
/// single-call shorthand using the default values.
///
/// ```rust,no_run
/// use libcudf_rs::DevicePoolConfig;
///
/// DevicePoolConfig {
///     initial_size: 4 * 1024 * 1024 * 1024,
///     max_size:     8 * 1024 * 1024 * 1024,
/// }.apply();
/// ```
pub struct DevicePoolConfig {
    /// Bytes of GPU VRAM to reserve upfront via a single `cudaMalloc`.
    ///
    /// The pool pre-allocates this slab when [`apply`] is called. If the slab fills
    /// up, the pool grows by calling `cudaMalloc` again until [`max_size`] is reached.
    ///
    /// Set to at least your pipeline's peak working set (the sum of all live column
    /// buffers across concurrent batches). Sizing below this causes growth pauses during
    /// warm-up.
    pub initial_size: usize,

    /// Hard cap in bytes on total pool VRAM.
    ///
    /// The pool will not grow beyond this. If an allocation would exceed it,
    /// cuDF throws `std::bad_alloc`. Set this to leave headroom for other GPU
    /// workloads sharing the same device.
    ///
    /// Must be at least `initial_size`.
    pub max_size: usize,
}

impl Default for DevicePoolConfig {
    /// Conservative defaults suitable for GPU analytics workloads.
    ///
    /// - `initial_size`: 512 MB
    /// - `max_size`: 4 GB
    ///
    /// Override `initial_size` upward if your pipeline's peak working set exceeds
    /// 512 MB to avoid growth pauses during warm-up.
    fn default() -> Self {
        Self {
            initial_size: 512 * 1024 * 1024,  // 512 MB
            max_size: 4 * 1024 * 1024 * 1024, //   4 GB
        }
    }
}

impl DevicePoolConfig {
    /// Apply this configuration globally for the current process.
    ///
    /// Must be called before any cuDF operations run.
    ///
    /// Returns `true` if the pool was configured, `false` if already set by
    /// a previous call.
    pub fn apply(&self) -> bool {
        libcudf_sys::ffi::config_device_memory_pool(self.initial_size, self.max_size)
    }
}

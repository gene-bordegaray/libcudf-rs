use std::cell::Cell;

thread_local! {
    static STREAM_ACCESSES: Cell<usize> = const { Cell::new(0) };
    static RESOURCE_ACCESSES: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn record_stream_access() {
    STREAM_ACCESSES.set(STREAM_ACCESSES.get() + 1);
}

pub(crate) fn record_resource_access() {
    RESOURCE_ACCESSES.set(RESOURCE_ACCESSES.get() + 1);
}

pub(crate) fn reset() {
    STREAM_ACCESSES.set(0);
    RESOURCE_ACCESSES.set(0);
}

pub(crate) fn accesses() -> (usize, usize) {
    (STREAM_ACCESSES.get(), RESOURCE_ACCESSES.get())
}

//! Simple deterministic RNG for dataset shuffling.
//! Does not need to match C rand() — conformance tests verify
//! model correctness via cross-tagging, not byte-identical models
//! for algorithms that shuffle (AP, PA, AROW).

use std::cell::Cell;

thread_local! {
    static STATE: Cell<u64> = const { Cell::new(12345) };
}

/// Returns a non-negative pseudo-random integer.
#[inline]
pub fn rand_int() -> usize {
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x >> 1) as usize
    })
}

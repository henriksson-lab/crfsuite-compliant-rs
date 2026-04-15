//! C-compatible RNG wrapper for trainer dataset shuffling.

use std::os::raw::c_int;

unsafe extern "C" {
    fn rand() -> c_int;
}

/// Returns a non-negative pseudo-random integer.
#[inline]
pub fn rand_int() -> usize {
    unsafe { rand() as usize }
}

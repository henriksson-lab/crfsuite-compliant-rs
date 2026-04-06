//! Cross-validation: Rust vecexp vs C SSE2 vecexp.
//! Both must produce bit-identical f64 values.

use crfsuite_core::vecmath::vecexp;

extern "C" {
    fn c_vecexp(values: *mut f64, n: i32);
}

/// Allocate a 16-byte aligned f64 buffer (required by C SSE2 code which uses _mm_load_pd).
fn aligned_vec(values: &[f64]) -> Vec<f64> {
    // Pad to multiple of 4
    let n = (values.len() + 3) & !3;
    let mut v = vec![0.0f64; n];
    v[..values.len()].copy_from_slice(values);
    v
}

fn run_c_vecexp(input: &[f64]) -> Vec<f64> {
    let mut buf = aligned_vec(input);
    // The C SSE code requires 16-byte alignment. Vec<f64> on x86_64 is 8-byte aligned.
    // We use posix_memalign via libc for the C call.
    let n = buf.len();
    let ptr: *mut f64;
    unsafe {
        let mut p: *mut libc::c_void = std::ptr::null_mut();
        let ret = libc::posix_memalign(&mut p, 16, n * std::mem::size_of::<f64>());
        assert_eq!(ret, 0, "posix_memalign failed");
        ptr = p as *mut f64;
        std::ptr::copy_nonoverlapping(buf.as_ptr(), ptr, n);
        c_vecexp(ptr, n as i32);
        std::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), n);
        libc::free(p);
    }
    buf.truncate(input.len());
    buf
}

fn run_rust_vecexp(input: &[f64]) -> Vec<f64> {
    let mut buf = input.to_vec();
    vecexp(&mut buf);
    buf
}

#[test]
fn vecexp_crossval_uniform_range() {
    // Test values from -710 to 710 in steps of 0.1
    let inputs: Vec<f64> = (-7100..=7100).map(|i| i as f64 * 0.1).collect();
    let c_results = run_c_vecexp(&inputs);
    let rs_results = run_rust_vecexp(&inputs);

    let mut mismatches = 0;
    for (i, ((x, c), r)) in inputs.iter().zip(c_results.iter()).zip(rs_results.iter()).enumerate() {
        if c.to_bits() != r.to_bits() {
            mismatches += 1;
            if mismatches <= 10 {
                eprintln!(
                    "MISMATCH at index {} x={}: C={} ({:#018x}) Rust={} ({:#018x})",
                    i, x, c, c.to_bits(), r, r.to_bits()
                );
            }
        }
    }
    assert_eq!(mismatches, 0, "{} bit-level mismatches out of {} values", mismatches, inputs.len());
}

#[test]
fn vecexp_crossval_integers() {
    let inputs: Vec<f64> = (-710..=710).map(|i| i as f64).collect();
    let c_results = run_c_vecexp(&inputs);
    let rs_results = run_rust_vecexp(&inputs);

    for (i, ((x, c), r)) in inputs.iter().zip(c_results.iter()).zip(rs_results.iter()).enumerate() {
        assert_eq!(
            c.to_bits(), r.to_bits(),
            "Mismatch at x={}: C={} ({:#018x}) Rust={} ({:#018x})",
            x, c, c.to_bits(), r, r.to_bits()
        );
    }
}

#[test]
fn vecexp_crossval_small_values() {
    let inputs: Vec<f64> = (-1000..=1000).map(|i| i as f64 * 0.001).collect();
    let c_results = run_c_vecexp(&inputs);
    let rs_results = run_rust_vecexp(&inputs);

    for ((x, c), r) in inputs.iter().zip(c_results.iter()).zip(rs_results.iter()) {
        assert_eq!(
            c.to_bits(), r.to_bits(),
            "Mismatch at x={}: C={} Rust={}",
            x, c, r
        );
    }
}

#[test]
fn vecexp_crossval_edge_cases() {
    let inputs = vec![
        0.0, -0.0, 1.0, -1.0,
        0.5, -0.5,
        1e-15, -1e-15,
        709.0, -708.0,
        709.78, -708.39,
        710.0, -710.0,     // near clamping boundaries
        800.0, -800.0,     // beyond clamping boundaries
    ];
    let c_results = run_c_vecexp(&inputs);
    let rs_results = run_rust_vecexp(&inputs);

    for ((x, c), r) in inputs.iter().zip(c_results.iter()).zip(rs_results.iter()) {
        assert_eq!(
            c.to_bits(), r.to_bits(),
            "Mismatch at x={}: C={} ({:#018x}) Rust={} ({:#018x})",
            x, c, c.to_bits(), r, r.to_bits()
        );
    }
}

#[test]
fn vecexp_crossval_negative_fractional() {
    // Specifically test values where the floor-via-truncate logic matters:
    // negative values where a = x * log2e is between -1 and 0
    let inputs: Vec<f64> = (-100..0).map(|i| i as f64 * 0.005).collect();
    let c_results = run_c_vecexp(&inputs);
    let rs_results = run_rust_vecexp(&inputs);

    for ((x, c), r) in inputs.iter().zip(c_results.iter()).zip(rs_results.iter()) {
        assert_eq!(
            c.to_bits(), r.to_bits(),
            "Mismatch at x={}: C={} ({:#018x}) Rust={} ({:#018x})",
            x, c, c.to_bits(), r, r.to_bits()
        );
    }
}

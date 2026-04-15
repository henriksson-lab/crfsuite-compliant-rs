//! Vector math operations for CRFsuite.
//!
//! `vecexp` reproduces the exact SSE2 polynomial approximation from the C code
//! so that forward-backward scores are bit-identical.

#[inline]
pub fn veczero(x: &mut [f64]) {
    for v in x.iter_mut() {
        *v = 0.0;
    }
}

#[inline]
pub fn vecset(x: &mut [f64], a: f64) {
    for v in x.iter_mut() {
        *v = a;
    }
}

#[inline]
pub fn veccopy(dst: &mut [f64], src: &[f64]) {
    dst[..src.len()].copy_from_slice(src);
}

#[inline]
pub fn vecadd(y: &mut [f64], x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += *xi;
    }
}

#[inline]
pub fn vecaadd(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * *xi;
    }
}

#[inline]
pub fn vecsub(y: &mut [f64], x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi -= *xi;
    }
}

#[inline]
pub fn vecasub(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi -= a * *xi;
    }
}

#[inline]
pub fn vecmul(y: &mut [f64], x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi *= *xi;
    }
}

#[inline]
pub fn vecinv(y: &mut [f64]) {
    for v in y.iter_mut() {
        *v = 1.0 / *v;
    }
}

#[inline]
pub fn vecscale(y: &mut [f64], a: f64) {
    for v in y.iter_mut() {
        *v *= a;
    }
}

#[inline]
pub fn vecdot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

#[inline]
pub fn vecsum(x: &[f64]) -> f64 {
    x.iter().sum()
}

#[inline]
pub fn vecsumlog(x: &[f64]) -> f64 {
    x.iter().map(|v| v.ln()).sum()
}

// ── vecexp: exact reproduction of the C SSE2 polynomial ─────────────────────

const LOG2E: f64 = std::f64::consts::LOG2_E;
const MAXLOG: f64 = 7.097_827_128_933_84e2;
const MINLOG: f64 = -7.083_964_185_322_641e2;
const C1: f64 = 6.93145751953125e-1;
const C2: f64 = 1.428_606_820_309_417_3e-6;

const W11: f64 = 3.5524625185478232665958141148891055719216674475023e-8;
const W10: f64 = 2.5535368519306500343384723775435166753084614063349e-7;
const W9: f64 = 2.77750562801295315877005242757916081614772210463065e-6;
const W8: f64 = 2.47868893393199945541176652007657202642495832996107e-5;
const W7: f64 = 1.98419213985637881240770890090795533564573406893163e-4;
const W6: f64 = 1.3888869684178659239014256260881685824525255547326e-3;
const W5: f64 = 8.3333337052009872221152811550156335074160546333973e-3;
const W4: f64 = 4.1666666621080810610346717440523105184720007971655e-2;
const W3: f64 = 0.166666666669960803484477734308515404418108830469798;
const W2: f64 = 0.499999999999877094481580370323249951329122224389189;
const W1: f64 = 1.0000000000000017952745258419615282194236357388884;
const W0: f64 = 0.99999999999999999566016490920259318691496540598896;

/// Compute exp() for each element, using the exact same polynomial
/// approximation as the C SSE2 code in vecmath.h.
#[inline]
fn fast_exp(x: f64) -> f64 {
    // Clamp
    let x = x.clamp(MINLOG, MAXLOG);

    // a = x * log2(e)
    let a = x * LOG2E;

    // Floor via: subtract 1.0 if negative, then truncate toward zero.
    // This matches the SSE2 pattern: cmplt + and(1.0) + sub + cvttpd_epi32
    let p = if a < 0.0 { 1.0 } else { 0.0 };
    let a = a - p;
    let k = a as i32; // truncate toward zero (same as _mm_cvttpd_epi32)
    let p = f64::from(k);

    // Cody-Waite reduction: x -= p * log(2)  split as (C1 + C2)
    let x = x - p * C1;
    let x = x - p * C2;

    // Horner evaluation of degree-11 polynomial
    let mut a = W11;
    a = a * x + W10;
    a = a * x + W9;
    a = a * x + W8;
    a = a * x + W7;
    a = a * x + W6;
    a = a * x + W5;
    a = a * x + W4;
    a = a * x + W3;
    a = a * x + W2;
    a = a * x + W1;
    a = a * x + W0;

    // Multiply by 2^k via bit construction
    let pow2k = f64::from_bits(((k as i64 + 1023) as u64) << 52);
    a * pow2k
}

/// In-place vectorized exp using the CRFsuite polynomial approximation.
/// The input slice length should be a multiple of 4 for C compatibility,
/// but this works for any length.
pub fn vecexp(values: &mut [f64]) {
    for v in values.iter_mut() {
        *v = fast_exp(*v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_veczero() {
        let mut v = vec![1.0, 2.0, 3.0];
        veczero(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_vecset() {
        let mut v = vec![0.0; 4];
        vecset(&mut v, 5.0);
        assert_eq!(v, vec![5.0; 4]);
    }

    #[test]
    fn test_veccopy() {
        let src = vec![1.0, 2.0, 3.0];
        let mut dst = vec![0.0; 3];
        veccopy(&mut dst, &src);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_vecadd() {
        let mut y = vec![1.0, 2.0, 3.0];
        let x = vec![4.0, 5.0, 6.0];
        vecadd(&mut y, &x);
        assert_eq!(y, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vecaadd() {
        let mut y = vec![1.0, 2.0];
        let x = vec![3.0, 4.0];
        vecaadd(&mut y, 2.0, &x);
        assert_eq!(y, vec![7.0, 10.0]);
    }

    #[test]
    fn test_vecscale() {
        let mut y = vec![1.0, 2.0, 3.0];
        vecscale(&mut y, 3.0);
        assert_eq!(y, vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_vecdot() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        assert_eq!(vecdot(&x, &y), 32.0);
    }

    #[test]
    fn test_vecsum() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(vecsum(&x), 10.0);
    }

    #[test]
    fn test_fast_exp_zero() {
        assert_eq!(fast_exp(0.0), W0); // polynomial at 0 = w0
    }

    #[test]
    fn test_fast_exp_one() {
        let result = fast_exp(1.0);
        // Should be close to e ≈ 2.718281828...
        assert!((result - std::f64::consts::E).abs() < 1e-12);
    }

    #[test]
    fn test_fast_exp_clamping() {
        // Very large → clamped to MAXLOG
        let big = fast_exp(1000.0);
        let maxlog_result = fast_exp(MAXLOG);
        assert_eq!(big, maxlog_result);

        // Very small → clamped to MINLOG
        let small = fast_exp(-1000.0);
        let minlog_result = fast_exp(MINLOG);
        assert_eq!(small, minlog_result);
    }

    #[test]
    fn test_fast_exp_matches_c_sse_bits() {
        let mut values = [-1000.0, MINLOG, -10.0, -1.0, 0.0, 1.0, 10.0, MAXLOG];
        let expected = [
            0x0000000000000000,
            0x0000000000000000,
            0x3f07cd79b5647c9b,
            0x3fd78b56362cef38,
            0x3ff0000000000000,
            0x4005bf0a8b14576a,
            0x40d5829dcf950560,
            0x7ff0000000000000,
        ];

        vecexp(&mut values);

        for (value, bits) in values.iter().zip(expected.iter()) {
            assert_eq!(value.to_bits(), *bits);
        }
    }

    #[test]
    fn test_vecexp_range() {
        // Test a range of values and ensure results are reasonable
        let mut values: Vec<f64> = (-100..=100).map(|i| i as f64 * 0.1).collect();
        let originals = values.clone();
        vecexp(&mut values);
        for (x, result) in originals.iter().zip(values.iter()) {
            assert!(result.is_finite(), "exp({}) should be finite", x);
            assert!(*result > 0.0, "exp({}) should be positive", x);
            // Check within ~1e-10 relative error of std exp
            let expected = x.exp();
            let rel_err = (result - expected).abs() / expected.max(1e-300);
            assert!(rel_err < 1e-8, "exp({}) = {}, expected {}, rel_err = {}", x, result, expected, rel_err);
        }
    }
}

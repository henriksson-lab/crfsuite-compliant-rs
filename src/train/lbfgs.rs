/// L-BFGS training with L1/L2 regularization (via liblbfgs-sys).

use crate::crf1d::encode::Crf1dEncoder;
use crate::train::LogFn;

use std::os::raw::{c_int, c_void};

struct LbfgsContext<'a> {
    encoder: &'a mut Crf1dEncoder,
    instances: &'a [crate::types::Instance],
    c2: f64,
    log: &'a mut LogFn,
}

unsafe extern "C" fn evaluate_callback(
    instance: *mut c_void,
    x: *const liblbfgs_sys::lbfgsfloatval_t,
    g: *mut liblbfgs_sys::lbfgsfloatval_t,
    n: c_int,
    _step: liblbfgs_sys::lbfgsfloatval_t,
) -> liblbfgs_sys::lbfgsfloatval_t {
    let ctx = &mut *(instance as *mut LbfgsContext);
    let k = n as usize;
    let w = std::slice::from_raw_parts(x, k);
    let grad = std::slice::from_raw_parts_mut(g, k);

    let f = ctx.encoder.objective_and_gradients_batch(ctx.instances, w, grad);

    if ctx.c2 > 0.0 {
        let mut norm2 = 0.0f64;
        for i in 0..k {
            grad[i] += 2.0 * ctx.c2 * w[i];
            norm2 += w[i] * w[i];
        }
        return f + ctx.c2 * norm2;
    }

    f
}

unsafe extern "C" fn progress_callback(
    instance: *mut c_void,
    _x: *const liblbfgs_sys::lbfgsfloatval_t,
    _g: *const liblbfgs_sys::lbfgsfloatval_t,
    fx: liblbfgs_sys::lbfgsfloatval_t,
    xnorm: liblbfgs_sys::lbfgsfloatval_t,
    gnorm: liblbfgs_sys::lbfgsfloatval_t,
    step: liblbfgs_sys::lbfgsfloatval_t,
    _n: c_int,
    k: c_int,
    ls: c_int,
) -> c_int {
    let ctx = &mut *(instance as *mut LbfgsContext);
    (ctx.log)(&format!(
        "***** Iteration #{} *****\nLoss: {:.6}\nFeature norm: {:.6}\nError norm: {:.6}\nActive features: {}\nLine search trials: {}\nLine search step: {:.6}\nSeconds required for this iteration: 0.000\n\n",
        k, fx, xnorm, gnorm, ctx.encoder.num_features, ls, step
    ));
    0
}

pub fn train_lbfgs(
    encoder: &mut Crf1dEncoder,
    instances: &[crate::types::Instance],
    c1: f64,
    c2: f64,
    max_iterations: i32,
    num_memories: i32,
    epsilon: f64,
    period: i32,
    delta: f64,
    linesearch: &str,
    max_linesearch: i32,
    log: &mut LogFn,
) -> Vec<f64> {
    let k = encoder.num_features;
    let mut w = vec![0.0f64; k];

    let mut param: liblbfgs_sys::lbfgs_parameter_t = unsafe { std::mem::zeroed() };
    unsafe { liblbfgs_sys::lbfgs_parameter_init(&mut param) };

    param.m = num_memories;
    param.epsilon = epsilon;
    param.past = period;
    param.delta = delta;
    param.max_iterations = max_iterations;
    param.max_linesearch = max_linesearch;

    param.linesearch = match linesearch {
        "Backtracking" => 1,
        "StrongBacktracking" => 3,
        _ => 0, // MoreThuente
    };

    if c1 > 0.0 {
        param.orthantwise_c = c1;
        param.linesearch = 2;
        param.orthantwise_start = 0;
        param.orthantwise_end = k as c_int;
    }

    let mut ctx = LbfgsContext {
        encoder,
        instances,
        c2,
        log,
    };

    let mut fx: f64 = 0.0;
    let ret = unsafe {
        liblbfgs_sys::lbfgs(
            k as c_int,
            w.as_mut_ptr(),
            &mut fx,
            Some(evaluate_callback),
            Some(progress_callback),
            &mut ctx as *mut _ as *mut c_void,
            &mut param,
        )
    };

    let msg = match ret {
        0 => "L-BFGS resulted in convergence",
        1 => "L-BFGS stopped (already minimum)",
        2 => "L-BFGS stopped (maximum iterations)",
        _ => "L-BFGS finished",
    };
    (ctx.log)(&format!("{}\n", msg));

    w
}

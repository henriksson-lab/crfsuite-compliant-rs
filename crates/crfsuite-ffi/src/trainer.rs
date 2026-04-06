use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr::{self, NonNull};

use crfsuite_sys::*;

use crate::error::CrfError;
use crate::params::Params;

// Defined in logging_shim.c
extern "C" {
    fn crfsuite_ffi_logging_stdout(
        user: *mut c_void,
        format: *const std::os::raw::c_char,
        args: *mut crfsuite_sys::__va_list_tag,
    ) -> std::os::raw::c_int;
}

pub struct Trainer {
    ptr: NonNull<crfsuite_trainer_t>,
}

impl Trainer {
    /// Create a trainer instance for the given model type and algorithm.
    /// E.g., model_type="1d", algorithm="lbfgs"
    pub fn new(model_type: &str, algorithm: &str) -> Result<Self, CrfError> {
        let iid = format!("train/crf{}/{}", model_type, algorithm);
        let ciid = CString::new(iid.clone()).map_err(|_| CrfError::InvalidArgument("null byte".into()))?;
        let mut ptr: *mut c_void = ptr::null_mut();
        let ret = unsafe { crfsuite_create_instance(ciid.as_ptr(), &mut ptr) };
        if ret == 0 || ptr.is_null() {
            return Err(CrfError::InvalidArgument(format!("failed to create trainer: {}", iid)));
        }
        let ptr = NonNull::new(ptr as *mut crfsuite_trainer_t)
            .ok_or(CrfError::NullPointer("trainer"))?;
        Ok(Trainer { ptr })
    }

    /// Get the parameter interface.
    pub fn params(&self) -> Result<Params, CrfError> {
        unsafe {
            let t = self.ptr.as_ptr();
            let func = (*t).params.ok_or(CrfError::NullPointer("trainer.params"))?;
            let params_ptr = func(t);
            Params::from_raw(params_ptr, false)
        }
    }

    /// Set the logging callback to print to stdout (matches original C CLI behavior).
    pub fn set_stdout_logging(&self) {
        unsafe {
            let t = self.ptr.as_ptr();
            if let Some(func) = (*t).set_message_callback {
                func(t, ptr::null_mut(), Some(crfsuite_ffi_logging_stdout));
            }
        }
    }

    /// Run training.
    pub fn train(
        &self,
        data: *const crfsuite_data_t,
        model_path: &str,
        holdout: i32,
    ) -> Result<(), CrfError> {
        let cpath = CString::new(model_path).map_err(|_| CrfError::InvalidArgument("null byte".into()))?;
        unsafe {
            let t = self.ptr.as_ptr();
            let func = (*t).train.ok_or(CrfError::NullPointer("trainer.train"))?;
            let ret = func(t, data, cpath.as_ptr(), holdout);
            if ret != 0 {
                return Err(CrfError::InternalLogic);
            }
            Ok(())
        }
    }
}

impl Drop for Trainer {
    fn drop(&mut self) {
        unsafe {
            let t = self.ptr.as_ptr();
            if let Some(release) = (*t).release {
                release(t);
            }
        }
    }
}

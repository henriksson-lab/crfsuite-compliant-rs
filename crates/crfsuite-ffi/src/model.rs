use std::ffi::CString;
use std::ptr::{self, NonNull};

use crfsuite_sys::*;

use crate::dictionary::Dictionary;
use crate::error::CrfError;
use crate::tagger::Tagger;

pub struct Model {
    ptr: NonNull<crfsuite_model_t>,
}

impl Model {
    /// Load a model from a file.
    pub fn from_file(path: &str) -> Result<Self, CrfError> {
        let cpath = CString::new(path).map_err(|_| CrfError::InvalidArgument("null byte in path".into()))?;
        let mut model_ptr: *mut std::os::raw::c_void = ptr::null_mut();
        let ret = unsafe { crfsuite_create_instance_from_file(cpath.as_ptr(), &mut model_ptr) };
        if ret != 0 || model_ptr.is_null() {
            return Err(CrfError::InvalidArgument(format!("failed to load model: {}", path)));
        }
        let ptr = NonNull::new(model_ptr as *mut crfsuite_model_t)
            .ok_or(CrfError::NullPointer("model"))?;
        Ok(Model { ptr })
    }

    /// Get the tagger interface.
    pub fn get_tagger(&self) -> Result<Tagger, CrfError> {
        unsafe {
            let m = self.ptr.as_ptr();
            let func = (*m).get_tagger.ok_or(CrfError::NullPointer("model.get_tagger"))?;
            let mut tagger_ptr: *mut crfsuite_tagger_t = ptr::null_mut();
            let ret = func(m, &mut tagger_ptr);
            if ret != 0 {
                return Err(CrfError::InternalLogic);
            }
            Tagger::from_raw(tagger_ptr)
        }
    }

    /// Get the label dictionary.
    pub fn get_labels(&self) -> Result<Dictionary, CrfError> {
        unsafe {
            let m = self.ptr.as_ptr();
            let func = (*m).get_labels.ok_or(CrfError::NullPointer("model.get_labels"))?;
            let mut dict_ptr: *mut crfsuite_dictionary_t = ptr::null_mut();
            let ret = func(m, &mut dict_ptr);
            if ret != 0 {
                return Err(CrfError::InternalLogic);
            }
            Dictionary::from_raw(dict_ptr)
        }
    }

    /// Get the attribute dictionary.
    pub fn get_attrs(&self) -> Result<Dictionary, CrfError> {
        unsafe {
            let m = self.ptr.as_ptr();
            let func = (*m).get_attrs.ok_or(CrfError::NullPointer("model.get_attrs"))?;
            let mut dict_ptr: *mut crfsuite_dictionary_t = ptr::null_mut();
            let ret = func(m, &mut dict_ptr);
            if ret != 0 {
                return Err(CrfError::InternalLogic);
            }
            Dictionary::from_raw(dict_ptr)
        }
    }

    /// Dump the model to a FILE*.
    pub fn dump_to_file(&self, file: *mut libc::FILE) -> Result<(), CrfError> {
        unsafe {
            let m = self.ptr.as_ptr();
            let func = (*m).dump.ok_or(CrfError::NullPointer("model.dump"))?;
            let ret = func(m, file as *mut _);
            if ret != 0 {
                return Err(CrfError::InternalLogic);
            }
            Ok(())
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            let m = self.ptr.as_ptr();
            if let Some(release) = (*m).release {
                release(m);
            }
        }
    }
}

use std::ffi::{CStr, CString};
use std::ptr::NonNull;

use crfsuite_sys::*;

use crate::error::CrfError;

pub struct Dictionary {
    pub ptr: NonNull<crfsuite_dictionary_t>,
}

impl Dictionary {
    pub unsafe fn from_raw(ptr: *mut crfsuite_dictionary_t) -> Result<Self, CrfError> {
        NonNull::new(ptr).map(|p| Dictionary { ptr: p }).ok_or(CrfError::NullPointer("dictionary"))
    }

    /// Get or create an ID for a string. Returns the ID.
    pub fn get(&self, s: &str) -> Result<i32, CrfError> {
        let cs = CString::new(s).map_err(|_| CrfError::InvalidArgument("null byte in string".into()))?;
        unsafe {
            let dict = self.ptr.as_ptr();
            let func = (*dict).get.ok_or(CrfError::NullPointer("dictionary.get"))?;
            let id = func(dict, cs.as_ptr());
            Ok(id)
        }
    }

    /// Look up ID for string (read-only). Returns -1 if not found.
    pub fn to_id(&self, s: &str) -> Result<i32, CrfError> {
        let cs = CString::new(s).map_err(|_| CrfError::InvalidArgument("null byte in string".into()))?;
        unsafe {
            let dict = self.ptr.as_ptr();
            let func = (*dict).to_id.ok_or(CrfError::NullPointer("dictionary.to_id"))?;
            Ok(func(dict, cs.as_ptr()))
        }
    }

    /// Get string for an ID.
    pub fn to_string(&self, id: i32) -> Result<String, CrfError> {
        unsafe {
            let dict = self.ptr.as_ptr();
            let func = (*dict).to_string.ok_or(CrfError::NullPointer("dictionary.to_string"))?;
            let free_fn = (*dict).free.ok_or(CrfError::NullPointer("dictionary.free"))?;
            let mut pstr: *const std::os::raw::c_char = std::ptr::null();
            let ret = func(dict, id, &mut pstr);
            if ret != 0 || pstr.is_null() {
                return Err(CrfError::InvalidArgument(format!("unknown id: {}", id)));
            }
            let s = CStr::from_ptr(pstr).to_string_lossy().into_owned();
            free_fn(dict, pstr);
            Ok(s)
        }
    }

    /// Number of strings in the dictionary.
    pub fn num(&self) -> i32 {
        unsafe {
            let dict = self.ptr.as_ptr();
            if let Some(func) = (*dict).num {
                func(dict)
            } else {
                0
            }
        }
    }
}

impl Drop for Dictionary {
    fn drop(&mut self) {
        unsafe {
            let dict = self.ptr.as_ptr();
            if let Some(release) = (*dict).release {
                release(dict);
            }
        }
    }
}

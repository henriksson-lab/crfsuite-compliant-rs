use std::ffi::{CStr, CString};
use std::ptr::NonNull;

use crfsuite_sys::*;

use crate::error::CrfError;

pub struct Params {
    pub(crate) ptr: NonNull<crfsuite_params_t>,
    owned: bool,
}

impl Params {
    /// Wrap a params pointer. If `owned` is true, release() is called on drop.
    pub(crate) unsafe fn from_raw(ptr: *mut crfsuite_params_t, owned: bool) -> Result<Self, CrfError> {
        NonNull::new(ptr).map(|p| Params { ptr: p, owned }).ok_or(CrfError::NullPointer("params"))
    }

    pub fn num(&self) -> i32 {
        unsafe {
            let p = self.ptr.as_ptr();
            if let Some(func) = (*p).num {
                func(p)
            } else {
                0
            }
        }
    }

    pub fn name(&self, i: i32) -> Result<String, CrfError> {
        unsafe {
            let p = self.ptr.as_ptr();
            let func = (*p).name.ok_or(CrfError::NullPointer("params.name"))?;
            let free_fn = (*p).free.ok_or(CrfError::NullPointer("params.free"))?;
            let mut name_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
            func(p, i, &mut name_ptr);
            if name_ptr.is_null() {
                return Err(CrfError::InvalidArgument(format!("param index {} not found", i)));
            }
            let s = CStr::from_ptr(name_ptr).to_string_lossy().into_owned();
            free_fn(p, name_ptr);
            Ok(s)
        }
    }

    pub fn set(&self, name: &str, value: &str) -> Result<(), CrfError> {
        let cname = CString::new(name).map_err(|_| CrfError::InvalidArgument("null byte".into()))?;
        let cvalue = CString::new(value).map_err(|_| CrfError::InvalidArgument("null byte".into()))?;
        unsafe {
            let p = self.ptr.as_ptr();
            let func = (*p).set.ok_or(CrfError::NullPointer("params.set"))?;
            let ret = func(p, cname.as_ptr(), cvalue.as_ptr());
            if ret != 0 {
                return Err(CrfError::InvalidArgument(format!("unknown parameter: {}", name)));
            }
            Ok(())
        }
    }

    pub fn get(&self, name: &str) -> Result<String, CrfError> {
        let cname = CString::new(name).map_err(|_| CrfError::InvalidArgument("null byte".into()))?;
        unsafe {
            let p = self.ptr.as_ptr();
            let func = (*p).get.ok_or(CrfError::NullPointer("params.get"))?;
            let free_fn = (*p).free.ok_or(CrfError::NullPointer("params.free"))?;
            let mut val_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
            let ret = func(p, cname.as_ptr(), &mut val_ptr);
            if ret != 0 || val_ptr.is_null() {
                return Err(CrfError::InvalidArgument(format!("unknown parameter: {}", name)));
            }
            let s = CStr::from_ptr(val_ptr).to_string_lossy().into_owned();
            free_fn(p, val_ptr);
            Ok(s)
        }
    }

    pub fn help(&self, name: &str) -> Result<(String, String), CrfError> {
        let cname = CString::new(name).map_err(|_| CrfError::InvalidArgument("null byte".into()))?;
        unsafe {
            let p = self.ptr.as_ptr();
            let func = (*p).help.ok_or(CrfError::NullPointer("params.help"))?;
            let free_fn = (*p).free.ok_or(CrfError::NullPointer("params.free"))?;
            let mut type_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
            let mut help_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
            let ret = func(p, cname.as_ptr(), &mut type_ptr, &mut help_ptr);
            if ret != 0 {
                return Err(CrfError::InvalidArgument(format!("unknown parameter: {}", name)));
            }
            let type_s = if type_ptr.is_null() {
                String::new()
            } else {
                let s = CStr::from_ptr(type_ptr).to_string_lossy().into_owned();
                free_fn(p, type_ptr);
                s
            };
            let help_s = if help_ptr.is_null() {
                String::new()
            } else {
                let s = CStr::from_ptr(help_ptr).to_string_lossy().into_owned();
                free_fn(p, help_ptr);
                s
            };
            Ok((type_s, help_s))
        }
    }
}

impl Drop for Params {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                let p = self.ptr.as_ptr();
                if let Some(release) = (*p).release {
                    release(p);
                }
            }
        }
    }
}

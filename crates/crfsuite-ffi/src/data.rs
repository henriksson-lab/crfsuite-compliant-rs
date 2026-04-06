use crfsuite_sys::*;

use crate::error::{check_status, CrfError};

/// RAII wrapper for crfsuite_data_t. Manages the lifecycle of the C data structures.
pub struct Data {
    pub inner: crfsuite_data_t,
}

impl Data {
    pub fn new() -> Self {
        let mut inner: crfsuite_data_t = unsafe { std::mem::zeroed() };
        unsafe { crfsuite_data_init(&mut inner) };
        Data { inner }
    }

    pub fn set_dictionaries(
        &mut self,
        attrs: *mut crfsuite_dictionary_t,
        labels: *mut crfsuite_dictionary_t,
    ) {
        self.inner.attrs = attrs;
        self.inner.labels = labels;
    }

    pub fn append_instance(&mut self, inst: &crfsuite_instance_t) -> Result<(), CrfError> {
        let ret = unsafe { crfsuite_data_append(&mut self.inner, inst) };
        check_status(ret)
    }

    pub fn num_instances(&self) -> i32 {
        self.inner.num_instances
    }

    pub fn as_ptr(&self) -> *const crfsuite_data_t {
        &self.inner
    }

    pub fn as_mut_ptr(&mut self) -> *mut crfsuite_data_t {
        &mut self.inner
    }
}

impl Drop for Data {
    fn drop(&mut self) {
        unsafe {
            crfsuite_data_finish(&mut self.inner);
        }
    }
}

/// Helper to build a crfsuite_instance_t.
pub struct InstanceBuilder {
    pub inner: crfsuite_instance_t,
}

impl InstanceBuilder {
    pub fn new() -> Self {
        let mut inner: crfsuite_instance_t = unsafe { std::mem::zeroed() };
        unsafe { crfsuite_instance_init(&mut inner) };
        inner.weight = 1.0;
        InstanceBuilder { inner }
    }

    pub fn set_weight(&mut self, weight: f64) {
        self.inner.weight = weight;
    }

    pub fn set_group(&mut self, group: i32) {
        self.inner.group = group;
    }

    pub fn append(&mut self, item: &crfsuite_item_t, label: i32) -> Result<(), CrfError> {
        let ret = unsafe { crfsuite_instance_append(&mut self.inner, item, label) };
        check_status(ret)
    }

    pub fn is_empty(&self) -> bool {
        self.inner.num_items == 0
    }

    pub fn num_items(&self) -> i32 {
        self.inner.num_items
    }

    pub fn as_ptr(&self) -> *const crfsuite_instance_t {
        &self.inner
    }

    pub fn as_mut_ptr(&mut self) -> *mut crfsuite_instance_t {
        &mut self.inner
    }

    /// Reset the instance for reuse (finishes and re-inits).
    pub fn reset(&mut self) {
        unsafe {
            crfsuite_instance_finish(&mut self.inner);
            crfsuite_instance_init(&mut self.inner);
        }
        self.inner.weight = 1.0;
    }
}

impl Drop for InstanceBuilder {
    fn drop(&mut self) {
        unsafe {
            crfsuite_instance_finish(&mut self.inner);
        }
    }
}

/// Helper to build a crfsuite_item_t.
pub struct ItemBuilder {
    pub inner: crfsuite_item_t,
}

impl ItemBuilder {
    pub fn new() -> Self {
        let mut inner: crfsuite_item_t = unsafe { std::mem::zeroed() };
        unsafe { crfsuite_item_init(&mut inner) };
        ItemBuilder { inner }
    }

    pub fn append_attribute(&mut self, aid: i32, value: f64) -> Result<(), CrfError> {
        let mut attr: crfsuite_attribute_t = unsafe { std::mem::zeroed() };
        unsafe { crfsuite_attribute_set(&mut attr, aid, value) };
        let ret = unsafe { crfsuite_item_append_attribute(&mut self.inner, &attr) };
        check_status(ret)
    }

    pub fn as_ptr(&self) -> *const crfsuite_item_t {
        &self.inner
    }

    /// Reset the item for reuse.
    pub fn reset(&mut self) {
        unsafe {
            crfsuite_item_finish(&mut self.inner);
            crfsuite_item_init(&mut self.inner);
        }
    }
}

impl Drop for ItemBuilder {
    fn drop(&mut self) {
        unsafe {
            crfsuite_item_finish(&mut self.inner);
        }
    }
}

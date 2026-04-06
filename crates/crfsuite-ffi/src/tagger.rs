use std::ptr::NonNull;

use crfsuite_sys::*;

use crate::error::{check_status, CrfError};

pub struct Tagger {
    pub(crate) ptr: NonNull<crfsuite_tagger_t>,
}

impl Tagger {
    pub(crate) unsafe fn from_raw(ptr: *mut crfsuite_tagger_t) -> Result<Self, CrfError> {
        NonNull::new(ptr).map(|p| Tagger { ptr: p }).ok_or(CrfError::NullPointer("tagger"))
    }

    /// Set an instance on the tagger for inference.
    pub fn set_instance(&self, inst: *mut crfsuite_instance_t) -> Result<(), CrfError> {
        unsafe {
            let t = self.ptr.as_ptr();
            let func = (*t).set.ok_or(CrfError::NullPointer("tagger.set"))?;
            check_status(func(t, inst))
        }
    }

    /// Get the number of items in the current instance.
    pub fn length(&self) -> Result<i32, CrfError> {
        unsafe {
            let t = self.ptr.as_ptr();
            let func = (*t).length.ok_or(CrfError::NullPointer("tagger.length"))?;
            Ok(func(t))
        }
    }

    /// Run Viterbi decoding. Returns (labels, score).
    pub fn viterbi(&self, num_items: usize) -> Result<(Vec<i32>, f64), CrfError> {
        let mut labels = vec![0i32; num_items];
        let mut score: f64 = 0.0;
        unsafe {
            let t = self.ptr.as_ptr();
            let func = (*t).viterbi.ok_or(CrfError::NullPointer("tagger.viterbi"))?;
            check_status(func(t, labels.as_mut_ptr(), &mut score))?;
        }
        Ok((labels, score))
    }

    /// Compute the score of a label path.
    pub fn score(&self, path: &[i32]) -> Result<f64, CrfError> {
        let mut score: f64 = 0.0;
        unsafe {
            let t = self.ptr.as_ptr();
            let func = (*t).score.ok_or(CrfError::NullPointer("tagger.score"))?;
            check_status(func(t, path.as_ptr() as *mut i32, &mut score))?;
        }
        Ok(score)
    }

    /// Compute log of the partition factor.
    pub fn lognorm(&self) -> Result<f64, CrfError> {
        let mut norm: f64 = 0.0;
        unsafe {
            let t = self.ptr.as_ptr();
            let func = (*t).lognorm.ok_or(CrfError::NullPointer("tagger.lognorm"))?;
            check_status(func(t, &mut norm))?;
        }
        Ok(norm)
    }

    /// Compute marginal probability P(y_t = l | x).
    pub fn marginal_point(&self, label: i32, position: i32) -> Result<f64, CrfError> {
        let mut prob: f64 = 0.0;
        unsafe {
            let t = self.ptr.as_ptr();
            let func = (*t).marginal_point.ok_or(CrfError::NullPointer("tagger.marginal_point"))?;
            check_status(func(t, label, position, &mut prob))?;
        }
        Ok(prob)
    }

    /// Compute marginal probability of a partial label sequence.
    pub fn marginal_path(&self, path: &[i32], begin: i32, end: i32) -> Result<f64, CrfError> {
        let mut prob: f64 = 0.0;
        unsafe {
            let t = self.ptr.as_ptr();
            let func = (*t).marginal_path.ok_or(CrfError::NullPointer("tagger.marginal_path"))?;
            check_status(func(t, path.as_ptr(), begin, end, &mut prob))?;
        }
        Ok(prob)
    }
}

impl Drop for Tagger {
    fn drop(&mut self) {
        unsafe {
            let t = self.ptr.as_ptr();
            if let Some(release) = (*t).release {
                release(t);
            }
        }
    }
}

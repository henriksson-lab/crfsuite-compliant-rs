use crfsuite_sys::*;

use crate::error::CrfError;

pub struct Evaluation {
    pub(crate) inner: crfsuite_evaluation_t,
}

impl Evaluation {
    pub fn new(num_labels: i32) -> Self {
        let mut inner: crfsuite_evaluation_t = unsafe { std::mem::zeroed() };
        unsafe { crfsuite_evaluation_init(&mut inner, num_labels) };
        Evaluation { inner }
    }

    pub fn clear(&mut self) {
        unsafe { crfsuite_evaluation_clear(&mut self.inner) };
    }

    pub fn accumulate(&mut self, reference: &[i32], prediction: &[i32]) -> Result<(), CrfError> {
        assert_eq!(reference.len(), prediction.len());
        let ret = unsafe {
            crfsuite_evaluation_accmulate(
                &mut self.inner,
                reference.as_ptr(),
                prediction.as_ptr(),
                reference.len() as i32,
            )
        };
        if ret != 0 {
            return Err(CrfError::InternalLogic);
        }
        Ok(())
    }

    pub fn finalize(&mut self) {
        unsafe { crfsuite_evaluation_finalize(&mut self.inner) };
    }

    pub fn item_accuracy(&self) -> f64 {
        self.inner.item_accuracy
    }

    pub fn item_total_correct(&self) -> i32 {
        self.inner.item_total_correct
    }

    pub fn item_total_num(&self) -> i32 {
        self.inner.item_total_num
    }

    pub fn inst_accuracy(&self) -> f64 {
        self.inner.inst_accuracy
    }

    pub fn inst_total_correct(&self) -> i32 {
        self.inner.inst_total_correct
    }

    pub fn inst_total_num(&self) -> i32 {
        self.inner.inst_total_num
    }

    pub fn macro_precision(&self) -> f64 {
        self.inner.macro_precision
    }

    pub fn macro_recall(&self) -> f64 {
        self.inner.macro_recall
    }

    pub fn macro_fmeasure(&self) -> f64 {
        self.inner.macro_fmeasure
    }

    pub fn num_labels(&self) -> i32 {
        self.inner.num_labels
    }

    /// Get per-label evaluation for label index i.
    pub fn label_eval(&self, i: i32) -> Option<LabelEval> {
        if i < 0 || i >= self.inner.num_labels || self.inner.tbl.is_null() {
            return None;
        }
        unsafe {
            let entry = &*self.inner.tbl.add(i as usize);
            Some(LabelEval {
                num_correct: entry.num_correct,
                num_observation: entry.num_observation,
                num_model: entry.num_model,
                precision: entry.precision,
                recall: entry.recall,
                fmeasure: entry.fmeasure,
            })
        }
    }
}

impl Drop for Evaluation {
    fn drop(&mut self) {
        unsafe { crfsuite_evaluation_finish(&mut self.inner) };
    }
}

#[derive(Debug, Clone)]
pub struct LabelEval {
    pub num_correct: i32,
    pub num_observation: i32,
    pub num_model: i32,
    pub precision: f64,
    pub recall: f64,
    pub fmeasure: f64,
}

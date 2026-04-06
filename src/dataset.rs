/// Dataset wrapper with permutation array for shuffling and train/test splits.

use crate::types::Instance;

pub struct Dataset<'a> {
    pub data: &'a [Instance],
    pub perm: Vec<usize>,
}

impl<'a> Dataset<'a> {
    /// Create training set: all instances whose group != holdout.
    pub fn init_trainset(data: &'a [Instance], holdout: i32) -> Self {
        let perm: Vec<usize> = data.iter().enumerate()
            .filter(|(_, inst)| holdout < 0 || inst.group != holdout)
            .map(|(i, _)| i)
            .collect();
        Dataset { data, perm }
    }

    /// Create test set: all instances whose group == holdout.
    pub fn init_testset(data: &'a [Instance], holdout: i32) -> Self {
        let perm: Vec<usize> = data.iter().enumerate()
            .filter(|(_, inst)| inst.group == holdout)
            .map(|(i, _)| i)
            .collect();
        Dataset { data, perm }
    }

    pub fn num_instances(&self) -> usize {
        self.perm.len()
    }

    pub fn get(&self, i: usize) -> &Instance {
        &self.data[self.perm[i]]
    }

    /// Fisher-Yates shuffle using libc rand() for C compatibility.
    pub fn shuffle(&mut self) {
        let n = self.perm.len();
        for i in 0..n {
            let j = crate::rng::rand_int() % n;
            self.perm.swap(i, j);
        }
    }
}

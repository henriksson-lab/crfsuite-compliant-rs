//! Pure Rust data types for CRFsuite sequences.

#[derive(Debug, Clone)]
pub struct Attribute {
    pub aid: i32,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct Item {
    pub contents: Vec<Attribute>,
}

#[derive(Debug, Clone)]
pub struct Instance {
    pub items: Vec<Item>,
    pub labels: Vec<i32>,
    pub weight: f64,
    pub group: i32,
}

impl Default for Instance {
    fn default() -> Self {
        Instance {
            items: Vec::new(),
            labels: Vec::new(),
            weight: 1.0,
            group: 0,
        }
    }
}

impl Instance {
    pub fn new() -> Self {
        Instance {
            items: Vec::new(),
            labels: Vec::new(),
            weight: 1.0,
            group: 0,
        }
    }

    pub fn num_items(&self) -> usize {
        self.items.len()
    }
}

/// String-to-sequential-ID mapping (replaces C quark.c + rumavl.c).
///
/// IDs are assigned sequentially starting from 0 in insertion order.

use std::collections::HashMap;

pub struct Quark {
    string_to_id: HashMap<String, i32>,
    id_to_string: Vec<String>,
}

impl Quark {
    pub fn new() -> Self {
        Quark {
            string_to_id: HashMap::new(),
            id_to_string: Vec::new(),
        }
    }

    /// Get or create an ID for the string. Returns the ID.
    pub fn get(&mut self, s: &str) -> i32 {
        if let Some(&id) = self.string_to_id.get(s) {
            return id;
        }
        let id = self.id_to_string.len() as i32;
        self.string_to_id.insert(s.to_string(), id);
        self.id_to_string.push(s.to_string());
        id
    }

    /// Look up ID for string (read-only). Returns None if not found.
    pub fn to_id(&self, s: &str) -> Option<i32> {
        self.string_to_id.get(s).copied()
    }

    /// Look up string for ID.
    pub fn to_string(&self, id: i32) -> Option<&str> {
        if id < 0 {
            return None;
        }
        self.id_to_string.get(id as usize).map(|s| s.as_str())
    }

    /// Number of strings.
    pub fn num(&self) -> usize {
        self.id_to_string.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut q = Quark::new();
        assert_eq!(q.get("hello"), 0);
        assert_eq!(q.get("world"), 1);
        assert_eq!(q.get("hello"), 0); // existing
        assert_eq!(q.num(), 2);
        assert_eq!(q.to_id("hello"), Some(0));
        assert_eq!(q.to_id("world"), Some(1));
        assert_eq!(q.to_id("missing"), None);
        assert_eq!(q.to_string(0), Some("hello"));
        assert_eq!(q.to_string(1), Some("world"));
        assert_eq!(q.to_string(2), None);
    }

    #[test]
    fn test_sequential_ids() {
        let mut q = Quark::new();
        for i in 0..100 {
            let s = format!("item_{}", i);
            assert_eq!(q.get(&s), i);
        }
        assert_eq!(q.num(), 100);
    }
}

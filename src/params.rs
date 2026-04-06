/// Parameter store for training algorithms.

#[derive(Debug, Clone)]
pub enum ParamValue {
    Int(i32),
    Float(f64),
    String(String),
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub value: ParamValue,
    pub help: String,
}

pub struct Params {
    params: Vec<Param>,
}

impl Params {
    pub fn new() -> Self {
        Params { params: Vec::new() }
    }

    pub fn add_int(&mut self, name: &str, value: i32, help: &str) {
        self.params.push(Param {
            name: name.to_string(),
            value: ParamValue::Int(value),
            help: help.to_string(),
        });
    }

    pub fn add_float(&mut self, name: &str, value: f64, help: &str) {
        self.params.push(Param {
            name: name.to_string(),
            value: ParamValue::Float(value),
            help: help.to_string(),
        });
    }

    pub fn add_string(&mut self, name: &str, value: &str, help: &str) {
        self.params.push(Param {
            name: name.to_string(),
            value: ParamValue::String(value.to_string()),
            help: help.to_string(),
        });
    }

    fn find(&self, name: &str) -> Option<usize> {
        self.params.iter().position(|p| p.name == name)
    }

    pub fn set(&mut self, name: &str, value: &str) -> bool {
        if let Some(idx) = self.find(name) {
            match &self.params[idx].value {
                ParamValue::Int(_) => {
                    if let Ok(v) = value.parse::<i32>() {
                        self.params[idx].value = ParamValue::Int(v);
                        return true;
                    }
                }
                ParamValue::Float(_) => {
                    if let Ok(v) = value.parse::<f64>() {
                        self.params[idx].value = ParamValue::Float(v);
                        return true;
                    }
                }
                ParamValue::String(_) => {
                    self.params[idx].value = ParamValue::String(value.to_string());
                    return true;
                }
            }
        }
        false
    }

    pub fn get_int(&self, name: &str) -> Option<i32> {
        self.find(name).and_then(|i| match self.params[i].value {
            ParamValue::Int(v) => Some(v),
            _ => None,
        })
    }

    pub fn get_float(&self, name: &str) -> Option<f64> {
        self.find(name).and_then(|i| match self.params[i].value {
            ParamValue::Float(v) => Some(v),
            _ => None,
        })
    }

    pub fn get_string(&self, name: &str) -> Option<&str> {
        self.find(name).and_then(|i| match &self.params[i].value {
            ParamValue::String(v) => Some(v.as_str()),
            _ => None,
        })
    }

    pub fn num(&self) -> usize {
        self.params.len()
    }

    pub fn name(&self, i: usize) -> Option<&str> {
        self.params.get(i).map(|p| p.name.as_str())
    }

    pub fn help(&self, name: &str) -> Option<&str> {
        self.find(name).map(|i| self.params[i].help.as_str())
    }
}

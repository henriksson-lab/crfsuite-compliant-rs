#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

// Re-export liblbfgs-sys so it gets linked
extern crate liblbfgs_sys;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

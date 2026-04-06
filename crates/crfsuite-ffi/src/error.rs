use std::fmt;

#[derive(Debug, Clone)]
pub enum CrfError {
    Unknown,
    OutOfMemory,
    NotSupported,
    Incompatible,
    InternalLogic,
    Overflow,
    NotImplemented,
    NullPointer(&'static str),
    InvalidArgument(String),
}

impl fmt::Display for CrfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CrfError::Unknown => write!(f, "Unknown error"),
            CrfError::OutOfMemory => write!(f, "Out of memory"),
            CrfError::NotSupported => write!(f, "Not supported"),
            CrfError::Incompatible => write!(f, "Incompatible data"),
            CrfError::InternalLogic => write!(f, "Internal logic error"),
            CrfError::Overflow => write!(f, "Overflow"),
            CrfError::NotImplemented => write!(f, "Not implemented"),
            CrfError::NullPointer(ctx) => write!(f, "Null pointer: {}", ctx),
            CrfError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
        }
    }
}

impl std::error::Error for CrfError {}

pub fn check_status(code: i32) -> Result<(), CrfError> {
    if code >= 0 {
        Ok(())
    } else {
        Err(status_to_error(code))
    }
}

fn status_to_error(code: i32) -> CrfError {
    match code {
        c if c == crfsuite_sys::CRFSUITEERR_OUTOFMEMORY => CrfError::OutOfMemory,
        c if c == crfsuite_sys::CRFSUITEERR_NOTSUPPORTED => CrfError::NotSupported,
        c if c == crfsuite_sys::CRFSUITEERR_INCOMPATIBLE => CrfError::Incompatible,
        c if c == crfsuite_sys::CRFSUITEERR_INTERNAL_LOGIC => CrfError::InternalLogic,
        c if c == crfsuite_sys::CRFSUITEERR_OVERFLOW => CrfError::Overflow,
        c if c == crfsuite_sys::CRFSUITEERR_NOTIMPLEMENTED => CrfError::NotImplemented,
        _ => CrfError::Unknown,
    }
}

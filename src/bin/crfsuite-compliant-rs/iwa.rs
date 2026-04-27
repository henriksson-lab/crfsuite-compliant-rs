use std::io::BufRead;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    None,
    Eof,
    Boi,
    Eoi,
    Item,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub attr: String,
    pub value: String,
}

/// Field parsed from a line: (attr, value) as owned Strings.
/// We keep them in a Vec to avoid re-parsing.
struct Field {
    attr: String,
    value: String,
}

/// IWA (Item With Attributes) format parser.
/// Parses tab-separated fields from lines, with blank lines separating sequences.
pub struct IwaReader<R: BufRead> {
    reader: R,
    line: String,
    fields: Vec<Field>,
    field_idx: usize,
    state: State,
}

#[derive(Debug, Clone, PartialEq)]
enum State {
    Start,
    InItem,
    EndOfItem,
    Done,
}

impl<R: BufRead> IwaReader<R> {
    pub fn new(reader: R) -> Self {
        IwaReader {
            reader,
            line: String::new(),
            fields: Vec::with_capacity(16),
            field_idx: 0,
            state: State::Start,
        }
    }

    /// Read the next token.
    pub fn read(&mut self) -> Token {
        loop {
            match self.state {
                State::Done => {
                    return Token {
                        token_type: TokenType::Eof,
                        attr: String::new(),
                        value: String::new(),
                    };
                }
                State::Start | State::EndOfItem => {
                    self.line.clear();
                    match self.reader.read_line(&mut self.line) {
                        Ok(0) => {
                            if self.state == State::EndOfItem {
                                self.state = State::Done;
                                return Token {
                                    token_type: TokenType::None,
                                    attr: String::new(),
                                    value: String::new(),
                                };
                            }
                            self.state = State::Done;
                            return Token {
                                token_type: TokenType::Eof,
                                attr: String::new(),
                                value: String::new(),
                            };
                        }
                        Ok(_) => {
                            let end = self.line.trim_end_matches('\n').len();
                            if end == 0 {
                                if self.state == State::EndOfItem {
                                    self.state = State::Start;
                                    return Token {
                                        token_type: TokenType::None,
                                        attr: String::new(),
                                        value: String::new(),
                                    };
                                }
                                continue;
                            }
                            self.parse_line(end);
                            self.field_idx = 0;
                            self.state = State::InItem;
                            return Token {
                                token_type: TokenType::Boi,
                                attr: String::new(),
                                value: String::new(),
                            };
                        }
                        Err(_) => {
                            self.state = State::Done;
                            return Token {
                                token_type: TokenType::Eof,
                                attr: String::new(),
                                value: String::new(),
                            };
                        }
                    }
                }
                State::InItem => {
                    if self.field_idx < self.fields.len() {
                        let f = &mut self.fields[self.field_idx];
                        self.field_idx += 1;
                        return Token {
                            token_type: TokenType::Item,
                            attr: std::mem::take(&mut f.attr),
                            value: std::mem::take(&mut f.value),
                        };
                    } else {
                        self.state = State::EndOfItem;
                        return Token {
                            token_type: TokenType::Eoi,
                            attr: String::new(),
                            value: String::new(),
                        };
                    }
                }
            }
        }
    }

    fn parse_line(&mut self, end: usize) {
        self.fields.clear();
        let line = &self.line[..end];
        for field in line.split('\t') {
            if field.is_empty() {
                continue;
            }
            let (attr, value) = parse_field(field);
            self.fields.push(Field { attr, value });
        }
    }
}

/// Parse a single field into (attr, value) splitting at first unescaped colon.
fn parse_field(field: &str) -> (String, String) {
    let bytes = field.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut attr = String::with_capacity(len);

    while i < len {
        let b = bytes[i];
        if b == b'\\' && i + 1 < len {
            let next = bytes[i + 1];
            if next == b'\\' || next == b':' {
                attr.push(next as char);
                i += 2;
                continue;
            }
        }
        if b == b':' {
            let value = parse_value(&bytes[i + 1..]);
            return (attr, value);
        }
        attr.push(b as char);
        i += 1;
    }

    (attr, String::new())
}

fn parse_value(bytes: &[u8]) -> String {
    let mut value = String::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\\' && i + 1 < bytes.len() {
            let next = bytes[i + 1];
            if next == b'\\' || next == b':' {
                value.push(next as char);
                i += 2;
                continue;
            }
        }
        if b == b':' {
            break;
        }
        value.push(b as char);
        i += 1;
    }
    value
}

/// Parse a floating point value with C `atof`-style prefix handling.
///
/// CRFsuite's frontend uses `atof`, so values like `2abc` are accepted as
/// `2.0`, while completely invalid values become `0.0`.
pub fn atof(value: &str) -> f64 {
    let value = value.trim_start();
    if let Some(v) = parse_hex_float(value) {
        return v;
    }
    for end in (1..=value.len()).rev() {
        if value.is_char_boundary(end) {
            if let Ok(v) = value[..end].parse::<f64>() {
                return v;
            }
        }
    }
    0.0
}

fn parse_hex_float(value: &str) -> Option<f64> {
    let bytes = value.as_bytes();
    let mut i = 0;
    let sign = if bytes.get(i) == Some(&b'-') {
        i += 1;
        -1.0
    } else {
        if bytes.get(i) == Some(&b'+') {
            i += 1;
        }
        1.0
    };

    if bytes.get(i) != Some(&b'0') || !matches!(bytes.get(i + 1), Some(b'x' | b'X')) {
        return None;
    }
    i += 2;

    let mut mantissa = 0.0;
    let mut digits = 0;
    while let Some(b) = bytes.get(i).copied() {
        let Some(digit) = hex_digit(b) else {
            break;
        };
        mantissa = mantissa * 16.0 + digit as f64;
        digits += 1;
        i += 1;
    }

    if bytes.get(i) == Some(&b'.') {
        i += 1;
        let mut scale = 1.0 / 16.0;
        while let Some(b) = bytes.get(i).copied() {
            let Some(digit) = hex_digit(b) else {
                break;
            };
            mantissa += digit as f64 * scale;
            scale /= 16.0;
            digits += 1;
            i += 1;
        }
    }

    if digits == 0 {
        return None;
    }

    if !matches!(bytes.get(i), Some(b'p' | b'P')) {
        return Some(sign * mantissa);
    }
    i += 1;

    let exp_sign = if bytes.get(i) == Some(&b'-') {
        i += 1;
        -1
    } else {
        if bytes.get(i) == Some(&b'+') {
            i += 1;
        }
        1
    };

    let exp_start = i;
    let mut exponent: i32 = 0;
    while let Some(b) = bytes.get(i).copied() {
        if !b.is_ascii_digit() {
            break;
        }
        exponent = exponent
            .saturating_mul(10)
            .saturating_add((b - b'0') as i32);
        i += 1;
    }
    if i == exp_start {
        return Some(sign * mantissa);
    }

    Some(sign * mantissa * 2.0_f64.powi(exp_sign * exponent))
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Parse an integer with C `atoi`-style prefix handling.
pub fn atoi(value: &str) -> i32 {
    let value = value.trim_start();
    let mut last = 0;
    for (i, ch) in value.char_indices() {
        if i == 0 && (ch == '-' || ch == '+') {
            last = ch.len_utf8();
            continue;
        }
        if ch.is_ascii_digit() {
            last = i + ch.len_utf8();
        } else {
            break;
        }
    }
    if last == 0 || value[..last].chars().all(|ch| ch == '-' || ch == '+') {
        return 0;
    }
    let parsed = value[..last].parse::<i128>().unwrap_or_else(|_| {
        if value.starts_with('-') {
            i128::from(i64::MIN)
        } else {
            i128::from(i64::MAX)
        }
    });
    parsed.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64 as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_field() {
        assert_eq!(parse_field("attr"), ("attr".to_string(), "".to_string()));
        assert_eq!(
            parse_field("attr:1.0"),
            ("attr".to_string(), "1.0".to_string())
        );
        assert_eq!(
            parse_field("a\\:b:val"),
            ("a:b".to_string(), "val".to_string())
        );
        assert_eq!(
            parse_field("attr:a\\:b\\\\c"),
            ("attr".to_string(), "a:b\\c".to_string())
        );
        assert_eq!(
            parse_field("attr:a:b"),
            ("attr".to_string(), "a".to_string())
        );
    }

    #[test]
    fn test_atof_prefix_handling() {
        assert_eq!(atof("2abc"), 2.0);
        assert_eq!(atof("1e+"), 1.0);
        assert_eq!(atof("not-a-number"), 0.0);
        assert_eq!(atof("  -3.5x"), -3.5);
        assert_eq!(atof("0x1p2"), 4.0);
        assert_eq!(atof("0x1.8p+1tail"), 3.0);
        assert!(atof("nan(payload)").is_nan());
        assert!(atof("Infinity").is_infinite());
    }

    #[test]
    fn test_atoi_prefix_handling() {
        assert_eq!(atoi("12abc"), 12);
        assert_eq!(atoi("  -7x"), -7);
        assert_eq!(atoi("+5"), 5);
        assert_eq!(atoi("not-a-number"), 0);
        assert_eq!(atoi("-"), 0);
        assert_eq!(atoi("2147483648"), i32::MIN);
        assert_eq!(atoi("999999999999999999999999"), -1);
        assert_eq!(atoi("-2147483649"), i32::MAX);
        assert_eq!(atoi("-999999999999999999999999"), 0);
    }

    #[test]
    fn test_basic_sequence() {
        let input = "LABEL1\tattr1:1.0\tattr2:2.0\nLABEL2\tattr3\n\n";
        let mut parser = IwaReader::new(Cursor::new(input));

        assert_eq!(parser.read().token_type, TokenType::Boi);
        let t = parser.read();
        assert_eq!(t.token_type, TokenType::Item);
        assert_eq!(t.attr, "LABEL1");
        let t = parser.read();
        assert_eq!(t.token_type, TokenType::Item);
        assert_eq!(t.attr, "attr1");
        assert_eq!(t.value, "1.0");
        let t = parser.read();
        assert_eq!(t.token_type, TokenType::Item);
        assert_eq!(t.attr, "attr2");
        assert_eq!(t.value, "2.0");
        assert_eq!(parser.read().token_type, TokenType::Eoi);

        assert_eq!(parser.read().token_type, TokenType::Boi);
        let t = parser.read();
        assert_eq!(t.attr, "LABEL2");
        let t = parser.read();
        assert_eq!(t.attr, "attr3");
        assert_eq!(parser.read().token_type, TokenType::Eoi);

        assert_eq!(parser.read().token_type, TokenType::None);
    }
}

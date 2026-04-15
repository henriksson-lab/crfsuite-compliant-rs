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
                            let end = self.line.trim_end_matches(['\n', '\r']).len();
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
            // Found unescaped colon — rest is value
            let value = String::from_utf8_lossy(&bytes[i + 1..]).into_owned();
            return (attr, value);
        }
        attr.push(b as char);
        i += 1;
    }

    (attr, String::new())
}

/// Parse a floating point value with C `atof`-style prefix handling.
///
/// CRFsuite's frontend uses `atof`, so values like `2abc` are accepted as
/// `2.0`, while completely invalid values become `0.0`.
pub fn atof(value: &str) -> f64 {
    let value = value.trim_start();
    for end in (1..=value.len()).rev() {
        if value.is_char_boundary(end) {
            if let Ok(v) = value[..end].parse::<f64>() {
                return v;
            }
        }
    }
    0.0
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
    value[..last].parse::<i32>().unwrap_or(0)
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
    }

    #[test]
    fn test_atof_prefix_handling() {
        assert_eq!(atof("2abc"), 2.0);
        assert_eq!(atof("1e+"), 1.0);
        assert_eq!(atof("not-a-number"), 0.0);
        assert_eq!(atof("  -3.5x"), -3.5);
    }

    #[test]
    fn test_atoi_prefix_handling() {
        assert_eq!(atoi("12abc"), 12);
        assert_eq!(atoi("  -7x"), -7);
        assert_eq!(atoi("+5"), 5);
        assert_eq!(atoi("not-a-number"), 0);
        assert_eq!(atoi("-"), 0);
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

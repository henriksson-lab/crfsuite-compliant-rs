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

/// IWA (Item With Attributes) format parser.
/// Parses tab-separated fields from lines, with blank lines separating sequences.
pub struct IwaReader<R: BufRead> {
    reader: R,
    line: String,
    // Fields parsed from current line
    fields: Vec<(String, String)>,
    field_idx: usize,
    // State
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
            fields: Vec::new(),
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
                    // Read next line
                    self.line.clear();
                    match self.reader.read_line(&mut self.line) {
                        Ok(0) => {
                            // EOF
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
                            let trimmed = self.line.trim_end_matches(|c| c == '\n' || c == '\r').to_owned();
                            if trimmed.is_empty() {
                                // Blank line = sequence separator
                                if self.state == State::EndOfItem {
                                    self.state = State::Start;
                                    return Token {
                                        token_type: TokenType::None,
                                        attr: String::new(),
                                        value: String::new(),
                                    };
                                }
                                // Skip consecutive blank lines at start
                                continue;
                            }
                            // Parse fields from the line
                            self.parse_line(&trimmed);
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
                        let (attr, value) = self.fields[self.field_idx].clone();
                        self.field_idx += 1;
                        return Token {
                            token_type: TokenType::Item,
                            attr,
                            value,
                        };
                    } else {
                        // All fields consumed
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

    /// Parse a line into tab-separated fields. Each field is split at the first
    /// unescaped colon into (attr, value). Supports \\ and \: escapes.
    fn parse_line(&mut self, line: &str) {
        self.fields.clear();
        for field in line.split('\t') {
            if field.is_empty() {
                continue;
            }
            let (attr, value) = parse_field(field);
            self.fields.push((attr, value));
        }
    }
}

/// Parse a single field into (attr, value) splitting at first unescaped colon.
/// Handles \\ and \: escape sequences.
fn parse_field(field: &str) -> (String, String) {
    let mut attr = String::new();
    let mut chars = field.chars().peekable();
    let mut found_colon = false;

    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(&next) = chars.peek() {
                match next {
                    '\\' => {
                        attr.push('\\');
                        chars.next();
                    }
                    ':' => {
                        attr.push(':');
                        chars.next();
                    }
                    _ => {
                        attr.push(c);
                    }
                }
            } else {
                attr.push(c);
            }
        } else if c == ':' {
            found_colon = true;
            break;
        } else {
            attr.push(c);
        }
    }

    if found_colon {
        let value: String = chars.collect();
        (attr, value)
    } else {
        (attr, String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_field() {
        assert_eq!(parse_field("attr"), ("attr".to_string(), "".to_string()));
        assert_eq!(parse_field("attr:1.0"), ("attr".to_string(), "1.0".to_string()));
        assert_eq!(parse_field("a\\:b:val"), ("a:b".to_string(), "val".to_string()));
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

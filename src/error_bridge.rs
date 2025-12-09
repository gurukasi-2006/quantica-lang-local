// src/error_bridge.rs

use crate::error::{Diagnostic, ErrorReporter, ErrorCategory};
use crate::error_codes::ErrorCode;
use crate::parser::ast::Loc;


pub fn parse_error_location(error_msg: &str) -> Loc {
    if let Some(start) = error_msg.find("[Line ") {
        let rest = &error_msg[start..];
        if let Some(end) = rest.find(']') {
            let content = &rest[6..end];
            let parts: Vec<&str> = content.split(", Col ").collect();
            if parts.len() == 2 {
                if let (Ok(line), Ok(column)) = (parts[0].parse(), parts[1].parse()) {
                    return Loc { line, column };
                }
            }
        }
    }
    if let Some(line_pos) = error_msg.find("line ") {
        let rest = &error_msg[line_pos + 5..];
        if let Some(comma_pos) = rest.find(',') {
            let line_str = &rest[..comma_pos].trim();
            if let Ok(line) = line_str.parse::<usize>() {
                if let Some(col_pos) = rest.find("column ") {
                    let col_rest = &rest[col_pos + 7..];
                    // Find the end of the number
                    let col_end = col_rest.find(|c: char| !c.is_numeric()).unwrap_or(col_rest.len());
                    if let Ok(column) = col_rest[..col_end].parse::<usize>() {
                        return Loc { line, column };
                    }
                }
            }
        }
    }

    Loc { line: 1, column: 1 }
}

pub fn detect_error_code(error_msg: &str) -> ErrorCode {
    let msg_lower = error_msg.to_lowercase();

    // Syntax errors
    if msg_lower.contains("unexpected token") {
        return ErrorCode::E100;
    }

    //token expectation errors
    if msg_lower.contains("expected") && (msg_lower.contains("found") || msg_lower.contains("but")) {
        // Check for specific delimiters
        if msg_lower.contains("colon") || msg_lower.contains("':'") {
            return ErrorCode::E103;
        }
        if msg_lower.contains("leftparen") || msg_lower.contains("'('") || msg_lower.contains("(") {
            return ErrorCode::E104;
        }
        if msg_lower.contains("rightparen") || msg_lower.contains("')'") || msg_lower.contains(")") {
            return ErrorCode::E104;
        }
        if msg_lower.contains("leftbracket") || msg_lower.contains("'['") || msg_lower.contains("[") {
            return ErrorCode::E105;
        }
        if msg_lower.contains("rightbracket") || msg_lower.contains("']'") || msg_lower.contains("]") {
            return ErrorCode::E105;
        }
        if msg_lower.contains("leftbrace") || msg_lower.contains("'{'") || msg_lower.contains("{") {
            return ErrorCode::E106;
        }
        if msg_lower.contains("rightbrace") || msg_lower.contains("'}'") || msg_lower.contains("}") {
            return ErrorCode::E106;
        }
        return ErrorCode::E101;
    }
    if msg_lower.contains("unexpected") {
        if msg_lower.contains("rightparen") || (msg_lower.contains(")") && !msg_lower.contains("expected")) {
            return ErrorCode::E104;
        }
        if msg_lower.contains("rightbracket") || (msg_lower.contains("]") && !msg_lower.contains("expected")) {
            return ErrorCode::E105;
        }
        if msg_lower.contains("rightbrace") || (msg_lower.contains("}") && !msg_lower.contains("expected")) {
            return ErrorCode::E106;
        }
    }

    if msg_lower.contains("unexpected end") || msg_lower.contains("unexpected eof") {
        return ErrorCode::E107;
    }

    // Unmatched delimiters
    if msg_lower.contains("unmatched") {
        if msg_lower.contains("parenthes") || msg_lower.contains("(") || msg_lower.contains(")") {
            return ErrorCode::E104;
        }
        if msg_lower.contains("bracket") || msg_lower.contains("[") || msg_lower.contains("]") {
            return ErrorCode::E105;
        }
        if msg_lower.contains("brace") || msg_lower.contains("{") || msg_lower.contains("}") {
            return ErrorCode::E106;
        }
    }

    // Lexical errors
    if msg_lower.contains("unterminated string") {
        return ErrorCode::E001;
    }
    if msg_lower.contains("unterminated character") {
        return ErrorCode::E002;
    }
    if msg_lower.contains("unterminated comment") {
        return ErrorCode::E003;
    }
    if msg_lower.contains("unexpected character") || msg_lower.contains("invalid character") {
        return ErrorCode::E004;
    }
    if msg_lower.contains("invalid escape") {
        return ErrorCode::E005;
    }
    if msg_lower.contains("invalid number") || msg_lower.contains("invalid integer") || msg_lower.contains("invalid float") {
        return ErrorCode::E006;
    }
    if msg_lower.contains("indentation error") {
        return ErrorCode::E010;
    }

    // Type errors
    if msg_lower.contains("type mismatch") || msg_lower.contains("mismatched types") {
        return ErrorCode::E200;
    }
    if msg_lower.contains("undefined variable") {
        return ErrorCode::E201;
    }
    if msg_lower.contains("undefined function") {
        return ErrorCode::E202;
    }
    if msg_lower.contains("undefined class") {
        return ErrorCode::E203;
    }
    if msg_lower.contains("undefined member") {
        return ErrorCode::E204;
    }
    if msg_lower.contains("wrong number of arguments") || msg_lower.contains("expected") && msg_lower.contains("arguments") {
        return ErrorCode::E205;
    }
    if msg_lower.contains("cannot assign to immutable") || msg_lower.contains("immutable variable") {
        return ErrorCode::E207;
    }
    if msg_lower.contains("method not found") {
        return ErrorCode::E214;
    }
    if msg_lower.contains("field not found") {
        return ErrorCode::E215;
    }

    // Runtime errors
    if msg_lower.contains("division by zero") {
        return ErrorCode::E301;
    }
    if msg_lower.contains("out of bounds") || msg_lower.contains("index") {
        return ErrorCode::E302;
    }
    if msg_lower.contains("assertion failed") {
        return ErrorCode::E305;
    }
    if msg_lower.contains("measurement error") {
        return ErrorCode::E312;
    }
    if msg_lower.contains("gate application") {
        return ErrorCode::E313;
    }

    // Semantic errors
    if msg_lower.contains("break outside loop") || msg_lower.contains("break") && msg_lower.contains("loop") {
        return ErrorCode::E402;
    }
    if msg_lower.contains("continue outside loop") {
        return ErrorCode::E403;
    }
    if msg_lower.contains("return outside function") {
        return ErrorCode::E404;
    }

    // Import errors
    if msg_lower.contains("module not found") || msg_lower.contains("failed to read") {
        return ErrorCode::E500;
    }
    if msg_lower.contains("circular import") {
        return ErrorCode::E501;
    }

    // Quantum errors
    if msg_lower.contains("invalid quantum gate") || msg_lower.contains("unknown gate") {
        return ErrorCode::E600;
    }
    if msg_lower.contains("gate parameter") {
        return ErrorCode::E601;
    }
    if msg_lower.contains("register size") && msg_lower.contains("too large") {
        return ErrorCode::E604;
    }
    if msg_lower.contains("qubit index") {
        return ErrorCode::E315;
    }

    // Hardware errors
    if msg_lower.contains("backend not available") || msg_lower.contains("backend") {
        return ErrorCode::E800;
    }
    if msg_lower.contains("device not found") {
        return ErrorCode::E801;
    }

    //general error types
    if msg_lower.contains("runtime error") {
        ErrorCode::E308
    } else if msg_lower.contains("type error") {
        ErrorCode::E200
    } else if msg_lower.contains("syntax error") {
        ErrorCode::E100
    } else {
        // If we can't determine the error type, default to E100 (unexpected token)
        // since most parsing errors fall into this category
        ErrorCode::E100
    }
}

fn error_code_to_category(code: ErrorCode) -> ErrorCategory {
    match code as u32 {
        1..=10 => ErrorCategory::Syntax,      // Lexical errors (E001-E010)
        100..=199 => ErrorCategory::Syntax,   // Syntax errors
        200..=299 => ErrorCategory::Type,
        300..=399 => ErrorCategory::Runtime,
        400..=499 => ErrorCategory::Semantic,
        500..=599 => ErrorCategory::Import,
        600..=699 => ErrorCategory::Quantum,
        700..=799 => ErrorCategory::Syntax,   // Compilation
        800..=899 => ErrorCategory::Runtime,  // Hardware
        _ => ErrorCategory::Runtime,
    }
}

pub fn string_to_diagnostic(error: String) -> Diagnostic {
    let loc = parse_error_location(&error);
    let code = detect_error_code(&error);
    let category = error_code_to_category(code);
    let message = if let Some(pos) = error.find("]: ") {
        &error[pos + 3..]
    } else if let Some(pos) = error.find(": ") {
        &error[pos + 2..]
    } else {
        &error
    };

    let mut diagnostic = Diagnostic::error(loc, category, message);


    diagnostic.code = Some(format!("{}", code));


    if let Some(hint) = code.hint() {
        diagnostic.hint = Some(hint.to_string());
    }

    if let Some(suggestion) = code.suggestion() {
        diagnostic.suggestion = Some(suggestion.to_string());
    }


    diagnostic.documentation_url = Some(code.docs_url());

    diagnostic
}


pub fn report_string_error(error: String, source_code: &str, filename: &str) {
    let diagnostic = string_to_diagnostic(error);
    let reporter = ErrorReporter::new(source_code, filename);
    reporter.report(&diagnostic);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_detection() {

        assert_eq!(
            detect_error_code("Unterminated string literal at line 5"),
            ErrorCode::E001
        );

        assert_eq!(
            detect_error_code("Type Error: Undefined variable 'x'"),
            ErrorCode::E201
        );

        assert_eq!(
            detect_error_code("Runtime Error: Division by zero"),
            ErrorCode::E301
        );

        assert_eq!(
            detect_error_code("Array index 5 out of bounds"),
            ErrorCode::E302
        );

        assert_eq!(
            detect_error_code("Expected '(', but found RightParen"),
            ErrorCode::E104
        );

        assert_eq!(
            detect_error_code("Unexpected token: RightParen at line 3"),
            ErrorCode::E100
        );
    }

    #[test]
    fn test_location_parsing() {
        let loc = parse_error_location("Error at [Line 10, Col 25]: Some message");
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, 25);

        let loc2 = parse_error_location("Syntax Error at line 5, column 12: message");
        assert_eq!(loc2.line, 5);
        assert_eq!(loc2.column, 12);
    }

    #[test]
    fn test_string_to_diagnostic() {
        let error = "Type Error at [Line 5, Col 10]: Undefined variable 'foo'".to_string();
        let diag = string_to_diagnostic(error);

        assert_eq!(diag.code, Some("E201".to_string()));
        assert_eq!(diag.loc.line, 5);
        assert_eq!(diag.loc.column, 10);
        assert!(diag.message.contains("Undefined variable 'foo'"));
    }
}
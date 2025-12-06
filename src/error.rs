// src/error.rs
use crate::parser::ast::Loc;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorLevel {
    Error,
    Warning,
    Note,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: ErrorLevel,
    pub message: String,
    pub code: Option<String>, // e.g. "E001"
    pub loc: Loc,
    pub hint: Option<String>,
}

impl Diagnostic {
    pub fn error(loc: Loc, msg: &str) -> Self {
        Self {
            level: ErrorLevel::Error,
            message: msg.to_string(),
            code: None,
            loc,
            hint: None,
        }
    }

    pub fn with_code(mut self, code: &str) -> Self {
        self.code = Some(code.to_string());
        self
    }

    pub fn with_hint(mut self, hint: &str) -> Self {
        self.hint = Some(hint.to_string());
        self
    }
}

pub struct ErrorReporter<'a> {
    source_code: &'a str,
    filename: &'a str,
}

impl<'a> ErrorReporter<'a> {
    pub fn new(source_code: &'a str, filename: &'a str) -> Self {
        Self {
            source_code,
            filename,
        }
    }

    pub fn report(&self, diagnostic: &Diagnostic) {
        let color_code = match diagnostic.level {
            ErrorLevel::Error => "\x1b[31m",   // Red
            ErrorLevel::Warning => "\x1b[33m", // Yellow
            ErrorLevel::Note => "\x1b[34m",    // Blue
        };
        let reset = "\x1b[0m";
        let bold = "\x1b[1m";

        // Header: "error[E001]: Message"
        let level_str = match diagnostic.level {
            ErrorLevel::Error => "error",
            ErrorLevel::Warning => "warning",
            ErrorLevel::Note => "note",
        };

        let code_display = if let Some(c) = &diagnostic.code {
            format!("[{}]", c)
        } else {
            String::new()
        };

        println!(
            "{}{}{}{}: {}{}",
            color_code, bold, level_str, code_display, reset, diagnostic.message
        );

        // Location: "  --> src/main.qc:10:5"
        println!(
            "  {}--> {}:{}:{}{}",
            "\x1b[34m", // Blue arrow
            self.filename,
            diagnostic.loc.line,
            diagnostic.loc.column,
            reset
        );

        // Code Snippet
        if let Some(line_content) = self.get_line_content(diagnostic.loc.line) {
            let line_num_padding = "     ";
            let line_num = diagnostic.loc.line.to_string();

            // Empty pipe line
            println!("{} |", line_num_padding);

            // The actual code
            println!(" {:<4} | {}", line_num, line_content);

            // The pointer line (^^^^^)
            print!("{} |", line_num_padding);

            // Calculate spaces to the column
            // (Note: column is 1-based, we treat it carefully)
            let spaces = if diagnostic.loc.column > 0 { diagnostic.loc.column - 1 } else { 0 };
            let pointer_str = " ".repeat(spaces);

            println!(" {}{}^", pointer_str, color_code);

            // Print hint if it exists
            if let Some(hint) = &diagnostic.hint {
                println!(
                    "{} = hint: {}{}",
                    " ".repeat(line_num_padding.len() + 3 + spaces),
                    hint,
                    reset
                );
            }

            // Reset color
            println!("{}", reset);
        }
    }

    fn get_line_content(&self, line_number: usize) -> Option<&str> {
        self.source_code.lines().nth(line_number - 1)
    }
}
// src/error.rs
use crate::parser::ast::Loc;
use std::fmt;
use colored::Colorize;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorLevel {
    Error,
    Warning,
    Note,
    Help,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorCategory {
    Syntax,
    Type,
    Runtime,
    Semantic,
    Import,
    Quantum,
    Memory,
    Performance,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: ErrorLevel,
    pub category: ErrorCategory,
    pub message: String,
    pub code: Option<String>,
    pub loc: Loc,
    pub hint: Option<String>,
    pub suggestion: Option<String>,
    pub related_info: Vec<(Loc, String)>,
    pub documentation_url: Option<String>,
}

impl Diagnostic {
    pub fn error(loc: Loc, category: ErrorCategory, msg: &str) -> Self {
        Self {
            level: ErrorLevel::Error,
            category,
            message: msg.to_string(),
            code: None,
            loc,
            hint: None,
            suggestion: None,
            related_info: Vec::new(),
            documentation_url: None,
        }
    }

    pub fn warning(loc: Loc, category: ErrorCategory, msg: &str) -> Self {
        Self {
            level: ErrorLevel::Warning,
            category,
            message: msg.to_string(),
            code: None,
            loc,
            hint: None,
            suggestion: None,
            related_info: Vec::new(),
            documentation_url: None,
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

    pub fn with_suggestion(mut self, suggestion: &str) -> Self {
        self.suggestion = Some(suggestion.to_string());
        self
    }

    pub fn with_related(mut self, loc: Loc, msg: &str) -> Self {
        self.related_info.push((loc, msg.to_string()));
        self
    }

    pub fn with_docs(mut self, url: &str) -> Self {
        self.documentation_url = Some(url.to_string());
        self
    }
}

pub struct ErrorReporter<'a> {
    source_code: &'a str,
    filename: &'a str,
    show_colors: bool,
}

impl<'a> ErrorReporter<'a> {
    pub fn new(source_code: &'a str, filename: &'a str) -> Self {
        Self {
            source_code,
            filename,
            show_colors: true,
        }
    }

    pub fn report(&self, diagnostic: &Diagnostic) {
        self.print_header(diagnostic);
        self.print_location(diagnostic);
        self.print_source_snippet(diagnostic);
        self.print_hint(diagnostic);
        self.print_suggestion(diagnostic);
        self.print_related_info(diagnostic);
        self.print_documentation(diagnostic);
        println!();
    }

    fn print_header(&self, diagnostic: &Diagnostic) {
        let (level_str, color) = match diagnostic.level {
            ErrorLevel::Error => ("error", "red"),
            ErrorLevel::Warning => ("warning", "yellow"),
            ErrorLevel::Note => ("note", "blue"),
            ErrorLevel::Help => ("help", "green"),
        };

        let category_str = format!("{:?}", diagnostic.category).to_lowercase();

        let code_display = if let Some(c) = &diagnostic.code {
            format!("[{}]", c)
        } else {
            String::new()
        };

        if self.show_colors {
            println!(
                "{}{}{}: {}",
                level_str.color(color).bold(),
                code_display.color(color),
                format!("({})", category_str).dimmed(),
                diagnostic.message.bold()
            );
        } else {
            println!("{}{}: {}", level_str, code_display, diagnostic.message);
        }
    }

    fn print_location(&self, diagnostic: &Diagnostic) {
        if self.show_colors {
            println!(
                "   {} {}:{}:{}",
                "-->".blue().bold(),
                self.filename,
                diagnostic.loc.line,
                diagnostic.loc.column
            );
        } else {
            println!(
                "   --> {}:{}:{}",
                self.filename,
                diagnostic.loc.line,
                diagnostic.loc.column
            );
        }
    }

    fn print_source_snippet(&self, diagnostic: &Diagnostic) {
        let line_num = diagnostic.loc.line;
        let col = diagnostic.loc.column;

        // Get surrounding lines for context
        let lines: Vec<&str> = self.source_code.lines().collect();
        let start_line = if line_num > 2 { line_num - 2 } else { 1 };
        let end_line = std::cmp::min(line_num + 2, lines.len());

        let line_num_width = end_line.to_string().len();

        // Print separator
        if self.show_colors {
            println!("      {}", "|".blue());
        } else {
            println!("      |");
        }

        // Print context lines
        for i in start_line..=end_line {
            if i > lines.len() {
                break;
            }

            let line_content = lines[i - 1];
            let is_error_line = i == line_num;

            if self.show_colors {
                if is_error_line {
                    println!(
                        " {:width$} {} {}",
                        i.to_string().blue().bold(),
                        "|".blue(),
                        line_content,
                        width = line_num_width
                    );
                } else {
                    println!(
                        " {:width$} {} {}",
                        i.to_string().blue(),
                        "|".blue(),
                        line_content.dimmed(),
                        width = line_num_width
                    );
                }
            } else {
                println!(" {:width$} | {}", i, line_content, width = line_num_width);
            }

            // Print error pointer for the error line
            if is_error_line {
                let pointer_padding = " ".repeat(line_num_width + 1);
                let spaces = " ".repeat(if col > 0 { col - 1 } else { 0 });

                if self.show_colors {
                    println!(
                        "{} {} {}{}",
                        pointer_padding,
                        "|".blue(),
                        spaces,
                        "^".repeat(std::cmp::min(
                            line_content.len().saturating_sub(col.saturating_sub(1)),
                            1
                        )).red().bold()
                    );
                } else {
                    println!("{} | {}^", pointer_padding, spaces);
                }
            }
        }

        // Print separator
        if self.show_colors {
            println!("      {}", "|".blue());
        } else {
            println!("      |");
        }
    }

    fn print_hint(&self, diagnostic: &Diagnostic) {
        if let Some(hint) = &diagnostic.hint {
            if self.show_colors {
                println!(
                    "      {} {}",
                    "=".blue().bold(),
                    format!("hint: {}", hint).cyan()
                );
            } else {
                println!("      = hint: {}", hint);
            }
        }
    }

    fn print_suggestion(&self, diagnostic: &Diagnostic) {
        if let Some(suggestion) = &diagnostic.suggestion {
            if self.show_colors {
                println!(
                    "      {} {}",
                    "=".blue().bold(),
                    format!("suggestion: {}", suggestion).green().bold()
                );
            } else {
                println!("      = suggestion: {}", suggestion);
            }
        }
    }

    fn print_related_info(&self, diagnostic: &Diagnostic) {
        if !diagnostic.related_info.is_empty() {
            if self.show_colors {
                println!("\n      {} Related information:", "=".blue().bold());
            } else {
                println!("\n      = Related information:");
            }

            for (loc, msg) in &diagnostic.related_info {
                if self.show_colors {
                    println!(
                        "        {} {}:{}:{}: {}",
                        "-->".blue(),
                        self.filename,
                        loc.line,
                        loc.column,
                        msg.dimmed()
                    );
                } else {
                    println!(
                        "        --> {}:{}:{}: {}",
                        self.filename,
                        loc.line,
                        loc.column,
                        msg
                    );
                }
            }
        }
    }

    fn print_documentation(&self, diagnostic: &Diagnostic) {
        if let Some(url) = &diagnostic.documentation_url {
            if self.show_colors {
                println!(
                    "\n      {} For more information, see: {}",
                    "=".blue().bold(),
                    url.cyan().underline()
                );
            } else {
                println!("\n      = For more information, see: {}", url);
            }
        }
    }
}

// Fuzzy string matching for suggestions
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();
    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if s1.chars().nth(i - 1) == s2.chars().nth(j - 1) {
                0
            } else {
                1
            };
            matrix[i][j] = std::cmp::min(
                std::cmp::min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1),
                matrix[i - 1][j - 1] + cost,
            );
        }
    }

    matrix[len1][len2]
}

pub fn find_similar_names(target: &str, candidates: &[String]) -> Vec<String> {
    let mut scored: Vec<(String, usize)> = candidates
        .iter()
        .map(|c| (c.clone(), levenshtein_distance(target, c)))
        .filter(|(_, dist)| *dist <= 3) // Only suggest if within 3 edits
        .collect();

    scored.sort_by_key(|(_, dist)| *dist);
    scored.into_iter().take(3).map(|(name, _)| name).collect()
}
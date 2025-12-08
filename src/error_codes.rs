// src/error_codes.rs
//! Quantica Error Code System
//!
//! Error codes are organized by category:
//! - E001-E099: Lexical Analysis Errors
//! - E100-E199: Syntax/Parser Errors
//! - E200-E299: Type Checking Errors
//! - E300-E399: Runtime Errors
//! - E400-E499: Semantic Errors
//! - E500-E599: Import/Module Errors
//! - E600-E699: Quantum-Specific Errors
//! - E700-E799: Compilation/Codegen Errors
//! - E800-E899: Hardware Integration Errors

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ErrorCode {
    // === LEXICAL ERRORS (E001-E099) ===
    E001 = 1, // Unterminated string
    E002 = 2, // Unterminated character literal
    E003 = 3, // Unterminated comment
    E004 = 4, // Invalid character
    E005 = 5, // Invalid escape sequence
    E006 = 6, // Invalid number literal
    E007 = 7, // Invalid unicode escape
    E008 = 8, // Unexpected end of file in lexer
    E009 = 9, // Invalid quantum ket/bra notation
    E010 = 10, // Indentation error

    // === SYNTAX ERRORS (E100-E199) ===
    E100 = 100, // Unexpected token
    E101 = 101, // Expected token not found
    E102 = 102, // Missing semicolon
    E103 = 103, // Missing colon
    E104 = 104, // Unmatched parenthesis
    E105 = 105, // Unmatched bracket
    E106 = 106, // Unmatched brace
    E107 = 107, // Unexpected end of file
    E108 = 108, // Invalid expression
    E109 = 109, // Invalid statement
    E110 = 110, // Invalid function declaration
    E111 = 111, // Invalid class declaration
    E112 = 112, // Invalid circuit declaration
    E113 = 113, // Invalid parameter list
    E114 = 114, // Invalid argument list
    E115 = 115, // Invalid block structure
    E116 = 116, // Invalid quantum declaration
    E117 = 117, // Invalid gate expression
    E118 = 118, // Invalid apply statement
    E119 = 119, // Malformed control flow
    E120 = 120, // Invalid assignment target

    // === TYPE ERRORS (E200-E299) ===
    E200 = 200, // Type mismatch
    E201 = 201, // Undefined variable
    E202 = 202, // Undefined function
    E203 = 203, // Undefined class
    E204 = 204, // Undefined member
    E205 = 205, // Wrong number of arguments
    E206 = 206, // Incompatible types for operator
    E207 = 207, // Cannot assign to immutable variable
    E208 = 208, // Return type mismatch
    E209 = 209, // Array type mismatch
    E210 = 210, // Invalid array index type
    E211 = 211, // Invalid quantum register size
    E212 = 212, // Quantum type mismatch
    E213 = 213, // Cannot call non-function
    E214 = 214, // Method not found
    E215 = 215, // Field not found
    E216 = 216, // Invalid member access
    E217 = 217, // Type annotation mismatch
    E218 = 218, // Cannot infer type
    E219 = 219, // Circular type dependency
    E220 = 220, // Invalid type for operation

    // === RUNTIME ERRORS (E300-E399) ===
    E300 = 300, // Null pointer dereference
    E301 = 301, // Division by zero
    E302 = 302, // Array index out of bounds
    E303 = 303, // Stack overflow
    E304 = 304, // Out of memory
    E305 = 305, // Assertion failed
    E306 = 306, // Invalid cast
    E307 = 307, // Uninitialized variable
    E308 = 308, // Invalid operation
    E309 = 309, // Resource not available
    E310 = 310, // Timeout error
    E311 = 311, // Invalid state
    E312 = 312, // Measurement error
    E313 = 313, // Gate application error
    E314 = 314, // Register size mismatch
    E315 = 315, // Qubit index out of bounds

    // === SEMANTIC ERRORS (E400-E499) ===
    E400 = 400, // Duplicate declaration
    E401 = 401, // Invalid scope
    E402 = 402, // Break outside loop
    E403 = 403, // Continue outside loop
    E404 = 404, // Return outside function
    E405 = 405, // Invalid use of 'self'
    E406 = 406, // Invalid superclass
    E407 = 407, // Cyclic inheritance
    E408 = 408, // Abstract method not implemented
    E409 = 409, // Invalid override
    E410 = 410, // Dead code detected
    E411 = 411, // Unreachable code
    E412 = 412, // Unused variable
    E413 = 413, // Unused function
    E414 = 414, // Invalid constant expression
    E415 = 415, // Mutability violation

    // === IMPORT/MODULE ERRORS (E500-E599) ===
    E500 = 500, // Module not found
    E501 = 501, // Circular import
    E502 = 502, // Invalid import path
    E503 = 503, // Symbol not found in module
    E504 = 504, // Conflicting imports
    E505 = 505, // Module parse error
    E506 = 506, // Module type error
    E507 = 507, // Invalid module structure
    E508 = 508, // Package not found
    E509 = 509, // Version conflict

    // === QUANTUM ERRORS (E600-E699) ===
    E600 = 600, // Invalid quantum gate
    E601 = 601, // Gate parameter error
    E602 = 602, // Incompatible qubit operation
    E603 = 603, // Invalid control configuration
    E604 = 604, // Quantum register too large
    E605 = 605, // Invalid initial state
    E606 = 606, // Measurement on uninitialized qubit
    E607 = 607, // Invalid dagger operation
    E608 = 608, // Gate arity mismatch
    E609 = 609, // Invalid tensor product
    E610 = 610, // Quantum state normalization error
    E611 = 611, // Invalid quantum circuit
    E612 = 612, // Superposition error
    E613 = 613, // Entanglement error
    E614 = 614, // Decoherence detected

    // === COMPILATION ERRORS (E700-E799) ===
    E700 = 700, // LLVM error
    E701 = 701, // Code generation failed
    E702 = 702, // Optimization failed
    E703 = 703, // Linking error
    E704 = 704, // Invalid target
    E705 = 705, // Unsupported feature
    E706 = 706, // Assembly error
    E707 = 707, // Object file error
    E708 = 708, // Debug info error
    E709 = 709, // ABI mismatch
    E710 = 710, // Platform not supported

    // === HARDWARE INTEGRATION ERRORS (E800-E899) ===
    E800 = 800, // Backend not available
    E801 = 801, // Device not found
    E802 = 802, // Connection error
    E803 = 803, // Authentication failed
    E804 = 804, // Job submission failed
    E805 = 805, // Job execution error
    E806 = 806, // Result retrieval failed
    E807 = 807, // Unsupported gate on hardware
    E808 = 808, // Hardware timeout
    E809 = 809, // Calibration error
    E810 = 810, // Noise model error
}

impl ErrorCode {
    /// Get the category name for this error code
    pub fn category(&self) -> &'static str {
        match *self as u32 {
            1..=99 => "Lexical",
            100..=199 => "Syntax",
            200..=299 => "Type",
            300..=399 => "Runtime",
            400..=499 => "Semantic",
            500..=599 => "Import",
            600..=699 => "Quantum",
            700..=799 => "Compilation",
            800..=899 => "Hardware",
            _ => "Unknown",
        }
    }

    /// Get a short description of this error
    pub fn description(&self) -> &'static str {
        match self {
            // Lexical
            ErrorCode::E001 => "Unterminated string literal",
            ErrorCode::E002 => "Unterminated character literal",
            ErrorCode::E003 => "Unterminated comment",
            ErrorCode::E004 => "Invalid or unexpected character",
            ErrorCode::E005 => "Invalid escape sequence",
            ErrorCode::E006 => "Invalid number literal",
            ErrorCode::E007 => "Invalid unicode escape",
            ErrorCode::E008 => "Unexpected end of input",
            ErrorCode::E009 => "Invalid quantum notation",
            ErrorCode::E010 => "Indentation error",

            // Syntax
            ErrorCode::E100 => "Unexpected token",
            ErrorCode::E101 => "Expected token not found",
            ErrorCode::E102 => "Missing semicolon",
            ErrorCode::E103 => "Missing colon",
            ErrorCode::E104 => "Unmatched parenthesis",
            ErrorCode::E105 => "Unmatched bracket",
            ErrorCode::E106 => "Unmatched brace",
            ErrorCode::E107 => "Unexpected end of file",
            ErrorCode::E108 => "Invalid expression syntax",
            ErrorCode::E109 => "Invalid statement syntax",
            ErrorCode::E110 => "Invalid function declaration",
            ErrorCode::E111 => "Invalid class declaration",
            ErrorCode::E112 => "Invalid circuit declaration",
            ErrorCode::E113 => "Invalid parameter list",
            ErrorCode::E114 => "Invalid argument list",
            ErrorCode::E115 => "Invalid block structure",
            ErrorCode::E116 => "Invalid quantum declaration",
            ErrorCode::E117 => "Invalid gate expression",
            ErrorCode::E118 => "Invalid apply statement",
            ErrorCode::E119 => "Malformed control flow statement",
            ErrorCode::E120 => "Invalid assignment target",

            // Type
            ErrorCode::E200 => "Type mismatch",
            ErrorCode::E201 => "Undefined variable",
            ErrorCode::E202 => "Undefined function",
            ErrorCode::E203 => "Undefined class",
            ErrorCode::E204 => "Undefined member",
            ErrorCode::E205 => "Wrong number of arguments",
            ErrorCode::E206 => "Incompatible types for operator",
            ErrorCode::E207 => "Cannot assign to immutable variable",
            ErrorCode::E208 => "Return type mismatch",
            ErrorCode::E209 => "Array element type mismatch",
            ErrorCode::E210 => "Invalid array index type",
            ErrorCode::E211 => "Invalid quantum register size",
            ErrorCode::E212 => "Quantum type mismatch",
            ErrorCode::E213 => "Cannot call non-function type",
            ErrorCode::E214 => "Method not found on type",
            ErrorCode::E215 => "Field not found on type",
            ErrorCode::E216 => "Invalid member access",
            ErrorCode::E217 => "Type annotation does not match inferred type",
            ErrorCode::E218 => "Cannot infer type",
            ErrorCode::E219 => "Circular type dependency",
            ErrorCode::E220 => "Invalid type for this operation",

            // Runtime
            ErrorCode::E300 => "Null pointer dereference",
            ErrorCode::E301 => "Division by zero",
            ErrorCode::E302 => "Array index out of bounds",
            ErrorCode::E303 => "Stack overflow",
            ErrorCode::E304 => "Out of memory",
            ErrorCode::E305 => "Assertion failed",
            ErrorCode::E306 => "Invalid type cast",
            ErrorCode::E307 => "Uninitialized variable access",
            ErrorCode::E308 => "Invalid operation at runtime",
            ErrorCode::E309 => "Required resource not available",
            ErrorCode::E310 => "Operation timeout",
            ErrorCode::E311 => "Invalid program state",
            ErrorCode::E312 => "Quantum measurement error",
            ErrorCode::E313 => "Gate application error",
            ErrorCode::E314 => "Quantum register size mismatch",
            ErrorCode::E315 => "Qubit index out of bounds",

            // Semantic
            ErrorCode::E400 => "Duplicate declaration",
            ErrorCode::E401 => "Invalid scope access",
            ErrorCode::E402 => "Break statement outside loop",
            ErrorCode::E403 => "Continue statement outside loop",
            ErrorCode::E404 => "Return statement outside function",
            ErrorCode::E405 => "Invalid use of 'self' keyword",
            ErrorCode::E406 => "Invalid superclass",
            ErrorCode::E407 => "Cyclic inheritance detected",
            ErrorCode::E408 => "Abstract method not implemented",
            ErrorCode::E409 => "Invalid method override",
            ErrorCode::E410 => "Dead code detected",
            ErrorCode::E411 => "Unreachable code",
            ErrorCode::E412 => "Unused variable",
            ErrorCode::E413 => "Unused function",
            ErrorCode::E414 => "Invalid constant expression",
            ErrorCode::E415 => "Mutability violation",

            // Import
            ErrorCode::E500 => "Module or package not found",
            ErrorCode::E501 => "Circular import detected",
            ErrorCode::E502 => "Invalid import path",
            ErrorCode::E503 => "Symbol not found in module",
            ErrorCode::E504 => "Conflicting import declarations",
            ErrorCode::E505 => "Error parsing imported module",
            ErrorCode::E506 => "Type error in imported module",
            ErrorCode::E507 => "Invalid module structure",
            ErrorCode::E508 => "Package not found",
            ErrorCode::E509 => "Package version conflict",

            // Quantum
            ErrorCode::E600 => "Unknown or invalid quantum gate",
            ErrorCode::E601 => "Invalid gate parameters",
            ErrorCode::E602 => "Incompatible qubit operation",
            ErrorCode::E603 => "Invalid control qubit configuration",
            ErrorCode::E604 => "Quantum register size exceeds limit",
            ErrorCode::E605 => "Invalid initial quantum state",
            ErrorCode::E606 => "Measurement on uninitialized qubit",
            ErrorCode::E607 => "Invalid dagger/adjoint operation",
            ErrorCode::E608 => "Gate arity mismatch",
            ErrorCode::E609 => "Invalid tensor product operation",
            ErrorCode::E610 => "Quantum state normalization error",
            ErrorCode::E611 => "Invalid quantum circuit structure",
            ErrorCode::E612 => "Superposition calculation error",
            ErrorCode::E613 => "Entanglement operation error",
            ErrorCode::E614 => "Quantum decoherence detected",

            // Compilation
            ErrorCode::E700 => "LLVM internal error",
            ErrorCode::E701 => "Code generation failed",
            ErrorCode::E702 => "Optimization pass failed",
            ErrorCode::E703 => "Linking error",
            ErrorCode::E704 => "Invalid compilation target",
            ErrorCode::E705 => "Feature not supported on this target",
            ErrorCode::E706 => "Assembly generation error",
            ErrorCode::E707 => "Object file generation error",
            ErrorCode::E708 => "Debug information error",
            ErrorCode::E709 => "ABI compatibility error",
            ErrorCode::E710 => "Platform not supported",

            // Hardware
            ErrorCode::E800 => "Quantum backend not available",
            ErrorCode::E801 => "Quantum device not found",
            ErrorCode::E802 => "Connection to quantum hardware failed",
            ErrorCode::E803 => "Authentication failed",
            ErrorCode::E804 => "Job submission to hardware failed",
            ErrorCode::E805 => "Job execution error on hardware",
            ErrorCode::E806 => "Failed to retrieve results",
            ErrorCode::E807 => "Gate not supported on this hardware",
            ErrorCode::E808 => "Hardware operation timeout",
            ErrorCode::E809 => "Calibration error",
            ErrorCode::E810 => "Noise model error",
        }
    }

    /// Get a hint for resolving this error
    pub fn hint(&self) -> Option<&'static str> {
        match self {
            ErrorCode::E001 => Some("Make sure all string literals are properly closed with matching quotes"),
            ErrorCode::E003 => Some("Multi-line comments must be closed with */"),
            ErrorCode::E010 => Some("Check that all lines in a block have consistent indentation"),
            ErrorCode::E101 => Some("Check for missing colons after function/class declarations, or missing parentheses"),
            ErrorCode::E104 => Some("Check that all opening parentheses '(' have matching closing parentheses ')'"),
            ErrorCode::E107 => Some("You may have an unclosed block, missing return statement, or incomplete expression"),
            ErrorCode::E201 => Some("Make sure the variable is declared before use with 'let' or 'mut'"),
            ErrorCode::E205 => Some("Check the function signature to see how many parameters it expects"),
            ErrorCode::E207 => Some("Variables declared with 'let' are immutable. Use 'mut' to make them mutable"),
            ErrorCode::E302 => Some("Add a length check before accessing array elements: 'if index < len(array)'"),
            ErrorCode::E301 => Some("Add a check for zero before division: 'if denominator != 0'"),
            ErrorCode::E305 => Some("Review the assertion condition and the values being tested"),
            ErrorCode::E402 => Some("Break statements can only be used inside for/while loops"),
            ErrorCode::E500 => Some("Check the file path and make sure the module exists"),
            ErrorCode::E604 => Some("Try using a smaller number of qubits or optimize your circuit"),
            ErrorCode::E800 => Some("Make sure the required quantum backend (qiskit, cirq) is installed"),
            _ => None,
        }
    }

    /// Get a suggestion for fixing this error
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            ErrorCode::E001 => Some("Add a closing quote: \"your string\""),
            ErrorCode::E101 => Some("Add the required colon ':' after the declaration"),
            ErrorCode::E104 => Some("Add the missing ')' or remove the extra ')'"),
            ErrorCode::E201 => Some("Declare the variable first: 'let variable_name = value'"),
            ErrorCode::E207 => Some("Change 'let' to 'mut': 'mut variable_name = value'"),
            ErrorCode::E200 => Some("Use type conversion functions: to_int(), to_float(), to_string()"),
            ErrorCode::E205 => Some("Adjust the number of arguments to match the function signature"),
            ErrorCode::E302 => Some("Check array bounds: 'if index < len(array): value = array[index]'"),
            ErrorCode::E301 => Some("Add a conditional: 'if denominator != 0: result = numerator / denominator'"),
            ErrorCode::E402 => Some("Move the break statement inside a loop, or remove it"),
            ErrorCode::E500 => Some("Check the import path or install the required package"),
            ErrorCode::E604 => Some("Reduce the number of qubits or use a hardware backend"),
            _ => None,
        }
    }

    /// Get documentation URL for this error
    pub fn docs_url(&self) -> String {
        format!("https://quantica-foundation.github.io/quantica-lang/errorcodes.html")
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{:03}", *self as u32)
    }
}

/// Helper function to create common error patterns
pub mod patterns {
    use super::ErrorCode;

    pub fn undefined_variable(name: &str, similar: &[String]) -> (ErrorCode, String, Option<String>) {
        let mut msg = format!("Undefined variable '{}'", name);
        let suggestion = if !similar.is_empty() {
            let sug = format!("Did you mean: {}?", similar.join(", "));
            msg.push_str(&format!(". {}", sug));
            Some(format!("Check the spelling or declare it with: let {} = ...", name))
        } else {
            Some(format!("Declare the variable first: let {} = value", name))
        };

        (ErrorCode::E201, msg, suggestion)
    }

    pub fn type_mismatch(expected: &str, got: &str, context: &str) -> (ErrorCode, String) {
        (
            ErrorCode::E200,
            format!("Type mismatch in {}: expected {}, but got {}", context, expected, got)
        )
    }

    pub fn wrong_arg_count(func: &str, expected: usize, got: usize) -> (ErrorCode, String) {
        (
            ErrorCode::E205,
            format!("Function '{}' expects {} arguments, but got {}", func, expected, got)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_display() {
        assert_eq!(format!("{}", ErrorCode::E001), "E001");
        assert_eq!(format!("{}", ErrorCode::E201), "E201");
        assert_eq!(format!("{}", ErrorCode::E600), "E600");
    }

    #[test]
    fn test_error_code_category() {
        assert_eq!(ErrorCode::E001.category(), "Lexical");
        assert_eq!(ErrorCode::E201.category(), "Type");
        assert_eq!(ErrorCode::E600.category(), "Quantum");
    }
}
// src/doc_generator.rs

use crate::parser::ast::{ASTNode, Parameter, Type};
use std::fs;
use std::io::Write;
use std::path::Path;

/// A struct responsible for generating documentation from an AST.
pub struct DocGenerator {
    markdown_buffer: String,
}

impl DocGenerator {
    /// Creates a new DocGenerator.
    pub fn new() -> Self {
        DocGenerator {
            markdown_buffer: String::new(),
        }
    }

    /// Public entry point.
    /// Generates documentation for the given program and saves it to a file.
    pub fn run(ast: &ASTNode, output_file: &str) -> Result<(), String> {
        let mut generator = Self::new();
        generator.walk_program(ast)?;
        
        // Ensure the output directory (e.g., "docs/") exists
        if let Some(parent_dir) = Path::new(output_file).parent() {
            if !parent_dir.exists() {
                fs::create_dir_all(parent_dir)
                    .map_err(|e| format!("Failed to create docs directory: {}", e))?;
            }
        }
        
        // Write the buffer to the file
        let mut file = fs::File::create(output_file)
            .map_err(|e| format!("Failed to create doc file: {}", e))?;
        
        file.write_all(generator.markdown_buffer.as_bytes())
            .map_err(|e| format!("Failed to write to doc file: {}", e))?;
            
        Ok(())
    }

    /// Walks the main program and generates docs for top-level statements.
    fn walk_program(&mut self, node: &ASTNode) -> Result<(), String> {
        if let ASTNode::Program(statements) = node {
            // Add a title to the document
            self.markdown_buffer.push_str("# Quantica API Reference\n\n");
            
            for stmt in statements {
                self.generate_doc_for_node(stmt);
            }
            Ok(())
        } else {
            Err("Expected Program node".to_string())
        }
    }

    /// The core logic. Checks a node for a doc comment and formats it.
    fn generate_doc_for_node(&mut self, node: &ASTNode) {
        match node {
            ASTNode::FunctionDeclaration { doc_comment, name, parameters, return_type, .. } => {
                if let Some(comment) = doc_comment {
                    self.add_entry(
                        &format!("func {}", name),
                        &self.format_params(parameters),
                        return_type,
                        comment,
                    );
                }
            }
            ASTNode::CircuitDeclaration { doc_comment, name, parameters, return_type, .. } => {
                if let Some(comment) = doc_comment {
                    self.add_entry(
                        &format!("circuit {}", name),
                        &self.format_params(parameters),
                        return_type,
                        comment,
                    );
                }
            }
            ASTNode::LetDeclaration { doc_comment, name, type_annotation, is_mutable, .. } => {
                if let Some(comment) = doc_comment {
                    let prefix = if *is_mutable { "mut" } else { "let" };
                    self.add_variable_entry(
                        &format!("{} {}", prefix, name),
                        type_annotation,
                        comment,
                    );
                }
            }
            // We can add other documented nodes here, like `Class`, `Module`, etc.
            _ => {
                // Not a documented node, just ignore it.
            }
        }
    }

    // --- Helper Functions ---

    /// Adds a formatted function/circuit entry to the Markdown buffer.
    fn add_entry(
        &mut self,
        name: &str,
        params_str: &str,
        return_type: &Option<Type>,
        comment: &str,
    ) {
        // 1. Add the code signature
        self.markdown_buffer.push_str(&format!(
            "### `{}({}){}`\n\n",
            name,
            params_str,
            self.format_return_type(return_type)
        ));
        
        // 2. Add the documentation
        self.markdown_buffer.push_str(comment);
        self.markdown_buffer.push_str("\n\n---\n\n"); // Add horizontal rule
    }
    
    /// Adds a formatted variable entry to the Markdown buffer.
    fn add_variable_entry(
        &mut self,
        name: &str,
        type_annotation: &Option<Type>,
        comment: &str,
    ) {
        // 1. Add the code signature
        let type_str = if let Some(t) = type_annotation {
            format!(": {}", self.format_type(t))
        } else {
            "".to_string()
        };
        self.markdown_buffer.push_str(&format!(
            "### `{}`\n\n",
            format!("{}{}", name, type_str)
        ));
        
        // 2. Add the documentation
        self.markdown_buffer.push_str(comment);
        self.markdown_buffer.push_str("\n\n---\n\n"); // Add horizontal rule
    }

    /// Formats a list of parameters into a string: "p1: int, p2: float"
    fn format_params(&self, params: &[Parameter]) -> String {
        params.iter()
            .map(|p| format!("{}: {}", p.name, self.format_type(&p.param_type)))
            .collect::<Vec<String>>()
            .join(", ")
    }

    /// Formats a return type into a string: " -> int"
    fn format_return_type(&self, return_type: &Option<Type>) -> String {
        if let Some(t) = return_type {
            format!(" -> {}", self.format_type(t))
        } else {
            "".to_string()
        }
    }
    
    /// Formats a Type enum into a readable string.
    fn format_type(&self, t: &Type) -> String {
        // This can be as simple or complex as you need
        format!("{:?}", t)
    }
}
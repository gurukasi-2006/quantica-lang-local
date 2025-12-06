// src/lib.rs
// This is the core library of your language

pub mod codegen;
pub mod doc_generator;
pub mod environment;
pub mod evaluator;
pub mod hardware_integration;
pub mod lexer;
pub mod parser;
pub mod quantum_backend;
pub mod runtime;
pub mod type_checker;
pub use runtime::{
    quantica_rt_apply_gate, quantica_rt_debug_state, quantica_rt_measure, quantica_rt_new_state,
};
pub mod linker;
pub mod graphics;
pub mod graphics_runtime;

pub use graphics_runtime::*;
pub mod error;
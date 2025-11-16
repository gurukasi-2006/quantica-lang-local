mod lexer;

use lexer::Lexer;
use std::env;
use std::fs;

fn main() {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        println!("Usage: quantica <filename.qc>");
        println!("   or: quantica --test");
        return;
    }
    
    if args[1] == "--test" {
        run_test_examples();
        return;
    }
    
    // Read file
    let filename = &args[1];
    let source = fs::read_to_string(filename)
        .expect(&format!("Failed to read file: {}", filename));
    
    // Tokenize
    let mut lexer = Lexer::new(&source);
    
    match lexer.tokenize() {
        Ok(tokens) => {
            println!("✓ Lexer succeeded! Found {} tokens\n", tokens.len());
            
            for (i, token) in tokens.iter().enumerate() {
                println!("{:4}: {:?}", i, token.token);
            }
        }
        Err(e) => {
            eprintln!("✗ Lexer error: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_test_examples() {
    println!("=== Running Test Examples ===\n");
    
    // Test 1: Basic quantum program
    println!("Test 1: Quantum Program");
    test_source(r#"
quantum q = |0}
apply Hadamard(q)
let result = measure(q)
"#);
    
    // Test 2: Classical program
    println!("\nTest 2: Classical Program");
    test_source(r#"
let x = 42
mut y = 3.14
for i in 0..10:
    print(i)
"#);
    
    // Test 3: Tensor product
    println!("\nTest 3: Tensor Product");
    test_source(r#"
quantum q1 = |0}
quantum q2 = |1}
quantum entangled = q1 *** q2
"#);
    
    // Test 4: Function definition
    println!("\nTest 4: Function");
    test_source(r#"
func add(a: int, b: int) -> int:
    return a + b
"#);
    
    // Test 5: All quantum states
    println!("\nTest 5: Quantum States");
    test_source(r#"
quantum q0 = |0}
quantum q1 = |1}
quantum qp = |+}
quantum qm = |-}
quantum psi = |psi}
"#);
}

fn test_source(source: &str) {
    let mut lexer = Lexer::new(source);
    
    match lexer.tokenize() {
        Ok(tokens) => {
            println!("✓ Success! {} tokens", tokens.len());
            for token in tokens.iter().filter(|t| !matches!(t.token, 
                lexer::token::Token::Newline | 
                lexer::token::Token::Indent | 
                lexer::token::Token::Dedent | 
                lexer::token::Token::Eof)) {
                println!("  {:?}", token.token);
            }
        }
        Err(e) => {
            println!("✗ Error: {}", e);
        }
    }
}
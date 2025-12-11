/* src/environment/mod.rs */
use crate::parser::ast::Parameter;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use crate::parser::ast::ClassMethod;
use crate::parser::ast::ClassField;
use crate::parser::ast::ASTNode;

#[derive(Debug, Clone)]
pub struct GateDefinition {

    pub name: String,

    pub params: Vec<f64>,

    pub controls: Vec<usize>,

    pub targets: Vec<usize>,

    pub register_size: usize,

    pub state_rc: Rc<RefCell<HashMap<usize, (f64, f64)>>>,
}

#[derive(Debug, Clone)]
pub enum RuntimeValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    None,



    Qubit {

        state: Rc<RefCell<HashMap<usize, (f64, f64)>>>,
        index: usize,
        size: usize,
        register_name: String,
        global_index: Option<usize>,
    },


    QuantumRegister {
        size: usize,

        state: Rc<RefCell<HashMap<usize, (f64, f64)>>>,
        register_name: String,
        global_start_index: Option<usize>,
    },

    Gate {

        base_name: String,
        is_dagger: bool,

        num_controls: usize,
    },
    Class {
        name: String,
        superclass: Option<String>,
        fields: Vec<ClassField>,
        methods: HashMap<String, ClassMethod>,
        constructor: Option<Box<ASTNode>>,
    },


    Instance {
        class_name: String,
        fields: HashMap<String, Rc<RefCell<RuntimeValue>>>,
        methods: HashMap<String, ClassMethod>,
    },

    Register(Vec<Rc<RefCell<RuntimeValue>>>),
    Dict(HashMap<String, Rc<RefCell<RuntimeValue>>>),
    Range(Vec<i64>),
    KetState(String),


    Function {
        parameters: Vec<Parameter>,
        body: Box<crate::parser::ast::ASTNode>,
        env: Rc<RefCell<Environment>>,
    },


    BuiltinFunction(String),
    Module(Rc<RefCell<Environment>>),

    ReturnValue(Box<RuntimeValue>),
    Break,
    Continue,

    Probabilistic {
        value: Box<RuntimeValue>,
        confidence: f64,
    },
}

impl RuntimeValue {
    pub fn type_name(&self) -> &str {
        match self {
            RuntimeValue::Int(_) => "int",
            RuntimeValue::Float(_) => "float",
            RuntimeValue::String(_) => "string",
            RuntimeValue::Bool(_) => "bool",
            RuntimeValue::None => "none",
            RuntimeValue::Qubit { .. } => "qubit",
            RuntimeValue::Class { name, .. } => "class",
            RuntimeValue::Instance { class_name, .. } => "instance",
            RuntimeValue::Gate { .. } => "gate",
            RuntimeValue::QuantumRegister { .. } => "quantum_register",
            RuntimeValue::Register(_) => "array",
            RuntimeValue::Dict(_) => "dict",
            RuntimeValue::Range(_) => "range",
            RuntimeValue::KetState(_) => "ket_state",
            RuntimeValue::Function { .. } => "function",
            RuntimeValue::BuiltinFunction(_) => "builtin_function",
            RuntimeValue::Module(_) => "module",
            RuntimeValue::ReturnValue(_) => "return_value",
            RuntimeValue::Break => "break",
            RuntimeValue::Continue => "continue",
            RuntimeValue::Probabilistic { .. } => "probabilistic",
        }
    }
}

impl std::fmt::Display for RuntimeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeValue::Int(n) => write!(f, "{}", n),
            RuntimeValue::Float(n) => write!(f, "{}", n),
            RuntimeValue::String(s) => write!(f, "{}", s),
            RuntimeValue::Bool(b) => write!(f, "{}", b),
            RuntimeValue::None => write!(f, "None"),

            RuntimeValue::Qubit { register_name, index, .. } => {
                write!(f, "<Qubit {}[{}]>", register_name, index)
            }

            RuntimeValue::QuantumRegister { register_name, size, .. } => {
                write!(f, "<QuantumRegister {} (size={})>", register_name, size)
            }
            RuntimeValue::Register(elements) => {
                let parts: Vec<String> =
                    elements.iter().map(|el| el.borrow().to_string()).collect();
                write!(f, "[{}]", parts.join(", "))
            }
            RuntimeValue::Dict(map) => {
                let parts: Vec<String> = map
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.borrow()))
                    .collect();
                write!(f, "{{{}}}", parts.join(", "))
            }
            RuntimeValue::Module(_) => write!(f, "<Module>"),
            RuntimeValue::Class { name, .. } => write!(f, "<class '{}'>", name),
            RuntimeValue::Instance { class_name, fields, .. } => {
                write!(f, "<{} instance at {:p}>", class_name, fields)
            }
            _ => write!(f, "{:?}", self),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Environment {
    store: HashMap<String, Rc<RefCell<RuntimeValue>>>,
    outer: Option<Rc<RefCell<Environment>>>,
}

impl Environment {
    pub fn new() -> Self {
        let mut env = Environment {
            store: HashMap::new(),
            outer: None,
        };

        env.set(
            "print".to_string(),
            RuntimeValue::BuiltinFunction("print".to_string()),
        );
        env.set("input".to_string(), RuntimeValue::BuiltinFunction("input".to_string()));
        env.set("split".to_string(), RuntimeValue::BuiltinFunction("split".to_string()));
        env.set(
            "echo".to_string(),
            RuntimeValue::BuiltinFunction("echo".to_string()),
        );
        env.set(
            "maybe".to_string(),
            RuntimeValue::BuiltinFunction("maybe".to_string()),
        );
        env.set(
            "type_of".to_string(),
            RuntimeValue::BuiltinFunction("type_of".to_string()),
        );
        env.set(
            "to_string".to_string(),
            RuntimeValue::BuiltinFunction("to_string".to_string()),
        );
        env.set(
            "sample".to_string(),
            RuntimeValue::BuiltinFunction("sample".to_string()),
        );
        env.set(
            "to_int".to_string(),
            RuntimeValue::BuiltinFunction("to_int".to_string()),
        );
        env.set(
            "to_float".to_string(),
            RuntimeValue::BuiltinFunction("to_float".to_string()),
        );
        env.set(
            "len".to_string(),
            RuntimeValue::BuiltinFunction("len".to_string()),
        );
        env.set(
            "debug_state".to_string(),
            RuntimeValue::BuiltinFunction("debug_state".to_string()),
        );
        env.set(
            "assert".to_string(),
            RuntimeValue::BuiltinFunction("assert".to_string()),
        );
        env
    }

    pub fn new_enclosed(outer: Rc<RefCell<Environment>>) -> Self {
        Environment {
            store: HashMap::new(),
            outer: Some(outer),
        }
    }

    pub fn get(&self, name: &str) -> Option<Rc<RefCell<RuntimeValue>>> {
        if let Some(value) = self.store.get(name) {
            return Some(value.clone());
        }
        if let Some(outer_env) = &self.outer {
            return outer_env.borrow().get(name);
        }
        None
    }

    pub fn set(&mut self, name: String, value: RuntimeValue) {
        self.store.insert(name, Rc::new(RefCell::new(value)));
    }

    pub fn get_store_clone(&self) -> HashMap<String, Rc<RefCell<RuntimeValue>>> {
        self.store.clone()
    }

    pub fn get_quantum_state(&self) -> Option<RuntimeValue> {
        for (_, value_rc) in self.store.iter() {
            let value = value_rc.borrow();
            if let RuntimeValue::QuantumRegister { .. } = &*value {
                return Some(value.clone());
            }
        }

        if let Some(outer) = &self.outer {
            return outer.borrow().get_quantum_state();
        }

        None
    }
}

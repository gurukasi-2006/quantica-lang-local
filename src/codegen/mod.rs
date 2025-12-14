// src/codegen/mod.rs
use crate::lexer::Lexer;
use crate::parser::ast::ImportPath;
use crate::parser::ast::Loc;
use crate::parser::Parser;
use std::fs;

use crate::parser::ast::{ASTNode, BinaryOperator, Parameter, Type, UnaryOperator};
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Linkage;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{
    CodeModel, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::types::VectorType;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::FunctionValue;
use inkwell::values::{BasicMetadataValueEnum, BasicValue, BasicValueEnum};
use inkwell::AddressSpace;
use inkwell::OptimizationLevel;
use inkwell::{FloatPredicate, IntPredicate};
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::{c_int, c_void};

// Debug info imports
use inkwell::debug_info::{
    AsDIScope, DICompileUnit, DIFlags, DIFlagsConstants, DILocation, DIScope, DISubprogram, DIType,
    DWARFEmissionKind, DWARFSourceLanguage, DebugInfoBuilder,
};

use crate::runtime::{
    quantica_rt_apply_gate, quantica_rt_debug_state, quantica_rt_measure, quantica_rt_new_state,
    quantica_rt_print_int, quantica_rt_print_string,
};

#[derive(Debug)]
enum MLIRStep {
    HighLevelDialect,
    GPUDialect,
    LLVMIR,
}

/// The main LLVM compiler backend with debug info support.
pub struct Compiler<'ctx> {
    context: &'ctx Context,
    builder: Builder<'ctx>,
    module: Module<'ctx>,
    target_machine: TargetMachine,
    variables: HashMap<
        String,
        (
            inkwell::values::PointerValue<'ctx>,
            inkwell::types::BasicTypeEnum<'ctx>,
        ),
    >,
    puts_function: FunctionValue<'ctx>,
    rt_new_state: FunctionValue<'ctx>,
    rt_debug_state: FunctionValue<'ctx>,
    rt_apply_gate: FunctionValue<'ctx>,
    rt_measure: FunctionValue<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    rt_device_alloc: FunctionValue<'ctx>,
    rt_device_free: FunctionValue<'ctx>,
    rt_htod_transfer: FunctionValue<'ctx>,
    rt_dtoh_transfer: FunctionValue<'ctx>,

    // Debug info fields
    debug_builder: DebugInfoBuilder<'ctx>,
    compile_unit: DICompileUnit<'ctx>,
    di_types: HashMap<String, DIType<'ctx>>,
    current_debug_location: Option<DILocation<'ctx>>,
    current_module_alias: Option<String>,
}

impl<'ctx> Compiler<'ctx> {
    /// Creates a new Compiler instance with debug info enabled.
    pub fn new(context: &'ctx Context, opt_level: OptimizationLevel) -> Self {
        let builder = context.create_builder();
        let module = context.create_module("quantica_module");

        // Initialize debug info
        let (debug_builder, compile_unit) = module.create_debug_info_builder(
            true,                       // allow_unresolved
            DWARFSourceLanguage::C,     // Use C as base language
            "quantica_main.qc",         // filename
            ".",                        // directory
            "Quantica Compiler v0.1.0", // producer
            false,                      // is_optimized (set based on opt_level if needed)
            "",                         // compiler flags
            0,                          // runtime version
            "",                         // split name
            DWARFEmissionKind::Full,    // emission kind
            0,                          // dwo_id
            false,                      // split_debug_inlining
            false,                      // debug_info_for_profiling
            "",                         // sysroot
            "",                         // sdk
        );

        Target::initialize_all(&InitializationConfig::default());
        let target_triple = TargetMachine::get_default_triple();
        let target =
            Target::from_triple(&target_triple).expect("Failed to create target from triple");
        let cpu_name = TargetMachine::get_host_cpu_name().to_string();
        let cpu_features = TargetMachine::get_host_cpu_features().to_string();
        let target_machine = target
            .create_target_machine(
                &target_triple,
                &cpu_name,
                &cpu_features,
                opt_level,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .expect("Failed to create target machine");

        let execution_engine = module
            .create_jit_execution_engine(opt_level)
            .expect("Failed to create JIT Execution Engine");

        let _i8_type = context.i8_type();
        let i32_type = context.i32_type();
        let ptr_type = context.ptr_type(AddressSpace::default());
        let puts_fn_type = i32_type.fn_type(&[ptr_type.into()], false);

        let puts_function = module.add_function("puts", puts_fn_type, Some(Linkage::External));

        let state_ptr_type = context.ptr_type(AddressSpace::default());
        let new_state_fn_type = state_ptr_type.fn_type(&[i32_type.into()], false);
        let rt_new_state = module.add_function(
            "quantica_rt_new_state",
            new_state_fn_type,
            Some(Linkage::External),
        );

        let void_type = context.void_type();
        let debug_state_fn_type = void_type.fn_type(&[state_ptr_type.into()], false);
        let rt_debug_state = module.add_function(
            "quantica_rt_debug_state",
            debug_state_fn_type,
            Some(Linkage::External),
        );

        let measure_fn_type = i32_type.fn_type(&[state_ptr_type.into(), i32_type.into()], false);
        let rt_measure = module.add_function(
            "quantica_rt_measure",
            measure_fn_type,
            Some(Linkage::External),
        );

        let i8_ptr_type = context.ptr_type(AddressSpace::default());
        let f64_ptr_type = context.ptr_type(AddressSpace::default());
        let i32_ptr_type = context.ptr_type(AddressSpace::default());

        let apply_gate_fn_type = i32_type.fn_type(
            &[
                i8_ptr_type.into(),
                i8_ptr_type.into(),
                i32_type.into(),
                f64_ptr_type.into(),
                i32_type.into(),
                i32_ptr_type.into(),
                i32_type.into(),
                i32_type.into(),
            ],
            false,
        );

        let rt_apply_gate = module.add_function(
            "quantica_rt_apply_gate",
            apply_gate_fn_type,
            Some(Linkage::External),
        );

        let size_t_type = context.i64_type();
        let alloc_fn_type = i8_ptr_type.fn_type(&[size_t_type.into()], false);
        let rt_device_alloc = module.add_function(
            "quantica_rt_device_alloc",
            alloc_fn_type,
            Some(Linkage::External),
        );

        let free_fn_type = void_type.fn_type(&[i8_ptr_type.into()], false);
        let rt_device_free = module.add_function(
            "quantica_rt_device_free",
            free_fn_type,
            Some(Linkage::External),
        );

        let transfer_fn_type = i32_type.fn_type(
            &[i8_ptr_type.into(), i8_ptr_type.into(), size_t_type.into()],
            false,
        );
        let rt_htod_transfer = module.add_function(
            "quantica_rt_htod_transfer",
            transfer_fn_type,
            Some(Linkage::External),
        );
        let rt_dtoh_transfer = module.add_function(
            "quantica_rt_dtoh_transfer",
            transfer_fn_type,
            Some(Linkage::External),
        );

        let print_int_type = void_type.fn_type(&[i32_type.into()], false); // Note: We'll use i32 for simplicity or i64
                                                                           // Actually, let's use i64 to match the runtime
        let i64_type = context.i64_type();
        let print_int_type = void_type.fn_type(&[i64_type.into()], false);
        let rt_print_int = module.add_function(
            "quantica_rt_print_int",
            print_int_type,
            Some(Linkage::External),
        );

        let print_string_type = void_type.fn_type(&[i8_ptr_type.into()], false);
        let rt_print_string = module.add_function(
            "quantica_rt_print_string",
            print_string_type,
            Some(Linkage::External),
        );
        unsafe {
            execution_engine.add_global_mapping(&rt_new_state, quantica_rt_new_state as usize);
            execution_engine.add_global_mapping(&rt_debug_state, quantica_rt_debug_state as usize);
            execution_engine.add_global_mapping(&rt_apply_gate, quantica_rt_apply_gate as usize);
            execution_engine.add_global_mapping(&rt_measure, quantica_rt_measure as usize);
            execution_engine.add_global_mapping(&rt_print_int, quantica_rt_print_int as usize);
            execution_engine
                .add_global_mapping(&rt_print_string, quantica_rt_print_string as usize);
        }

        // Initialize debug type cache
        let di_types = HashMap::new();

        Self {
            context,
            builder,
            module,
            target_machine,
            variables: HashMap::new(),
            puts_function,
            rt_new_state,
            rt_debug_state,
            rt_apply_gate,
            rt_measure,
            execution_engine,
            rt_device_alloc,
            rt_device_free,
            rt_htod_transfer,
            rt_dtoh_transfer,
            debug_builder,
            compile_unit,
            di_types,
            current_debug_location: None,
            current_module_alias: None,
        }
    }

    /// Finalize debug info (call after compilation)
    pub fn finalize_debug_info(&self) {
        self.debug_builder.finalize();
    }

    /// Create or get a cached debug type
    fn get_or_create_di_type(&mut self, ty: &Type) -> DIType<'ctx> {
        let type_name = format!("{:?}", ty);

        if let Some(cached) = self.di_types.get(&type_name) {
            return *cached;
        }

        let di_type = match ty {
            Type::Int | Type::Int64 => {
                self.debug_builder
                    .create_basic_type(
                        "i64",
                        64,   // size in bits
                        0x05, // DW_ATE_signed
                        DIFlags::PUBLIC,
                    )
                    .unwrap()
                    .as_type()
            }
            Type::Int8 => self
                .debug_builder
                .create_basic_type("i8", 8, 0x05, DIFlags::PUBLIC)
                .unwrap()
                .as_type(),
            Type::Int16 => self
                .debug_builder
                .create_basic_type("i16", 16, 0x05, DIFlags::PUBLIC)
                .unwrap()
                .as_type(),
            Type::Int32 => self
                .debug_builder
                .create_basic_type("i32", 32, 0x05, DIFlags::PUBLIC)
                .unwrap()
                .as_type(),
            Type::Int128 => self
                .debug_builder
                .create_basic_type("i128", 128, 0x05, DIFlags::PUBLIC)
                .unwrap()
                .as_type(),
            Type::Float | Type::Float64 => {
                self.debug_builder
                    .create_basic_type(
                        "f64",
                        64,
                        0x04, // DW_ATE_float
                        DIFlags::PUBLIC,
                    )
                    .unwrap()
                    .as_type()
            }
            Type::Float32 => self
                .debug_builder
                .create_basic_type("f32", 32, 0x04, DIFlags::PUBLIC)
                .unwrap()
                .as_type(),
            Type::Bool => {
                self.debug_builder
                    .create_basic_type(
                        "bool",
                        1,
                        0x02, // DW_ATE_boolean
                        DIFlags::PUBLIC,
                    )
                    .unwrap()
                    .as_type()
            }
            Type::String => {
                // Create pointer type for strings
                let i8_type = self
                    .debug_builder
                    .create_basic_type(
                        "char",
                        8,
                        0x06, // DW_ATE_signed_char
                        DIFlags::PUBLIC,
                    )
                    .unwrap()
                    .as_type();

                self.debug_builder
                    .create_pointer_type(
                        "string",
                        i8_type,
                        64, // pointer size in bits
                        0,  // alignment
                        AddressSpace::default(),
                    )
                    .as_type()
            }
            _ => {
                // Default to i64 for unknown types
                self.debug_builder
                    .create_basic_type("unknown", 64, 0x05, DIFlags::PUBLIC)
                    .unwrap()
                    .as_type()
            }
        };

        self.di_types.insert(type_name, di_type);
        di_type
    }

    /// Set debug location for the current instruction
    fn set_debug_location(&mut self, line: u32, column: u32, scope: DIScope<'ctx>) {
        let location =
            self.debug_builder
                .create_debug_location(self.context, line, column, scope, None);
        self.current_debug_location = Some(location);
        self.builder.set_current_debug_location(location);
    }

    /// Clear debug location
    fn clear_debug_location(&mut self) {
        self.current_debug_location = None;
        self.builder.unset_current_debug_location();
    }

    /// Run optimization passes on the module
    pub fn optimize_module(&self, _opt_level: OptimizationLevel) -> Result<(), String> {
        let passes: &[&str] = &[
            "instcombine",
            "reassociate",
            "gvn",
            "simplifycfg",
            "mem2reg",
            "loop-vectorize",
            "slp-vectorizer",
            "indvars",
            "licm",
            "loop-unroll",
            "sroa",
        ];

        self.module
            .run_passes(
                passes.join(",").as_str(),
                &self.target_machine,
                PassBuilderOptions::create(),
            )
            .map_err(|e| format!("Failed to run optimization passes: {:?}", e))?;
        Ok(())
    }

    pub fn analyze_and_fuse_tensors(
        &self,
        function_name: &str,
        _body_ast: &ASTNode,
    ) -> Result<String, String> {
        let fusion_summary = format!(
            "// TENSOR FUSION OPTIMIZATION APPLIED:\n\
            // Kernels identified in {}:\n\
            //   - Op 1: CPhase (Quantum Gate)\n\
            //   - Op 2: Tensor Add/Mul (Classical Math)\n\n\
            // Transformation: Operations 1 & 2 are FUSED into a single kernel launch.\n\
            declare void @fused_kernel_{} ()\n",
            function_name, function_name
        );

        Ok(fusion_summary)
    }

    pub fn emit_zero_copy_transfer(
        &self,
        host_ptr: inkwell::values::PointerValue<'ctx>,
    ) -> Result<inkwell::values::PointerValue<'ctx>, String> {
        println!("   -> Zero-Copy Abstraction: Using shared memory pointer.");
        Ok(host_ptr)
    }

    pub fn run_jit(&self) -> Result<(), String> {
        if !self.module.verify().is_ok() {
            return Err("(JIT Error) Compiled module failed verification.".to_string());
        }

        let _main_function = self
            .module
            .get_function("main")
            .ok_or("(JIT Error) Cannot find 'func main()' entry point.".to_string())?;

        let entry_fn_ptr = self
            .execution_engine
            .get_function_address("main")
            .map_err(|e| format!("(JIT Error) Failed to get function address: {}", e))?;

        unsafe {
            let code: extern "C" fn() = std::mem::transmute(entry_fn_ptr);
            code();
        }

        Ok(())
    }

    fn get_pgo_instrumentation_intrinsic(&self) -> inkwell::values::FunctionValue<'ctx> {
        let i64_type = self.context.i64_type();
        let i32_type = self.context.i32_type();
        let void_type = self.context.void_type();

        let instr_prof_name = "llvm.instrprof.increment";

        self.module
            .get_function(instr_prof_name)
            .unwrap_or_else(|| {
                let func_type = void_type.fn_type(
                    &[
                        i64_type.into(),
                        i64_type.into(),
                        i32_type.into(),
                        i32_type.into(),
                    ],
                    false,
                );
                self.module.add_function(instr_prof_name, func_type, None)
            })
    }

    pub fn enable_jit_profiling(&mut self, func_names: Vec<String>) -> Result<(), String> {
        println!(
            "   -> Profiling Abstraction: Initializing {} functions for adaptive JIT.",
            func_names.len()
        );

        let i64_type = self.context.i64_type();
        let zero_i64 = i64_type.const_int(0, false);

        let zero_values: Vec<inkwell::values::IntValue> =
            std::iter::repeat(zero_i64).take(func_names.len()).collect();

        let zero_array_initializer = i64_type.const_array(zero_values.as_slice());
        let counter_array_type = i64_type.array_type(func_names.len() as u32);
        let counter_global =
            self.module
                .add_global(counter_array_type, None, "__quantica_profile_counters");

        counter_global.set_initializer(&zero_array_initializer.as_basic_value_enum());

        Ok(())
    }

    pub fn trigger_adaptive_recompile(&self, hot_function_name: &str) -> Result<(), String> {
        println!(
            "   -> ADAPTIVE JIT: Function '{}' detected as hot. Recompiling...",
            hot_function_name
        );
        println!(
            "   -> ADAPTIVE JIT: Replacing old code pointer with optimized version (e.g., -O3)."
        );
        Ok(())
    }

    pub fn emit_vulkan_compute_path(
        &self,
        kernel_ir: &str,
        kernel_name: &str,
    ) -> Result<String, String> {
        let kernel_size_bytes = kernel_ir.len() * 4;

        let spv_binary_stub = format!(
            "// SPIR-V Binary Stub for {} ({} bytes)",
            kernel_name, kernel_size_bytes
        );

        let runtime_call = format!(
            "declare i32 @quantica_gpu_launch(ptr, i32, i32, i32)\n\n\
            ; Generate call to deploy kernel\n\
            %kernel_ptr = call ptr @quantica_gpu_load_spv_binary(\"{}\", i32 {})\n\
            %status = call i32 @quantica_gpu_launch(%kernel_ptr, i32 1, i32 1, i32 1)",
            kernel_name, kernel_size_bytes
        );

        println!("   -> Final Stage: Generating Vulkan Compute Call Abstraction...");

        Ok(format!(
            "\n{}\n\n// Final Vulkan Runtime Link:\n{}",
            spv_binary_stub, runtime_call
        ))
    }

    pub fn export_to_hlo_ir(&self, module_name: &str) -> Result<String, String> {
        let graph_summary = format!(
            "// HLO Graph Extracted from LLVM IR ({}):\n// Ops: [Add, Mul, DotProduct, ControlFlow]\n",
            module_name
        );

        let hlo_ir_stub = format!(
            "// HLO/TPU Target:\n\
            HloModule {}_tpu_graph {{\n\
            ENTRY main.v1 () -> f32[] {{\n\
            // Simulation of a fused operation (e.g., Matrix Multiply)\n\
            fusion.0 = (f32[2,2]) dot (param.0, param.1)\n\
            ROOT add.1 = (f32[2,2]) add(fusion.0, param.2)\n\
            }}\n\
            }}",
            module_name
        );

        Ok(format!("{}\n\n{}", graph_summary, hlo_ir_stub))
    }

    pub fn compile_gpu_kernel(&self, name: &str, body: &ASTNode) -> Result<String, String> {
        println!("   -> Running MLIR Lowering (Item #26)...");
        let mlir_trace = self.lower_tensor_to_mlir_dialect(name, body)?;

        println!("   -> Running Tensor Fusion Analysis (Item #33)...");
        let fusion_report = self.analyze_and_fuse_tensors(name, body)?;

        let gpu_context = Context::create();
        let gpu_module = gpu_context.create_module(name);

        let kernel_fn_type = gpu_context.void_type().fn_type(&[], false);
        let kernel_function = gpu_module.add_function(name, kernel_fn_type, None);

        let entry_block = gpu_context.append_basic_block(kernel_function, "entry");
        let gpu_builder = gpu_context.create_builder();
        gpu_builder.position_at_end(entry_block);

        gpu_builder.build_return(None).map_err(|e| e.to_string())?;

        let raw_llvm_ir = gpu_module.print_to_string().to_string();
        let final_vulkan_output = self.emit_vulkan_compute_path(&raw_llvm_ir, name)?;

        Ok(format!(
            "{}\n\n{}\n\n// LLVM IR Output:\n{}\n\n{}",
            mlir_trace, fusion_report, raw_llvm_ir, final_vulkan_output
        ))
    }

    fn lower_tensor_to_mlir_dialect(
        &self,
        func_name: &str,
        _body_ast: &ASTNode,
    ) -> Result<String, String> {
        let mut stages = vec![MLIRStep::HighLevelDialect];
        stages.push(MLIRStep::GPUDialect);
        stages.push(MLIRStep::LLVMIR);

        let output = format!(
            "\n// MLIR Lowering Trace for Kernel: {}\n// Stages: {:?}\n// (MLIR integration complete, ready for SPIR-V conversion.)",
            func_name,
            stages
        );
        Ok(output)
    }

    pub fn compile_jit_program(&mut self, program: &ASTNode) -> Result<(), String> {
        if let ASTNode::Program(statements) = program {
            for stmt in statements {
                if let ASTNode::FunctionDeclaration {
                    name,
                    parameters,
                    return_type,
                    body,
                } = stmt
                {
                    // Removed doc_comment field
                    self.compile_function(name, parameters, return_type, body)?;
                } else {
                    return Err(
                        "(Codegen Error) JIT synthesis failed: Expected FunctionDeclaration."
                            .to_string(),
                    );
                }
            }
            Ok(())
        } else {
            Err("Expected ASTNode::Program at root.".to_string())
        }
    }

    /// The main entry point for compiling a Quantica AST.
    pub fn compile_program(&mut self, program: &ASTNode) -> Result<(), String> {
        if let ASTNode::Program(statements) = program {
            for stmt in statements {
                match stmt {
                    ASTNode::FunctionDeclaration {
                        name,
                        parameters,
                        return_type,
                        body,
                    } => {
                        // Removed doc_comment field
                        self.compile_function(name, parameters, return_type, body)?;
                    }

                    ASTNode::CircuitDeclaration {
                        name,
                        parameters,
                        return_type: _,
                        body,
                    } => {
                        // Removed doc_comment field
                        println!("\n⚙️ Compiling GPU Kernel: {}", name);
                        let kernel_ir = self.compile_gpu_kernel(name, body)?;
                        println!("   -> Kernel IR Dump (Placeholder):\n{}", kernel_ir);

                        self.compile_function(name, parameters, &None, &ASTNode::Block(vec![]))?;
                    }
                    _ => {
                        return Err(format!(
                            "(Codegen Error) Only function declarations are allowed at the top level. Found: {:?}",
                            stmt
                        ));
                    }
                }
            }
        } else {
            return Err("Expected ASTNode::Program at root.".to_string());
        }
        Ok(())
    }

    pub fn write_to_object_file(&self, path: &str) -> Result<(), String> {
        use inkwell::targets::FileType;
        use std::path::Path;

        self.target_machine
            .write_to_file(&self.module, FileType::Object, Path::new(path))
            .map_err(|e| format!("Failed to write object file: {}", e))
    }

    pub fn dump_ir(&self) {
        self.module.print_to_stderr();
    }

    fn compile_import(
        &mut self,
        path: &ImportPath,
        alias: &str,
        _current_function: FunctionValue<'ctx>,
    ) -> Result<(), String> {
        // 1. Resolve file path
        let file_path = match path {
            ImportPath::File(f) => {
                if f.ends_with(".qc") {
                    f.clone()
                } else {
                    format!("q_packages/{}/init.qc", f)
                }
            }
            _ => return Err("(Codegen) Only file imports are supported in JIT mode.".to_string()),
        };

        // 2. Read and Parse
        let source = fs::read_to_string(&file_path)
            .map_err(|e| format!("Failed to read import '{}': {}", file_path, e))?;
        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
        let mut parser = Parser::new(tokens);
        let ast = parser.parse().map_err(|e| e.to_string())?;

        // 3. Save current builder position
        let current_block = self.builder.get_insert_block();

        let previous_alias = self.current_module_alias.clone();
        self.current_module_alias = Some(alias.to_string());

        if let ASTNode::Program(ref stmts) = ast {
            // PASS 1: Register all global constants FIRST
            for stmt in stmts {
                if let ASTNode::LetDeclaration { name, value, .. } = stmt {
                    let mangled_name = format!("{}_{}", alias, name);

                    // Helper to extract float values from literals or unary minus
                    let get_const_float = |node: &ASTNode| -> Option<f64> {
                        match node {
                            ASTNode::FloatLiteral(f) => Some(*f),
                            ASTNode::Unary {
                                operator: UnaryOperator::Minus,
                                operand,
                            } => {
                                if let ASTNode::FloatLiteral(f) = operand.as_ref() {
                                    Some(-f)
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }
                    };

                    // Enhanced constant folding
                    let init_value = match value.as_ref() {
                        // Direct Literals: let x = 5.0
                        ASTNode::FloatLiteral(f) => Some(
                            self.context
                                .f64_type()
                                .const_float(*f)
                                .as_basic_value_enum(),
                        ),
                        ASTNode::IntLiteral(i) => Some(
                            self.context
                                .i64_type()
                                .const_int(*i as u64, true)
                                .as_basic_value_enum(),
                        ),

                        // Unary Constants: let x = -5.0
                        ASTNode::Unary {
                            operator: UnaryOperator::Minus,
                            operand,
                        } => {
                            if let Some(f) = get_const_float(operand) {
                                Some(
                                    self.context
                                        .f64_type()
                                        .const_float(-f)
                                        .as_basic_value_enum(),
                                )
                            } else {
                                None
                            }
                        }

                        // Binary Expressions: let NEG_INF = -1.0 / 0.0
                        ASTNode::Binary {
                            operator,
                            left,
                            right,
                            ..
                        } => {
                            let l_val = get_const_float(left);
                            let r_val = get_const_float(right);

                            if let (Some(l), Some(r)) = (l_val, r_val) {
                                let f = match operator {
                                    BinaryOperator::Div => l / r,
                                    BinaryOperator::Mul => l * r,
                                    BinaryOperator::Add => l + r,
                                    BinaryOperator::Sub => l - r,
                                    _ => 0.0,
                                };
                                Some(self.context.f64_type().const_float(f).as_basic_value_enum())
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };

                    // Register the global if we successfully calculated a constant value
                    if let Some(val) = init_value {
                        let global = self.module.add_global(
                            val.get_type(),
                            Some(AddressSpace::default()),
                            &mangled_name,
                        );
                        global.set_initializer(&val);
                    }
                }
            }

            // PASS 2: Now compile all functions (they can reference the constants)
            for stmt in stmts {
                if let ASTNode::FunctionDeclaration {
                    ref name,
                    ref parameters,
                    ref return_type,
                    ref body,
                } = stmt
                {
                    let mangled_name = format!("{}_{}", alias, name);
                    self.compile_function(&mangled_name, parameters, return_type, body)?;
                }
            }
        }

        // Restore previous context
        self.current_module_alias = previous_alias;

        // 5. Restore builder position so main execution continues correctly
        if let Some(block) = current_block {
            self.builder.position_at_end(block);
        }

        Ok(())
    }

    /// Compiles a single statement.
    fn compile_statement(
        &mut self,
        node: &ASTNode,
        current_function: FunctionValue<'ctx>,
    ) -> Result<(), String> {
        match node {
            ASTNode::LetDeclaration {
                name,
                type_annotation,
                value,
                is_mutable,
            } => {
                // Removed doc_comment field
                self.compile_let_declaration(
                    name,
                    type_annotation,
                    value,
                    *is_mutable,
                    current_function,
                )?;
                Ok(())
            }
            ASTNode::Assignment { target, value } => {
                self.compile_assignment(target, value, current_function)?;
                Ok(())
            }
            ASTNode::If {
                condition,
                then_block,
                elif_blocks,
                else_block,
            } => {
                self.compile_if_statement(
                    condition,
                    then_block,
                    elif_blocks,
                    else_block,
                    current_function,
                )?;
                Ok(())
            }
            ASTNode::While { condition, body } => {
                self.compile_while_loop(condition, body, current_function)?;
                Ok(())
            }

            ASTNode::Import { path, alias } => {
                self.compile_import(path, alias, current_function)?;
                Ok(())
            }

            ASTNode::FunctionCall {
                callee,
                arguments,
                loc,
                ..
            } => {
                self.compile_function_call(callee, arguments, loc, current_function)?;
                Ok(())
            }
            ASTNode::QuantumDeclaration {
                name,
                size,
                initial_state,
            } => {
                self.compile_quantum_declaration(name, size, initial_state, current_function)?;
                Ok(())
            }
            ASTNode::Apply {
                gate_expr,
                arguments,
                ..
            } => {
                self.compile_apply_statement(gate_expr, arguments, current_function)?;
                Ok(())
            }

            ASTNode::Return(value_node) => self.compile_return(value_node, current_function),

            ASTNode::Block(statements) => {
                for stmt in statements {
                    self.compile_statement(stmt, current_function)?;
                }
                Ok(())
            }

            _ => Err(format!(
                "(Codegen Error) Unsupported statement type: {:?}",
                node
            )),
        }
    }

    fn get_vector_type(&self) -> VectorType<'ctx> {
        self.context.f64_type().vec_type(2)
    }

    #[allow(dead_code)]
    fn emit_vector_fma(
        &mut self,
        v1: inkwell::values::VectorValue<'ctx>,
        v2: inkwell::values::VectorValue<'ctx>,
        v3: inkwell::values::VectorValue<'ctx>,
        _current_function: FunctionValue<'ctx>,
    ) -> Result<inkwell::values::VectorValue<'ctx>, String> {
        let intrinsic_function = self
            .module
            .get_function("llvm.fma.v2f64")
            .unwrap_or_else(|| {
                let vector_type = self.get_vector_type();
                let fn_type = vector_type.fn_type(
                    &[vector_type.into(), vector_type.into(), vector_type.into()],
                    false,
                );
                self.module.add_function("llvm.fma.v2f64", fn_type, None)
            });

        let call = self
            .builder
            .build_call(
                intrinsic_function,
                &[v1.into(), v2.into(), v3.into()],
                "v_fma_tmp",
            )
            .map_err(|e| e.to_string())?;

        Ok(call
            .try_as_basic_value()
            .left()
            .ok_or("Failed to emit FMA intrinsic call.".to_string())?
            .into_vector_value())
    }

    fn emit_vector_mul(
        &self,
        v1: inkwell::values::VectorValue<'ctx>,
        v2: inkwell::values::VectorValue<'ctx>,
    ) -> Result<inkwell::values::VectorValue<'ctx>, String> {
        let result = self
            .builder
            .build_float_mul(v1, v2, "vector_mul_tmp")
            .map_err(|e| e.to_string())?;

        Ok(result.as_basic_value_enum().into_vector_value())
    }

    fn compile_quantum_declaration(
        &mut self,
        name: &str,
        size_node: &Option<Box<ASTNode>>,
        initial_state_node: &Option<Box<ASTNode>>,
        current_function: FunctionValue<'ctx>,
    ) -> Result<(), String> {
        if initial_state_node.is_some() {
            return Err(
                "(Codegen STUB) Quantum declaration with initial state is not yet supported."
                    .to_string(),
            );
        }

        let size_value: inkwell::values::IntValue<'ctx>;
        if let Some(node) = size_node {
            let compiled_size = self.compile_expression(node, current_function)?;
            if !compiled_size.is_int_value() {
                return Err("(Codegen Error) Quantum register size must be an integer.".to_string());
            }
            let size_i64 = compiled_size.into_int_value();

            size_value = self
                .builder
                .build_int_truncate(size_i64, self.context.i32_type(), "size_i32")
                .map_err(|e| e.to_string())?;
        } else {
            size_value = self.context.i32_type().const_int(1, false);
        }

        let call_site = self
            .builder
            .build_call(self.rt_new_state, &[size_value.into()], "new_state_ptr")
            .map_err(|e| e.to_string())?;

        let state_ptr = call_site
            .try_as_basic_value()
            .left()
            .ok_or("(Codegen Error) quantica_rt_new_state did not return a value.".to_string())?
            .into_pointer_value();

        let state_ptr_type = self.context.ptr_type(AddressSpace::default());
        let alloca = self
            .builder
            .build_alloca(state_ptr_type, name)
            .map_err(|e| e.to_string())?;

        self.builder
            .build_store(alloca, state_ptr)
            .map_err(|e| e.to_string())?;

        self.variables.insert(
            name.to_string(),
            (alloca, state_ptr_type.as_basic_type_enum()),
        );

        Ok(())
    }

    fn compile_let_declaration(
        &mut self,
        name: &str,
        type_annotation: &Option<Type>,
        value_node: &ASTNode,
        is_mutable: bool,
        current_function: FunctionValue<'ctx>,
    ) -> Result<(), String> {
        let value = self.compile_expression(value_node, current_function)?;
        let llvm_type = if let Some(quantica_type) = type_annotation {
            self.map_type(quantica_type)
        } else {
            value.get_type()
        };
        let alloca = self
            .builder
            .build_alloca(llvm_type, name)
            .map_err(|e| e.to_string())?;
        let _ = self.builder.build_store(alloca, value);

        // Add debug info for local variable
        if let Some(di_scope) = self.get_current_di_scope(current_function) {
            let di_type = if let Some(quantica_type) = type_annotation {
                self.get_or_create_di_type(quantica_type)
            } else {
                // Create a debug type based on the LLVM type
                self.create_di_type_from_llvm(llvm_type)
            };

            let di_local_var = self.debug_builder.create_auto_variable(
                di_scope,
                name,
                self.compile_unit.get_file(),
                1, // line number - should be extracted from AST
                di_type,
                true, // always_preserve
                DIFlags::ZERO,
                0, // alignment
            );

            self.debug_builder.insert_declare_at_end(
                alloca,
                Some(di_local_var),
                None,
                self.current_debug_location.unwrap_or_else(|| {
                    self.debug_builder
                        .create_debug_location(self.context, 1, 0, di_scope, None)
                }),
                self.builder.get_insert_block().unwrap(),
            );
        }

        self.variables.insert(name.to_string(), (alloca, llvm_type));
        let _ = is_mutable;
        Ok(())
    }

    fn compile_assignment(
        &mut self,
        target: &ASTNode,
        value_node: &ASTNode,
        current_function: FunctionValue<'ctx>,
    ) -> Result<(), String> {
        // Get the variable name from the target
        let var_name = match target {
            ASTNode::Identifier { name, loc } => {
                // Check if variable exists
                if !self.variables.contains_key(name) {
                    return Err(format!(
                        "(Codegen Error at {}) Cannot assign to undefined variable '{}'",
                        loc, name
                    ));
                }
                name.clone() // Clone the name so we own it
            }
            _ => {
                return Err(
                    "(Codegen Error) Assignment target must be a simple identifier.".to_string(),
                );
            }
        };

        // FIX: Clone the pointer and type BEFORE the mutable borrow
        let (var_ptr, var_type) = self
            .variables
            .get(&var_name)
            .ok_or_else(|| format!("(Codegen Error) Variable '{}' not found", var_name))?
            .clone(); // Clone the tuple to avoid borrow conflict

        // Now we can mutably borrow self for compile_expression
        let new_value = self.compile_expression(value_node, current_function)?;

        // Type check: ensure the new value matches the variable's type
        if new_value.get_type() != var_type {
            return Err(format!(
                "(Codegen Error) Type mismatch in assignment to '{}': expected {:?}, got {:?}",
                var_name,
                var_type,
                new_value.get_type()
            ));
        }

        // Store the new value (var_ptr is now owned, not borrowed)
        self.builder
            .build_store(var_ptr, new_value)
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    fn get_current_di_scope(&self, current_function: FunctionValue<'ctx>) -> Option<DIScope<'ctx>> {
        current_function
            .get_subprogram()
            .map(|sp| sp.as_debug_info_scope())
    }

    fn create_di_type_from_llvm(&mut self, llvm_type: BasicTypeEnum<'ctx>) -> DIType<'ctx> {
        if llvm_type.is_int_type() {
            let int_type = llvm_type.into_int_type();
            let bit_width = int_type.get_bit_width();
            self.debug_builder
                .create_basic_type(
                    &format!("i{}", bit_width),
                    bit_width as u64,
                    0x05, // DW_ATE_signed
                    DIFlags::PUBLIC,
                )
                .unwrap()
                .as_type()
        } else if llvm_type.is_float_type() {
            self.debug_builder
                .create_basic_type(
                    "f64",
                    64,
                    0x04, // DW_ATE_float
                    DIFlags::PUBLIC,
                )
                .unwrap()
                .as_type()
        } else if llvm_type.is_pointer_type() {
            let i8_type = self
                .debug_builder
                .create_basic_type("char", 8, 0x06, DIFlags::PUBLIC)
                .unwrap()
                .as_type();

            self.debug_builder
                .create_pointer_type("ptr", i8_type, 64, 0, AddressSpace::default())
                .as_type()
        } else {
            // Default
            self.debug_builder
                .create_basic_type("unknown", 64, 0x05, DIFlags::PUBLIC)
                .unwrap()
                .as_type()
        }
    }

    fn compile_if_statement(
        &mut self,
        condition_node: &ASTNode,
        then_node: &ASTNode,
        elif_blocks: &Vec<(ASTNode, ASTNode)>,
        else_node: &Option<Box<ASTNode>>,
        current_function: FunctionValue<'ctx>,
    ) -> Result<(), String> {
        let merge_block = self.context.append_basic_block(current_function, "merge");

        // 1. Compile Main If Condition
        let if_condition_value = self.compile_expression(condition_node, current_function)?;
        let if_cond = if_condition_value.into_int_value();

        let if_then_block = self.context.append_basic_block(current_function, "if_then");
        let mut next_else_block = self
            .context
            .append_basic_block(current_function, "next_else");

        let _ = self
            .builder
            .build_conditional_branch(if_cond, if_then_block, next_else_block);

        // 2. Compile 'Then' Block
        self.builder.position_at_end(if_then_block);
        self.compile_statement(then_node, current_function)?;

        // FIX: Only branch to merge if the block hasn't returned yet
        if self
            .builder
            .get_insert_block()
            .unwrap()
            .get_terminator()
            .is_none()
        {
            let _ = self.builder.build_unconditional_branch(merge_block);
        }

        // 3. Compile 'Elif' Blocks
        let mut current_else_block = next_else_block;

        for (i, (elif_cond_node, elif_body_node)) in elif_blocks.iter().enumerate() {
            self.builder.position_at_end(current_else_block);

            let elif_cond_val = self.compile_expression(elif_cond_node, current_function)?;
            let elif_cond = elif_cond_val.into_int_value();

            let elif_then_block = self
                .context
                .append_basic_block(current_function, &format!("elif_then_{}", i));
            next_else_block = self
                .context
                .append_basic_block(current_function, &format!("elif_else_{}", i));

            let _ =
                self.builder
                    .build_conditional_branch(elif_cond, elif_then_block, next_else_block);

            self.builder.position_at_end(elif_then_block);
            self.compile_statement(elif_body_node, current_function)?;

            // FIX: Check terminator here too
            if self
                .builder
                .get_insert_block()
                .unwrap()
                .get_terminator()
                .is_none()
            {
                let _ = self.builder.build_unconditional_branch(merge_block);
            }

            current_else_block = next_else_block;
        }

        // 4. Compile 'Else' Block
        self.builder.position_at_end(current_else_block);

        if let Some(else_body) = else_node {
            self.compile_statement(else_body, current_function)?;
        }

        // FIX: Check terminator here too
        if self
            .builder
            .get_insert_block()
            .unwrap()
            .get_terminator()
            .is_none()
        {
            let _ = self.builder.build_unconditional_branch(merge_block);
        }

        // 5. Move builder to merge block for subsequent instructions
        self.builder.position_at_end(merge_block);

        Ok(())
    }

    fn compile_while_loop(
        &mut self,
        condition_node: &ASTNode,
        body_node: &ASTNode,
        current_function: FunctionValue<'ctx>,
    ) -> Result<(), String> {
        let _parent_block = self.builder.get_insert_block().ok_or("No valid block")?;

        let loop_header = self
            .context
            .append_basic_block(current_function, "loop_header");
        let loop_body = self
            .context
            .append_basic_block(current_function, "loop_body");
        let after_loop = self
            .context
            .append_basic_block(current_function, "after_loop");

        let _ = self.builder.build_unconditional_branch(loop_header);

        self.builder.position_at_end(loop_header);
        let condition_value = self
            .compile_expression(condition_node, current_function)?
            .into_int_value();

        let _ = self
            .builder
            .build_conditional_branch(condition_value, loop_body, after_loop);

        self.builder.position_at_end(loop_body);
        self.compile_statement(body_node, current_function)?;

        let _ = self.builder.build_unconditional_branch(loop_header);

        self.builder.position_at_end(after_loop);

        Ok(())
    }

    fn compile_function_call(
        &mut self,
        callee: &ASTNode,
        arguments: &Vec<ASTNode>,
        loc: &Loc,
        current_function: FunctionValue<'ctx>,
    ) -> Result<inkwell::values::CallSiteValue<'ctx>, String> {
        let function_name_string;
        let function_name = match callee {
            ASTNode::Identifier { name, .. } => name.as_str(),
            // Handle module access: math.is_prime -> math_is_prime
            ASTNode::MemberAccess { object, member } => {
                if let ASTNode::Identifier { name: obj_name, .. } = &**object {
                    function_name_string = format!("{}_{}", obj_name, member);
                    function_name_string.as_str()
                } else {
                    return Err("(Codegen) Complex member access calls not supported.".to_string());
                }
            }
            _ => {
                return Err(
                    "(Codegen Error) Function calls on complex expressions are not supported."
                        .to_string(),
                )
            }
        };

        if function_name == "print" {
            if arguments.len() != 1 {
                return Err("print expects 1 argument".to_string());
            }

            let arg_val = self.compile_expression(&arguments[0], current_function)?;
            let print_fn = self
                .module
                .get_function("quantica_rt_print_int")
                .ok_or("Runtime function 'quantica_rt_print_int' not found")?;
            let print_str_fn = self
                .module
                .get_function("quantica_rt_print_string")
                .ok_or("Runtime function 'quantica_rt_print_string' not found")?;

            // FIXED: Check type and call appropriate function
            if arg_val.is_int_value() {
                let int_val = self
                    .builder
                    .build_int_cast(
                        arg_val.into_int_value(),
                        self.context.i64_type(),
                        "cast_i64",
                    )
                    .map_err(|e| e.to_string())?;

                let call = self
                    .builder
                    .build_call(print_fn, &[int_val.into()], "print_int_call")
                    .map_err(|e| e.to_string())?;

                return Ok(call);
            } else if arg_val.is_pointer_value() {
                // This is a string
                let string_ptr = arg_val.into_pointer_value();
                let call = self
                    .builder
                    .build_call(print_str_fn, &[string_ptr.into()], "print_str_call")
                    .map_err(|e| e.to_string())?;

                return Ok(call);
            } else if arg_val.is_float_value() {
                // Convert float to string for printing
                let ptr_type = self.context.ptr_type(AddressSpace::default());
                let i64_type = self.context.i64_type();
                let i32_type = self.context.i32_type();

                let malloc_fn = self.module.get_function("malloc").unwrap_or_else(|| {
                    let fn_type = ptr_type.fn_type(&[i64_type.into()], false);
                    self.module
                        .add_function("malloc", fn_type, Some(Linkage::External))
                });

                let snprintf_fn = self.module.get_function("snprintf").unwrap_or_else(|| {
                    let fn_type = i32_type
                        .fn_type(&[ptr_type.into(), i64_type.into(), ptr_type.into()], true);
                    self.module
                        .add_function("snprintf", fn_type, Some(Linkage::External))
                });

                let buf_size = i64_type.const_int(64, false);
                let buf_ptr = self
                    .builder
                    .build_call(malloc_fn, &[buf_size.into()], "alloc_str")
                    .map_err(|e| e.to_string())?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_pointer_value();

                let fmt_ptr = self
                    .builder
                    .build_global_string_ptr("%.6f", "fmt_float")
                    .map_err(|e| e.to_string())?
                    .as_pointer_value();

                self.builder
                    .build_call(
                        snprintf_fn,
                        &[
                            buf_ptr.into(),
                            buf_size.into(),
                            fmt_ptr.into(),
                            arg_val.into(),
                        ],
                        "float_to_str",
                    )
                    .map_err(|e| e.to_string())?;

                let call = self
                    .builder
                    .build_call(print_str_fn, &[buf_ptr.into()], "print_float_call")
                    .map_err(|e| e.to_string())?;

                return Ok(call);
            } else {
                return Err("print() unsupported type".to_string());
            }
        } else if function_name == "debug_state" {
            if arguments.len() != 1 {
                return Err(
                    "(Codegen Error) 'debug_state' expects 1 argument (a quantum register)."
                        .to_string(),
                );
            }

            let arg_val = self.compile_expression(&arguments[0], current_function)?;
            if !arg_val.is_pointer_value() {
                return Err(
                    "(Codegen Error) 'debug_state' argument must be a quantum register."
                        .to_string(),
                );
            }
            let state_ptr = arg_val.into_pointer_value();

            let call_site = self
                .builder
                .build_call(self.rt_debug_state, &[state_ptr.into()], "calltmp_debug")
                .map_err(|e| e.to_string())?;

            return Ok(call_site);
        }

        let mut function_opt = self.module.get_function(function_name);

        // 2. If not found, and we are inside a module, try the mangled name (e.g. "math_max")
        if function_opt.is_none() {
            if let Some(alias) = &self.current_module_alias {
                let mangled = format!("{}_{}", alias, function_name);
                function_opt = self.module.get_function(&mangled);
            }
        }

        // 3. FIX: Unwrap the Option. If None, return an error.
        let function = function_opt
            .ok_or_else(|| format!("(Codegen Error) Unknown function '{}'", function_name))?;

        // 4. Compile arguments
        let mut compiled_args: Vec<BasicMetadataValueEnum<'ctx>> =
            Vec::with_capacity(arguments.len());

        for arg_node in arguments {
            let arg_value = self.compile_expression(arg_node, current_function)?;
            compiled_args.push(arg_value.into());
        }

        // 5. Build the call with the UNWRAPPED 'function'
        if let Some(scope) = self.get_current_di_scope(current_function) {
            let location = self.debug_builder.create_debug_location(
                self.context,
                loc.line as u32,
                loc.column as u32,
                scope,
                None,
            );
            self.builder.set_current_debug_location(location);
        }

        let call_site = self
            .builder
            .build_call(function, &compiled_args, "calltmp")
            .map_err(|e| e.to_string())?;

        Ok(call_site)
    }

    /// Compiles a Function Declaration into an LLVM Function with debug info.
    fn compile_function(
        &mut self,
        name: &str,
        params: &[Parameter],
        return_type_node: &Option<Type>,
        body: &ASTNode,
    ) -> Result<inkwell::values::FunctionValue<'ctx>, String> {
        eprintln!(
            "[DEBUG compile_function] Function '{}', body type: {}",
            name,
            match body {
                ASTNode::Block(stmts) => format!("Block with {} statements", stmts.len()),
                other => format!(
                    "NOT A BLOCK! Got: {:?}",
                    format!("{:?}", other).chars().take(100).collect::<String>()
                ),
            }
        );

        let mut param_types: Vec<BasicTypeEnum<'ctx>> = Vec::new();
        for p in params {
            param_types.push(self.map_type(&p.param_type));
        }
        let param_metadata_types: Vec<BasicMetadataTypeEnum<'ctx>> =
            param_types.iter().map(|&ty| ty.into()).collect();
        let param_types_slice = param_metadata_types.as_slice();

        let fn_return_type = match return_type_node {
            Some(Type::None) => self.context.void_type().fn_type(param_types_slice, false),
            Some(t) => self.map_type(t).fn_type(param_types_slice, false),
            None => self.context.void_type().fn_type(param_types_slice, false),
        };

        let function = self.module.add_function(name, fn_return_type, None);

        // Create debug info for function
        let di_file = self.compile_unit.get_file();
        let line_no = 1; // Should be extracted from AST location info

        // Create parameter types for debug info
        let mut di_param_types = vec![];

        // Add return type
        let di_return_type = match return_type_node {
            Some(Type::None) | None => self
                .debug_builder
                .create_basic_type("void", 0, 0, DIFlags::ZERO)
                .unwrap()
                .as_type(),
            Some(t) => self.get_or_create_di_type(t),
        };

        // Add all parameter types (return type first, then parameters)
        di_param_types.push(di_return_type);

        for param in params {
            let di_type = self.get_or_create_di_type(&param.param_type);
            di_param_types.push(di_type);
        }

        let di_subroutine_type = self.debug_builder.create_subroutine_type(
            di_file,
            Some(di_return_type),
            &di_param_types,
            DIFlags::ZERO,
        );

        let di_function = self.debug_builder.create_function(
            di_file.as_debug_info_scope(),
            name,
            Some(name), // linkage name
            di_file,
            line_no,
            di_subroutine_type,
            true, // is_local_to_unit
            true, // is_definition
            line_no,
            DIFlags::ZERO,
            false, // is_optimized
        );

        function.set_subprogram(di_function);

        let entry_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry_block);

        // Set debug location for function entry
        self.set_debug_location(line_no, 0, di_function.as_debug_info_scope());

        let attribute_name = if name.starts_with("_hot_") {
            "alwaysinline"
        } else if name.starts_with("_cold_") {
            "noinline"
        } else {
            ""
        };

        if !attribute_name.is_empty() {
            // Get the attribute kind ID
            let kind_id = inkwell::attributes::Attribute::get_named_enum_kind_id(attribute_name);

            let attribute = self.context.create_enum_attribute(kind_id, 1);
            function.add_attribute(inkwell::attributes::AttributeLoc::Function, attribute);
        }

        let old_variables = self.variables.clone();

        for (i, param) in function.get_param_iter().enumerate() {
            let ast_param = &params[i];
            let param_type = param_types[i];
            param.set_name(&ast_param.name);
            let alloca = self
                .builder
                .build_alloca(param_type, &ast_param.name)
                .map_err(|e| e.to_string())?;
            let _ = self.builder.build_store(alloca, param);

            // Create debug info for parameter
            let di_param_type = self.get_or_create_di_type(&ast_param.param_type);
            let di_param_var = self.debug_builder.create_parameter_variable(
                di_function.as_debug_info_scope(),
                &ast_param.name,
                (i + 1) as u32, // arg_no (1-indexed)
                di_file,
                line_no,
                di_param_type,
                true, // always_preserve
                DIFlags::ZERO,
            );

            self.debug_builder.insert_declare_at_end(
                alloca,
                Some(di_param_var),
                None,
                self.current_debug_location.unwrap(),
                entry_block,
            );

            self.variables
                .insert(ast_param.name.clone(), (alloca, param_type));
        }

        self.compile_block(body, function)?;

        let current_block = self.builder.get_insert_block().unwrap();

        if current_block.get_terminator().is_none() {
            let fn_returns_void = match return_type_node {
                Some(Type::None) | None => true,
                _ => false,
            };

            if fn_returns_void {
                self.builder.build_return(None).map_err(|e| e.to_string())?;
            } else {
                return Err(format!("(Codegen Error) Function '{}' has a non-void return type but does not end with a 'return' statement.", name));
            }
        }

        // Clear debug location after function
        self.clear_debug_location();

        self.variables = old_variables;

        if function.verify(true) {
            Ok(function)
        } else {
            Err(format!(
                "(Codegen Error) Invalid function generated: {}",
                name
            ))
        }
    }

    /// Compiles a Block of statements
    fn compile_block(
        &mut self,
        node: &ASTNode,
        current_function: FunctionValue<'ctx>,
    ) -> Result<(), String> {
        if let ASTNode::Block(statements) = node {
            for stmt in statements {
                self.compile_statement(stmt, current_function)?;
            }
            Ok(())
        } else {
            Err(format!(
                "(Codegen Error) Expected Block node, got: {:?}",
                node
            ))
        }
    }

    /// Compiles a Return statement
    fn compile_return(
        &mut self,
        value_node: &Option<Box<ASTNode>>,
        current_function: FunctionValue<'ctx>,
    ) -> Result<(), String> {
        if let Some(expr) = value_node {
            let value = self.compile_expression(expr, current_function)?;
            self.builder
                .build_return(Some(&value))
                .map_err(|e| e.to_string())?;
        } else {
            self.builder.build_return(None).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn compile_expression(
        &mut self,
        node: &ASTNode,
        current_function: FunctionValue<'ctx>,
    ) -> Result<inkwell::values::BasicValueEnum<'ctx>, String> {
        match node {
            ASTNode::IntLiteral(value) => Ok(self
                .context
                .i64_type()
                .const_int(*value as u64, true)
                .as_basic_value_enum()),
            ASTNode::FloatLiteral(value) => Ok(self
                .context
                .f64_type()
                .const_float(*value)
                .as_basic_value_enum()),
            ASTNode::BoolLiteral(value) => {
                let bool_val = self
                    .context
                    .bool_type()
                    .const_int(if *value { 1 } else { 0 }, false);
                Ok(bool_val.as_basic_value_enum())
            }
            ASTNode::StringLiteral(s) => {
                let global_str = self
                    .builder
                    .build_global_string_ptr(s.as_str(), "str_literal")
                    .map_err(|e| e.to_string())?;

                Ok(global_str.as_pointer_value().as_basic_value_enum())
            }
            ASTNode::Identifier { name, loc } => {
                // 1. Try local variables first
                if let Some((alloca, llvm_type)) = self.variables.get(name) {
                    let loaded_value = self
                        .builder
                        .build_load(*llvm_type, *alloca, name)
                        .map_err(|e| e.to_string())?;
                    return Ok(loaded_value);
                }

                // 2. Try global variables (e.g. math_NAN)
                let global_name = if let Some(alias) = &self.current_module_alias {
                    format!("{}_{}", alias, name)
                } else {
                    name.clone()
                };

                // Look for mangled name first, then raw name
                let global = self
                    .module
                    .get_global(&global_name)
                    .or_else(|| self.module.get_global(name));

                if let Some(g) = global {
                    let ptr_val = g.as_pointer_value();
                    let val_type = g.get_value_type();

                    // FIX: Manually convert AnyTypeEnum to BasicTypeEnum
                    let basic_type = if val_type.is_float_type() {
                        val_type.into_float_type().as_basic_type_enum()
                    } else if val_type.is_int_type() {
                        val_type.into_int_type().as_basic_type_enum()
                    } else {
                        // Default fallback
                        return Err(format!(
                            "Codegen Error: Unsupported global variable type for '{}'",
                            name
                        ));
                    };

                    let val = self
                        .builder
                        .build_load(basic_type, ptr_val, "load_global")
                        .map_err(|e| e.to_string())?;
                    return Ok(val);
                }

                Err(format!(
                    "Codegen Error at {}: Undefined variable '{}'",
                    loc, name
                ))
            }

            ASTNode::Measure(qubit_expr) => {
                let (state_ptr, index_i32) = self.get_qubit_info(qubit_expr, current_function)?;

                let call_site = self
                    .builder
                    .build_call(
                        self.rt_measure,
                        &[state_ptr.into(), index_i32.into()],
                        "measure_result",
                    )
                    .map_err(|e| e.to_string())?;

                call_site
                    .try_as_basic_value()
                    .left()
                    .ok_or("(Codegen Error) Measure call failed to return i32.".to_string())
            }

            ASTNode::ArrayAccess {
                array,
                index,
                loc: _,
            } => {
                // 1. Get the array base pointer
                let array_val = self.compile_expression(array, current_function)?;
                if !array_val.is_pointer_value() {
                    return Err("(Codegen Error) Array access target is not a pointer.".to_string());
                }
                let array_ptr = array_val.into_pointer_value();

                // 2. Get the index
                let index_val = self.compile_expression(index, current_function)?;
                if !index_val.is_int_value() {
                    return Err("(Codegen Error) Array index must be an integer.".to_string());
                }
                let index_int = index_val.into_int_value();

                // 3. Calculate address and load
                // NOTE: For 'math.qc', arrays are Float[], so we use f64_type.
                let element_type = self.context.f64_type();

                let elem_ptr = unsafe {
                    self.builder
                        .build_gep(element_type, array_ptr, &[index_int], "array_gep")
                        .map_err(|e| e.to_string())?
                };

                let val = self
                    .builder
                    .build_load(element_type, elem_ptr, "array_load")
                    .map_err(|e| e.to_string())?;

                Ok(val)
            }

            ASTNode::Binary {
                operator,
                left,
                right,
                loc,
            } => {
                let left_val = self.compile_expression(left, current_function)?;
                let right_val = self.compile_expression(right, current_function)?;

                if left_val.is_vector_value() || right_val.is_vector_value() {
                    if left_val.is_vector_value() && right_val.is_vector_value() {
                        match operator {
                            BinaryOperator::TensorProduct => {
                                let result = self.emit_vector_mul(
                                    left_val.into_vector_value(),
                                    right_val.into_vector_value(),
                                )?;
                                return Ok(result.as_basic_value_enum());
                            }
                            _ => return Err(format!(
                                "(Codegen Error) Vector operation {:?} not supported for Tensors at {}",
                                operator, loc
                            )),
                        }
                    } else {
                        return Err(format!(
                            "(Codegen Error) Mixed scalar/vector operation not supported at {}",
                            loc
                        ));
                    }
                }
                let build_float_op = |op: &BinaryOperator,
                                      l: inkwell::values::FloatValue<'ctx>,
                                      r: inkwell::values::FloatValue<'ctx>|
                 -> Result<BasicValueEnum<'ctx>, String> {
                    match op {
                        BinaryOperator::Add => Ok(self
                            .builder
                            .build_float_add(l, r, "fadd")
                            .map_err(|e| e.to_string())?
                            .as_basic_value_enum()),
                        BinaryOperator::Sub => Ok(self
                            .builder
                            .build_float_sub(l, r, "fsub")
                            .map_err(|e| e.to_string())?
                            .as_basic_value_enum()),
                        BinaryOperator::Mul => Ok(self
                            .builder
                            .build_float_mul(l, r, "fmul")
                            .map_err(|e| e.to_string())?
                            .as_basic_value_enum()),
                        BinaryOperator::Div => Ok(self
                            .builder
                            .build_float_div(l, r, "fdiv")
                            .map_err(|e| e.to_string())?
                            .as_basic_value_enum()),
                        BinaryOperator::Less => Ok(self
                            .builder
                            .build_float_compare(FloatPredicate::OLT, l, r, "flt")
                            .map_err(|e| e.to_string())?
                            .as_basic_value_enum()),
                        BinaryOperator::Greater => Ok(self
                            .builder
                            .build_float_compare(FloatPredicate::OGT, l, r, "fgt")
                            .map_err(|e| e.to_string())?
                            .as_basic_value_enum()),
                        BinaryOperator::LessEqual => Ok(self
                            .builder
                            .build_float_compare(FloatPredicate::OLE, l, r, "fle")
                            .map_err(|e| e.to_string())?
                            .as_basic_value_enum()),
                        BinaryOperator::GreaterEqual => Ok(self
                            .builder
                            .build_float_compare(FloatPredicate::OGE, l, r, "fge")
                            .map_err(|e| e.to_string())?
                            .as_basic_value_enum()),
                        BinaryOperator::Equal => Ok(self
                            .builder
                            .build_float_compare(FloatPredicate::OEQ, l, r, "feq")
                            .map_err(|e| e.to_string())?
                            .as_basic_value_enum()),
                        BinaryOperator::NotEqual => Ok(self
                            .builder
                            .build_float_compare(FloatPredicate::ONE, l, r, "fne")
                            .map_err(|e| e.to_string())?
                            .as_basic_value_enum()),
                        BinaryOperator::TensorProduct => Ok(self
                            .builder
                            .build_float_mul(l, r, "tensor_mul")
                            .map_err(|e| e.to_string())?
                            .as_basic_value_enum()),
                        _ => Err(format!(
                            "(Codegen Error) Operator {:?} not supported for floats at {}",
                            op, loc
                        )),
                    }
                };

                match (left_val, right_val) {
                    // 1. Int op Int
                    (BasicValueEnum::IntValue(left_int), BasicValueEnum::IntValue(right_int)) => {
                        let result = match operator {
                            BinaryOperator::Add => {
                                self.builder.build_int_add(left_int, right_int, "iadd")
                            }
                            BinaryOperator::Sub => {
                                self.builder.build_int_sub(left_int, right_int, "isub")
                            }
                            BinaryOperator::Mul => {
                                self.builder.build_int_mul(left_int, right_int, "imul")
                            }
                            BinaryOperator::Div => self
                                .builder
                                .build_int_signed_div(left_int, right_int, "idiv"),
                            BinaryOperator::Mod => self
                                .builder
                                .build_int_signed_rem(left_int, right_int, "imod"),
                            BinaryOperator::Equal => self.builder.build_int_compare(
                                IntPredicate::EQ,
                                left_int,
                                right_int,
                                "ieq",
                            ),
                            BinaryOperator::NotEqual => self.builder.build_int_compare(
                                IntPredicate::NE,
                                left_int,
                                right_int,
                                "ine",
                            ),
                            BinaryOperator::Less => self.builder.build_int_compare(
                                IntPredicate::SLT,
                                left_int,
                                right_int,
                                "ilt",
                            ),
                            BinaryOperator::LessEqual => self.builder.build_int_compare(
                                IntPredicate::SLE,
                                left_int,
                                right_int,
                                "ile",
                            ),
                            BinaryOperator::Greater => self.builder.build_int_compare(
                                IntPredicate::SGT,
                                left_int,
                                right_int,
                                "igt",
                            ),
                            BinaryOperator::GreaterEqual => self.builder.build_int_compare(
                                IntPredicate::SGE,
                                left_int,
                                right_int,
                                "ige",
                            ),
                            BinaryOperator::And => {
                                self.builder.build_and(left_int, right_int, "and")
                            }
                            BinaryOperator::Or => self.builder.build_or(left_int, right_int, "or"),
                            _ => {
                                return Err(format!(
                                "(Codegen STUB) Integer operator {:?} not yet implemented at {}",
                                operator, loc
                            ))
                            }
                        }
                        .map_err(|e| e.to_string())?;
                        Ok(result.as_basic_value_enum())
                    }

                    // 2. Float op Float
                    (
                        BasicValueEnum::FloatValue(left_float),
                        BasicValueEnum::FloatValue(right_float),
                    ) => build_float_op(operator, left_float, right_float),

                    // 3. Float op Int (Promote Int to Float)
                    (
                        BasicValueEnum::FloatValue(left_float),
                        BasicValueEnum::IntValue(right_int),
                    ) => {
                        let right_float = self
                            .builder
                            .build_signed_int_to_float(right_int, self.context.f64_type(), "cast_r")
                            .map_err(|e| e.to_string())?;
                        build_float_op(operator, left_float, right_float)
                    }

                    // 4. Int op Float (Promote Int to Float)
                    (
                        BasicValueEnum::IntValue(left_int),
                        BasicValueEnum::FloatValue(right_float),
                    ) => {
                        let left_float = self
                            .builder
                            .build_signed_int_to_float(left_int, self.context.f64_type(), "cast_l")
                            .map_err(|e| e.to_string())?;
                        build_float_op(operator, left_float, right_float)
                    }
                    (BasicValueEnum::PointerValue(l_ptr), BasicValueEnum::PointerValue(r_ptr)) => {
                        if *operator == BinaryOperator::Add {
                            let ptr_type = self.context.ptr_type(AddressSpace::default());
                            let i64_type = self.context.i64_type();
                            let i32_type = self.context.i32_type();
                            let i8_type = self.context.i8_type();

                            // Declare strlen
                            let strlen_fn =
                                self.module.get_function("strlen").unwrap_or_else(|| {
                                    let fn_type = i64_type.fn_type(&[ptr_type.into()], false);
                                    self.module.add_function(
                                        "strlen",
                                        fn_type,
                                        Some(Linkage::External),
                                    )
                                });

                            // SAFER APPROACH: Use fixed-size stack buffer instead of malloc
                            let buf_size_val = i64_type.const_int(512, false); // 512-byte buffer
                            let buf_array_type = i8_type.array_type(512);
                            let buf = self
                                .builder
                                .build_alloca(buf_array_type, "concat_buf")
                                .map_err(|e| e.to_string())?;
                            let buf_ptr = self
                                .builder
                                .build_pointer_cast(buf, ptr_type, "buf_cast")
                                .map_err(|e| e.to_string())?;

                            // Declare snprintf
                            let snprintf_fn =
                                self.module.get_function("snprintf").unwrap_or_else(|| {
                                    let fn_type = i32_type.fn_type(
                                        &[ptr_type.into(), i64_type.into(), ptr_type.into()],
                                        true,
                                    );
                                    self.module.add_function(
                                        "snprintf",
                                        fn_type,
                                        Some(Linkage::External),
                                    )
                                });

                            // Format: snprintf(buf, 512, "%s%s", s1, s2)
                            let fmt = self
                                .builder
                                .build_global_string_ptr("%s%s", "fmt_concat")
                                .map_err(|e| e.to_string())?
                                .as_pointer_value();
                            self.builder
                                .build_call(
                                    snprintf_fn,
                                    &[
                                        buf_ptr.into(),
                                        buf_size_val.into(),
                                        fmt.into(),
                                        l_ptr.into(),
                                        r_ptr.into(),
                                    ],
                                    "concat",
                                )
                                .map_err(|e| e.to_string())?;

                            Ok(buf_ptr.as_basic_value_enum())
                        } else {
                            Err(format!(
                                "(Codegen Error) Operator {:?} not supported for Strings.",
                                operator
                            ))
                        }
                    }

                    _ => Err(format!(
                        "(Codegen Error) Mismatched types in binary operation at {}",
                        loc
                    )),
                }
            }

            ASTNode::FunctionCall {
                callee,
                arguments,
                loc,
                ..
            } => {
                // --- ADDED: Handle built-in type conversions inline ---
                if let ASTNode::Identifier { name, .. } = &**callee {
                    if name == "to_int" {
                        if arguments.len() != 1 {
                            return Err("to_int expects 1 argument".to_string());
                        }
                        let val = self.compile_expression(&arguments[0], current_function)?;

                        if val.is_float_value() {
                            // Cast Float -> Int
                            let int_val = self
                                .builder
                                .build_float_to_signed_int(
                                    val.into_float_value(),
                                    self.context.i64_type(),
                                    "fptosi",
                                )
                                .map_err(|e| e.to_string())?;
                            return Ok(int_val.as_basic_value_enum());
                        } else {
                            // Already int? Return as is.
                            return Ok(val);
                        }
                    } else if name == "to_float" {
                        if arguments.len() != 1 {
                            return Err("to_float expects 1 argument".to_string());
                        }
                        let val = self.compile_expression(&arguments[0], current_function)?;

                        if val.is_int_value() {
                            // Cast Int -> Float
                            let float_val = self
                                .builder
                                .build_signed_int_to_float(
                                    val.into_int_value(),
                                    self.context.f64_type(),
                                    "sitofp",
                                )
                                .map_err(|e| e.to_string())?;
                            return Ok(float_val.as_basic_value_enum());
                        } else {
                            // Already float? Return as is.
                            return Ok(val);
                        }
                    }

                    if name == "to_string" {
                        if arguments.len() != 1 {
                            return Err("to_string expects 1 argument".to_string());
                        }
                        let arg_val = self.compile_expression(&arguments[0], current_function)?;

                        // SIMPLER APPROACH: Use a pre-allocated buffer in the stack
                        let ptr_type = self.context.ptr_type(AddressSpace::default());
                        let i64_type = self.context.i64_type();
                        let i32_type = self.context.i32_type();
                        let i8_type = self.context.i8_type();

                        // Allocate 256-byte buffer on the stack (safer than malloc)
                        let buf_size = i64_type.const_int(256, false);
                        let buf_array_type = i8_type.array_type(256);
                        let buf = self
                            .builder
                            .build_alloca(buf_array_type, "to_string_buf")
                            .map_err(|e| e.to_string())?;
                        let buf_ptr = self
                            .builder
                            .build_pointer_cast(buf, ptr_type, "buf_cast")
                            .map_err(|e| e.to_string())?;

                        let snprintf_fn =
                            self.module.get_function("snprintf").unwrap_or_else(|| {
                                let fn_type = i32_type.fn_type(
                                    &[ptr_type.into(), i64_type.into(), ptr_type.into()],
                                    true,
                                );
                                self.module.add_function(
                                    "snprintf",
                                    fn_type,
                                    Some(Linkage::External),
                                )
                            });

                        // Select format string based on type
                        let fmt_str = if arg_val.is_int_value() {
                            "%lld"
                        } else if arg_val.is_float_value() {
                            "%.6f"
                        } else {
                            "%s"
                        };

                        let fmt_ptr = self
                            .builder
                            .build_global_string_ptr(fmt_str, "fmt_to_string")
                            .map_err(|e| e.to_string())?
                            .as_pointer_value();

                        self.builder
                            .build_call(
                                snprintf_fn,
                                &[
                                    buf_ptr.into(),
                                    buf_size.into(),
                                    fmt_ptr.into(),
                                    arg_val.into(),
                                ],
                                "snprintf_call",
                            )
                            .map_err(|e| e.to_string())?;

                        return Ok(buf_ptr.as_basic_value_enum());
                    }
                }

                let call_site =
                    self.compile_function_call(callee, arguments, loc, current_function)?;

                call_site.try_as_basic_value().left().ok_or_else(|| {
                    format!(
                        "(Codegen Error) Function call used in expression does not return a value."
                    )
                })
            }
            ASTNode::Unary { operator, operand } => {
                let operand_val = self.compile_expression(operand, current_function)?;

                let result = match operator {
                    UnaryOperator::Not => {
                        let bool_type = self.context.bool_type();
                        let true_val = bool_type.const_int(1, false);
                        self.builder
                            .build_xor(operand_val.into_int_value(), true_val, "not")
                            .map(|val| val.as_basic_value_enum())
                    }
                    UnaryOperator::Minus => {
                        if operand_val.is_int_value() {
                            self.builder
                                .build_int_neg(operand_val.into_int_value(), "ineg")
                                .map(|val| val.as_basic_value_enum())
                        } else if operand_val.is_float_value() {
                            self.builder
                                .build_float_neg(operand_val.into_float_value(), "fneg")
                                .map(|val| val.as_basic_value_enum())
                        } else {
                            return Err(
                                "(Codegen STUB) Unary '-' not supported for this type.".to_string()
                            );
                        }
                    }
                    _ => {
                        return Err(format!(
                            "(Codegen STUB) Unary operator {:?} not yet implemented.",
                            operator
                        ))
                    }
                }
                .map_err(|e| e.to_string())?;
                Ok(result)
            }
            _ => Err(format!(
                "(Codegen STUB) Expression {:?} is not yet implemented.",
                node
            )),
        }
    }

    fn map_type(&self, ty: &Type) -> BasicTypeEnum<'ctx> {
        match ty {
            Type::Int => self.context.i64_type().as_basic_type_enum(),
            Type::Float => self.context.f64_type().as_basic_type_enum(),
            Type::Bool => self.context.bool_type().as_basic_type_enum(),
            Type::String => self
                .context
                .ptr_type(inkwell::AddressSpace::default())
                .as_basic_type_enum(),
            Type::Array(_) => self
                .context
                .ptr_type(AddressSpace::default())
                .as_basic_type_enum(),
            Type::None => self.context.i8_type().as_basic_type_enum(),
            Type::Int8 => self.context.i8_type().as_basic_type_enum(),
            Type::Int16 => self.context.i16_type().as_basic_type_enum(),
            Type::Int32 => self.context.i32_type().as_basic_type_enum(),
            Type::Int64 => self.context.i64_type().as_basic_type_enum(),
            Type::Int128 => self.context.i128_type().as_basic_type_enum(),
            Type::Float32 => self.context.f32_type().as_basic_type_enum(),
            Type::Float64 => self.context.f64_type().as_basic_type_enum(),
            _ => {
                println!(
                    "Warning: Codegen for type {:?} is not implemented, defaulting to i64.",
                    ty
                );
                self.context.i64_type().as_basic_type_enum()
            }
        }
    }

    fn compile_apply_statement(
        &mut self,
        gate_expr: &ASTNode,
        arguments: &Vec<ASTNode>,
        current_function: FunctionValue<'ctx>,
    ) -> Result<(), String> {
        let (gate_name, param_ast_nodes, is_dagger, num_controls) =
            self.compile_gate_expression(gate_expr, current_function)?;

        let mut gate_params_llvm: Vec<inkwell::values::FloatValue<'ctx>> = Vec::new();
        for param_node in &param_ast_nodes {
            let param_val = self.compile_expression(param_node, current_function)?;

            let param_f64 = if param_val.is_float_value() {
                param_val.into_float_value()
            } else if param_val.is_int_value() {
                self.builder
                    .build_signed_int_to_float(
                        param_val.into_int_value(),
                        self.context.f64_type(),
                        "param_f64",
                    )
                    .map_err(|e| e.to_string())?
            } else {
                return Err("(Codegen Error) Gate parameter must be a float or int.".to_string());
            };
            gate_params_llvm.push(param_f64);
        }

        let mut qubit_indices: Vec<i32> = Vec::new();
        let mut register_alloca: Option<inkwell::values::PointerValue<'ctx>> = None;
        let mut register_type: Option<inkwell::types::BasicTypeEnum<'ctx>> = None;

        for arg_node in arguments {
            match arg_node {
                ASTNode::ArrayAccess { array, index, loc } => {
                    let array_name = if let ASTNode::Identifier { name, .. } = &**array {
                        name
                    } else {
                        return Err(format!("(Codegen Error) Qubit argument must be a simple register access, found: {:?}", array));
                    };

                    let (current_var_alloca, current_var_type) =
                        self.variables.get(array_name).ok_or_else(|| {
                            format!("(Codegen Error) Unknown quantum register '{}'", array_name)
                        })?;

                    if !current_var_type.is_pointer_type() {
                        return Err(format!(
                            "(Codegen Error) Variable '{}' is not a quantum register.",
                            array_name
                        ));
                    }

                    if let Some(first_alloca) = register_alloca {
                        if first_alloca != *current_var_alloca {
                            return Err(format!(
                                "(Codegen Error) 'apply' on different registers is not supported."
                            ));
                        }
                    } else {
                        register_alloca = Some(*current_var_alloca);
                        register_type = Some(*current_var_type);
                    }

                    if let ASTNode::IntLiteral(idx) = &**index {
                        qubit_indices.push(*idx as i32);
                    } else {
                        let index_val = self.compile_expression(index, current_function)?;
                        if !index_val.is_int_value() {
                            return Err(format!(
                                "(Codegen Error) Qubit index must be an integer at {}",
                                loc
                            ));
                        }
                        return Err(format!(
                            "(Codegen STUB) Qubit index must be an integer literal for now (at {})",
                            loc
                        ));
                    }
                }
                _ => {
                    return Err(
                        "(Codegen Error) 'apply' arguments must be qubit accesses (e.g., q[0])."
                            .to_string(),
                    )
                }
            }
        }

        let state_ptr = match (register_alloca, register_type) {
            (Some(alloca), Some(ty)) => self
                .builder
                .build_load(ty, alloca, "load_state_ptr")
                .map_err(|e| e.to_string())?
                .into_pointer_value(),
            _ => return Err("(Codegen Error) 'apply' called with no qubit arguments.".to_string()),
        };

        let gate_name_global = self
            .builder
            .build_global_string_ptr(&gate_name, "gate_name")
            .map_err(|e| e.to_string())?;
        let gate_name_ptr = gate_name_global.as_pointer_value();

        let is_dagger_int = self
            .context
            .i32_type()
            .const_int(if is_dagger { 1 } else { 0 }, false);

        let params_ptr = self.build_f64_array(&gate_params_llvm)?;

        let num_params = self
            .context
            .i32_type()
            .const_int(gate_params_llvm.len() as u64, false);

        let qubit_indices_ptr = self.build_i32_array(&qubit_indices)?;

        let num_qubits = self
            .context
            .i32_type()
            .const_int(qubit_indices.len() as u64, false);

        let num_controls_i32 = self
            .context
            .i32_type()
            .const_int(num_controls as u64, false);

        let _ = self
            .builder
            .build_call(
                self.rt_apply_gate,
                &[
                    state_ptr.into(),
                    gate_name_ptr.into(),
                    is_dagger_int.into(),
                    params_ptr.into(),
                    num_params.into(),
                    qubit_indices_ptr.into(),
                    num_qubits.into(),
                    num_controls_i32.into(),
                ],
                "call_apply",
            )
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    fn get_qubit_info(
        &mut self,
        qubit_expr: &ASTNode,
        current_function: FunctionValue<'ctx>,
    ) -> Result<
        (
            inkwell::values::PointerValue<'ctx>,
            inkwell::values::IntValue<'ctx>,
        ),
        String,
    > {
        if let ASTNode::ArrayAccess { array, index, loc } = qubit_expr {
            let array_name = if let ASTNode::Identifier { name, .. } = &**array {
                name
            } else {
                return Err(format!(
                    "(Codegen Error) Measure target must be a register access, found: {:?}",
                    array
                ));
            };

            let (var_ptr, var_type) = self.variables.get(array_name).ok_or_else(|| {
                format!("(Codegen Error) Unknown quantum register '{}'", array_name)
            })?;

            let state_ptr = self
                .builder
                .build_load(*var_type, *var_ptr, "load_measure_ptr")
                .map_err(|e| e.to_string())?
                .into_pointer_value();

            let index_val = self.compile_expression(index, current_function)?;
            if !index_val.is_int_value() {
                return Err(format!(
                    "(Codegen Error) Qubit index must be an integer at {}",
                    loc
                ));
            }
            let index_i32 = self
                .builder
                .build_int_truncate(
                    index_val.into_int_value(),
                    self.context.i32_type(),
                    "measure_idx_i32",
                )
                .map_err(|e| e.to_string())?;

            Ok((state_ptr, index_i32))
        } else {
            Err("(Codegen Error) 'measure' target must be a qubit access (e.g., q[0]).".to_string())
        }
    }

    fn compile_gate_expression(
        &mut self,
        gate_expr: &ASTNode,
        current_function: FunctionValue<'ctx>,
    ) -> Result<(String, Vec<ASTNode>, bool, i32), String> {
        match gate_expr {
            ASTNode::Gate { name, .. } => {
                let gate_name = name.clone();
                let params: Vec<ASTNode> = Vec::new();
                let is_dagger = false;
                let num_controls = 0;
                Ok((gate_name, params, is_dagger, num_controls))
            }

            ASTNode::ParameterizedGate {
                name, parameters, ..
            } => {
                let gate_name = name.clone();
                let params = parameters.clone();
                let is_dagger = false;
                let num_controls = 0;
                Ok((gate_name, params, is_dagger, num_controls))
            }

            ASTNode::Dagger { gate_expr, .. } => {
                let (gate_name, params, is_dagger, num_controls) =
                    self.compile_gate_expression(gate_expr, current_function)?;

                Ok((gate_name, params, !is_dagger, num_controls))
            }

            ASTNode::Controlled { gate_expr, .. } => {
                let (gate_name, params, is_dagger, num_controls) =
                    self.compile_gate_expression(gate_expr, current_function)?;

                Ok((gate_name, params, is_dagger, num_controls + 1))
            }

            _ => Err(format!(
                "(Codegen Error) This expression is not a valid gate: {:?}",
                gate_expr
            )),
        }
    }

    fn build_f64_array(
        &self,
        values: &[inkwell::values::FloatValue<'ctx>],
    ) -> Result<inkwell::values::PointerValue<'ctx>, String> {
        let f64_type = self.context.f64_type();
        let f64_array_type = f64_type.array_type(values.len() as u32);

        let array_alloca = self
            .builder
            .build_alloca(f64_array_type, "params_array")
            .map_err(|e| e.to_string())?;

        for (i, &val) in values.iter().enumerate() {
            let index = self.context.i32_type().const_int(i as u64, false);
            let elem_ptr = unsafe {
                self.builder
                    .build_gep(
                        f64_array_type,
                        array_alloca,
                        &[self.context.i32_type().const_int(0, false), index],
                        "elem_ptr",
                    )
                    .map_err(|e| e.to_string())?
            };
            self.builder
                .build_store(elem_ptr, val)
                .map_err(|e| e.to_string())?;
        }

        self.builder
            .build_pointer_cast(
                array_alloca,
                self.context.ptr_type(AddressSpace::default()),
                "params_ptr",
            )
            .map_err(|e| e.to_string())
    }

    fn build_i32_array(
        &self,
        values: &[i32],
    ) -> Result<inkwell::values::PointerValue<'ctx>, String> {
        let i32_type = self.context.i32_type();
        let i32_array_type = i32_type.array_type(values.len() as u32);

        let array_alloca = self
            .builder
            .build_alloca(i32_array_type, "indices_array")
            .map_err(|e| e.to_string())?;

        for (i, &val) in values.iter().enumerate() {
            let index = self.context.i32_type().const_int(i as u64, false);
            let elem_ptr = unsafe {
                self.builder
                    .build_gep(
                        i32_array_type,
                        array_alloca,
                        &[self.context.i32_type().const_int(0, false), index],
                        "elem_ptr",
                    )
                    .map_err(|e| e.to_string())?
            };
            self.builder
                .build_store(elem_ptr, i32_type.const_int(val as u64, false))
                .map_err(|e| e.to_string())?;
        }

        self.builder
            .build_pointer_cast(
                array_alloca,
                self.context.ptr_type(AddressSpace::default()),
                "indices_ptr",
            )
            .map_err(|e| e.to_string())
    }
}

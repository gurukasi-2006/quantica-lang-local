// src/runtime.rs

use crate::environment::{Environment, RuntimeValue};
use crate::evaluator::Evaluator;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::os::raw::c_int;
use libc;
use std::rc::Rc;
use std::cell::RefCell;
use crate::environment::GateDefinition;
use std::slice;

/// The C-compatible, opaque pointer to our quantum state.
/// We are just hiding the complex Rust type `Rc<RefCell<HashMap<usize, (f64, f64)>>>`
/// behind a raw pointer.
type StatePtr = *mut c_void;

/// Creates a new, zero-initialized quantum state.
/// This function is called by the 'quantum' keyword in the compiler.
#[no_mangle]
pub extern "C" fn quantica_rt_new_state(num_qubits: c_int) -> StatePtr {
    // 1. Create the sparse state map (HashMap)
    let mut state_map: HashMap<usize, (f64, f64)> = HashMap::new();
    state_map.insert(0, (1.0, 0.0)); // Initialize to |0> state

    // 2. Wrap it in our standard RuntimeValue
    let register = RuntimeValue::QuantumRegister {
        state: Rc::new(RefCell::new(state_map)),
        size: num_qubits as usize,
    };

    // 3. Put it on the heap (Box) and cast it to a raw, opaque pointer
    let boxed_register = Box::new(register);
    Box::into_raw(boxed_register) as StatePtr
}

#[no_mangle]
pub extern "C" fn quantica_rt_measure(state_ptr: StatePtr, qubit_index: c_int) -> c_int {
    if state_ptr.is_null() {
        eprintln!("(Runtime Error) quantica_rt_measure called with null pointer.");
        return -1; // Indicate error
    }

    let register = unsafe { &*(state_ptr as *mut RuntimeValue) };

    match register {
        RuntimeValue::QuantumRegister { state, size } => {
            // 1. Create a temporary Qubit handle for the Evaluator
            let qubit_handle = RuntimeValue::Qubit {
                state: state.clone(),
                index: qubit_index as usize,
                size: *size,
            };

            // 2. Call the existing Evaluator function
            let result = Evaluator::builtin_measure(vec![qubit_handle]);

            // 3. Unwrap the result (IntLiteral)
            match result {
                Ok(RuntimeValue::Int(r)) => r as c_int,
                Ok(_) => -2, // Unexpected return type
                Err(e) => {
                    eprintln!("(Runtime Error) Measurement failed: {}", e);
                    -1
                }
            }
        }
        _ => {
            eprintln!("(Runtime Error) Invalid state pointer passed to measure.");
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn quantica_rt_device_alloc(size_bytes: usize) -> *mut c_void {
    // In a production runtime, this would call Vulkan's vkAllocateMemory or XLA's device allocation.
    // Here, we simulate by allocating CPU memory and flagging it as device memory.
    
    // Safety: Allocating raw memory using the standard library's C allocation equivalent
    let ptr = unsafe { libc::malloc(size_bytes) }; 
    
    if ptr.is_null() {
        eprintln!("(Runtime Error) Device memory allocation failed for {} bytes.", size_bytes);
    }
    
    ptr
}

#[no_mangle]
pub extern "C" fn quantica_rt_device_free(device_ptr: *mut c_void) {
    if device_ptr.is_null() {
        return;
    }
    // Safety: Freeing raw memory allocated by quantica_rt_device_alloc
    unsafe {
        libc::free(device_ptr);
    }
}

#[no_mangle]
pub extern "C" fn quantica_rt_htod_transfer(
    host_ptr: *const c_void, 
    device_ptr: *mut c_void, 
    size_bytes: usize
) -> c_int {
    // In a production system, this calls vkQueueSubmit or XLA transfer API.
    // We simulate a successful memory copy.
    unsafe {
        std::ptr::copy_nonoverlapping(host_ptr, device_ptr, size_bytes);
    }
    // Return 0 for success
    0
}

#[no_mangle]
pub extern "C" fn quantica_rt_dtoh_transfer(
    host_ptr: *mut c_void, 
    device_ptr: *const c_void, 
    size_bytes: usize
) -> c_int {
    // We simulate a successful memory copy.
    unsafe {
        std::ptr::copy_nonoverlapping(device_ptr, host_ptr, size_bytes);
    }
    // Return 0 for success
    0
}

/// Frees the memory associated with a quantum state.
#[no_mangle]
pub extern "C" fn quantica_rt_free_state(state_ptr: StatePtr) {
    if state_ptr.is_null() {
        return;
    }
    // Take the raw pointer and turn it back into a Box,
    // which will then be immediately dropped, freeing the memory.
    unsafe {
        let _=Box::from_raw(state_ptr as *mut RuntimeValue);
    }
}

/// Calls the 'debug_state' logic on a quantum state.
#[no_mangle]
pub extern "C" fn quantica_rt_debug_state(state_ptr: StatePtr) {
    if state_ptr.is_null() {
        println!("(Runtime Error) quantica_rt_debug_state called with null pointer.");
        return;
    }

    // 1. "Unwrap" the opaque pointer.
    // We *must* use unsafe here because we are trusting the caller
    // (our compiler) to pass a valid pointer.
    let register = unsafe {
        // We cast it back to a reference to our RuntimeValue
        &*(state_ptr as *mut RuntimeValue)
    };
    
    // 2. Check the type
    if let RuntimeValue::QuantumRegister { state, size } = register {
        // 3. Call our existing Rust logic!
        println!("(Quantum Runtime) Debugging state ({} qubits):", size);
        Evaluator::print_quantum_state(state, *size, 10); // 10 is max_entries
    } else {
        println!("(Runtime Error) Invalid state pointer passed to debug_state.");
    }
}
#[no_mangle]
pub extern "C" fn quantica_rt_apply_gate(
    state_ptr: StatePtr,
    gate_name_ptr: *const c_char,
    is_dagger_int: c_int,
    params_ptr: *const f64,
    num_params: c_int,
    qubit_indices_ptr: *const c_int,
    num_qubits: c_int,
    num_controls: c_int,
) -> c_int {
    // We use a helper function to contain the unsafe logic
    // and return a 0 for success, 1 for failure.
    match unsafe { apply_gate_unsafe(
        state_ptr,
        gate_name_ptr,
        is_dagger_int,
        params_ptr,
        num_params,
        qubit_indices_ptr,
        num_qubits,
        num_controls
    ) } {
        Ok(_) => 0, // Success
        Err(e) => {
            eprintln!("(Quantum Runtime Error) {}", e);
            1 // Failure
        }
    }
}

/// Unsafe helper to handle the actual gate application logic.
unsafe fn apply_gate_unsafe(
    state_ptr: StatePtr,
    gate_name_ptr: *const c_char,
    is_dagger_int: c_int,
    params_ptr: *const f64,
    num_params: c_int,
    qubit_indices_ptr: *const c_int,
    num_qubits: c_int,
    num_controls: c_int,
) -> Result<RuntimeValue, String> {

    // 1. Unwrap State Ptr
    if state_ptr.is_null() {
        return Err("State pointer is null.".to_string());
    }
    let register = &*(state_ptr as *mut RuntimeValue);
    let (state_rc, reg_size) = match register {
        RuntimeValue::QuantumRegister { state, size } => (state.clone(), *size),
        _ => return Err("Invalid state pointer passed to apply_gate.".to_string()),
    };

    // 2. Unwrap Gate Name
    let gate_name_cstr = CStr::from_ptr(gate_name_ptr);
    let gate_name = gate_name_cstr.to_str().unwrap_or("").to_string();

    // 3. Unwrap Dagger Flag
    let is_dagger = is_dagger_int != 0;

    // 4. Unwrap Parameters (from C array to Rust Vec)
    let params: Vec<f64> = if params_ptr.is_null() {
        Vec::new()
    } else {
        slice::from_raw_parts(params_ptr, num_params as usize).to_vec()
    };

    // 5. Unwrap Qubit Indices (from C array to Rust Vec)
    let qubit_indices: Vec<usize> = if qubit_indices_ptr.is_null() {
        Vec::new()
    } else {
        slice::from_raw_parts(qubit_indices_ptr, num_qubits as usize)
            .iter().map(|&x| x as usize).collect()
    };

    // 6. Split indices into controls and targets
    if (num_controls as usize) > qubit_indices.len() {
        return Err("More controls specified than total qubits.".to_string());
    }
    let (controls, targets) = qubit_indices.split_at(num_controls as usize);

    // 7. Build the GateDefinition
    let gate_def = GateDefinition {
        name: gate_name,
        params: params,
        controls: controls.to_vec(),
        targets: targets.to_vec(),
        register_size: reg_size,
        state_rc: state_rc,
    };

    // 8. Call the existing Evaluator logic!
    Evaluator::apply_multi_controlled_gate(gate_def, is_dagger)
}
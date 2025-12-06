// src/quantum_backend/native_simulator.rs

use super::{HardwareCircuit, QuantumBackend, QuantumConfig, QuantumResult};
use std::collections::HashMap;
use rayon::prelude::*;

/// Complex number representation
#[derive(Clone, Copy, Debug)]
struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    fn new(real: f64, imag: f64) -> Self {
        Complex { real, imag }
    }

    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }

    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }

    fn i() -> Self {
        Complex::new(0.0, 1.0)
    }

    fn magnitude_squared(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }

    fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    fn conj(&self) -> Complex {
        Complex::new(self.real, -self.imag)
    }

    fn add(&self, other: &Complex) -> Complex {
        Complex::new(self.real + other.real, self.imag + other.imag)
    }

    fn mul(&self, other: &Complex) -> Complex {
        Complex::new(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )
    }

    fn scale(&self, scalar: f64) -> Complex {
        Complex::new(self.real * scalar, self.imag * scalar)
    }
}

/// Native quantum state vector simulator
pub struct NativeSimulator {
    state: Vec<Complex>,
    num_qubits: usize,
}

impl NativeSimulator {
    pub fn new(num_qubits: usize) -> Self {
        let size = 1 << num_qubits; // 2^num_qubits
        let mut state = vec![Complex::zero(); size];
        state[0] = Complex::one(); // Initialize to |0...0⟩

        NativeSimulator { state, num_qubits }
    }

    /// Get the probability of measuring a specific state
    fn get_probability(&self, index: usize) -> f64 {
        if index < self.state.len() {
            self.state[index].magnitude_squared()
        } else {
            0.0
        }
    }

    /// Apply a single-qubit gate
    fn apply_single_qubit_gate(&mut self, qubit: usize, matrix: [[Complex; 2]; 2]) {
        let size = self.state.len();
        let mask = 1 << qubit;

        // Parallel application for better performance
        let mut new_state = self.state.clone();

        (0..size)
            .into_par_iter()
            .filter(|&i| (i & mask) == 0)
            .for_each(|i| {
                let i0 = i;
                let i1 = i | mask;

                let amp0 = self.state[i0];
                let amp1 = self.state[i1];

                let new_amp0 = matrix[0][0].mul(&amp0).add(&matrix[0][1].mul(&amp1));
                let new_amp1 = matrix[1][0].mul(&amp0).add(&matrix[1][1].mul(&amp1));

                unsafe {
                    let ptr = new_state.as_ptr() as *mut Complex;
                    *ptr.add(i0) = new_amp0;
                    *ptr.add(i1) = new_amp1;
                }
            });

        self.state = new_state;
    }

    /// Apply a two-qubit gate
    fn apply_two_qubit_gate(
        &mut self,
        control: usize,
        target: usize,
        matrix: [[Complex; 4]; 4],
    ) {
        let size = self.state.len();
        let control_mask = 1 << control;
        let target_mask = 1 << target;

        let mut new_state = self.state.clone();

        (0..size)
            .into_par_iter()
            .filter(|&i| (i & control_mask) == 0 && (i & target_mask) == 0)
            .for_each(|i| {
                let i00 = i;
                let i01 = i | target_mask;
                let i10 = i | control_mask;
                let i11 = i | control_mask | target_mask;

                let amps = [
                    self.state[i00],
                    self.state[i01],
                    self.state[i10],
                    self.state[i11],
                ];

                let mut new_amps = [Complex::zero(); 4];

                for row in 0..4 {
                    for col in 0..4 {
                        new_amps[row] = new_amps[row].add(&matrix[row][col].mul(&amps[col]));
                    }
                }

                unsafe {
                    let ptr = new_state.as_ptr() as *mut Complex;
                    *ptr.add(i00) = new_amps[0];
                    *ptr.add(i01) = new_amps[1];
                    *ptr.add(i10) = new_amps[2];
                    *ptr.add(i11) = new_amps[3];
                }
            });

        self.state = new_state;
    }

    /// Simulate measurements and return counts
    fn measure(&self, shots: u32) -> HashMap<String, u32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut counts = HashMap::new();

        // Build cumulative probability distribution
        let mut cumulative_probs = Vec::with_capacity(self.state.len());
        let mut sum = 0.0;
        for amp in &self.state {
            sum += amp.magnitude_squared();
            cumulative_probs.push(sum);
        }

        // Perform measurements
        for _ in 0..shots {
            let rand_val: f64 = rng.gen();
            let index = cumulative_probs
                .binary_search_by(|&p| p.partial_cmp(&rand_val).unwrap())
                .unwrap_or_else(|i| i);

            let bitstring = format!("{:0width$b}", index, width = self.num_qubits);
            *counts.entry(bitstring).or_insert(0) += 1;
        }

        counts
    }

    /// Apply a gate based on name and parameters
    fn apply_gate(&mut self, gate_name: &str, qubits: &[usize], params: &[f64]) -> Result<(), String> {
        match gate_name {
            "hadamard" | "h" => {
                if qubits.len() != 1 {
                    return Err("Hadamard gate requires 1 qubit".to_string());
                }
                let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
                let h_matrix = [
                    [Complex::new(inv_sqrt2, 0.0), Complex::new(inv_sqrt2, 0.0)],
                    [Complex::new(inv_sqrt2, 0.0), Complex::new(-inv_sqrt2, 0.0)],
                ];
                self.apply_single_qubit_gate(qubits[0], h_matrix);
            }
            "x" => {
                if qubits.len() != 1 {
                    return Err("X gate requires 1 qubit".to_string());
                }
                let x_matrix = [
                    [Complex::zero(), Complex::one()],
                    [Complex::one(), Complex::zero()],
                ];
                self.apply_single_qubit_gate(qubits[0], x_matrix);
            }
            "y" => {
                if qubits.len() != 1 {
                    return Err("Y gate requires 1 qubit".to_string());
                }
                let y_matrix = [
                    [Complex::zero(), Complex::new(0.0, -1.0)],
                    [Complex::new(0.0, 1.0), Complex::zero()],
                ];
                self.apply_single_qubit_gate(qubits[0], y_matrix);
            }
            "z" => {
                if qubits.len() != 1 {
                    return Err("Z gate requires 1 qubit".to_string());
                }
                let z_matrix = [
                    [Complex::one(), Complex::zero()],
                    [Complex::zero(), Complex::new(-1.0, 0.0)],
                ];
                self.apply_single_qubit_gate(qubits[0], z_matrix);
            }
            "s" => {
                if qubits.len() != 1 {
                    return Err("S gate requires 1 qubit".to_string());
                }
                let s_matrix = [
                    [Complex::one(), Complex::zero()],
                    [Complex::zero(), Complex::i()],
                ];
                self.apply_single_qubit_gate(qubits[0], s_matrix);
            }
            "t" => {
                if qubits.len() != 1 {
                    return Err("T gate requires 1 qubit".to_string());
                }
                let phase = std::f64::consts::PI / 4.0;
                let t_matrix = [
                    [Complex::one(), Complex::zero()],
                    [Complex::zero(), Complex::new(phase.cos(), phase.sin())],
                ];
                self.apply_single_qubit_gate(qubits[0], t_matrix);
            }
            "rx" => {
                if qubits.len() != 1 || params.is_empty() {
                    return Err("RX gate requires 1 qubit and 1 parameter".to_string());
                }
                let theta = params[0];
                let cos = (theta / 2.0).cos();
                let sin = (theta / 2.0).sin();
                let rx_matrix = [
                    [Complex::new(cos, 0.0), Complex::new(0.0, -sin)],
                    [Complex::new(0.0, -sin), Complex::new(cos, 0.0)],
                ];
                self.apply_single_qubit_gate(qubits[0], rx_matrix);
            }
            "ry" => {
                if qubits.len() != 1 || params.is_empty() {
                    return Err("RY gate requires 1 qubit and 1 parameter".to_string());
                }
                let theta = params[0];
                let cos = (theta / 2.0).cos();
                let sin = (theta / 2.0).sin();
                let ry_matrix = [
                    [Complex::new(cos, 0.0), Complex::new(-sin, 0.0)],
                    [Complex::new(sin, 0.0), Complex::new(cos, 0.0)],
                ];
                self.apply_single_qubit_gate(qubits[0], ry_matrix);
            }
            "rz" => {
                if qubits.len() != 1 || params.is_empty() {
                    return Err("RZ gate requires 1 qubit and 1 parameter".to_string());
                }
                let theta = params[0];
                let phase_neg = Complex::new((theta / 2.0).cos(), -(theta / 2.0).sin());
                let phase_pos = Complex::new((theta / 2.0).cos(), (theta / 2.0).sin());
                let rz_matrix = [
                    [phase_neg, Complex::zero()],
                    [Complex::zero(), phase_pos],
                ];
                self.apply_single_qubit_gate(qubits[0], rz_matrix);
            }
            "cnot" | "cx" => {
                if qubits.len() != 2 {
                    return Err("CNOT gate requires 2 qubits".to_string());
                }
                let cnot_matrix = [
                    [Complex::one(), Complex::zero(), Complex::zero(), Complex::zero()],
                    [Complex::zero(), Complex::one(), Complex::zero(), Complex::zero()],
                    [Complex::zero(), Complex::zero(), Complex::zero(), Complex::one()],
                    [Complex::zero(), Complex::zero(), Complex::one(), Complex::zero()],
                ];
                self.apply_two_qubit_gate(qubits[0], qubits[1], cnot_matrix);
            }
            "cz" => {
                if qubits.len() != 2 {
                    return Err("CZ gate requires 2 qubits".to_string());
                }
                let cz_matrix = [
                    [Complex::one(), Complex::zero(), Complex::zero(), Complex::zero()],
                    [Complex::zero(), Complex::one(), Complex::zero(), Complex::zero()],
                    [Complex::zero(), Complex::zero(), Complex::one(), Complex::zero()],
                    [Complex::zero(), Complex::zero(), Complex::zero(), Complex::new(-1.0, 0.0)],
                ];
                self.apply_two_qubit_gate(qubits[0], qubits[1], cz_matrix);
            }
            "swap" => {
                if qubits.len() != 2 {
                    return Err("SWAP gate requires 2 qubits".to_string());
                }
                let swap_matrix = [
                    [Complex::one(), Complex::zero(), Complex::zero(), Complex::zero()],
                    [Complex::zero(), Complex::zero(), Complex::one(), Complex::zero()],
                    [Complex::zero(), Complex::one(), Complex::zero(), Complex::zero()],
                    [Complex::zero(), Complex::zero(), Complex::zero(), Complex::one()],
                ];
                self.apply_two_qubit_gate(qubits[0], qubits[1], swap_matrix);
            }
            _ => return Err(format!("Unknown gate: {}", gate_name)),
        }
        Ok(())
    }
}

/// Native Backend implementation
pub struct NativeBackend;

impl NativeBackend {
    pub fn new() -> Self {
        NativeBackend
    }
}

impl QuantumBackend for NativeBackend {
    fn execute(
        &self,
        circuit: &HardwareCircuit,
        config: &QuantumConfig,
    ) -> Result<QuantumResult, String> {
        // Initialize simulator
        let mut simulator = NativeSimulator::new(circuit.num_qubits);

        // Apply gates
        for gate in &circuit.gates {
            simulator.apply_gate(&gate.name, &gate.qubits, &gate.params)?;
        }

        // Measure
        let counts = simulator.measure(config.shots);

        Ok(QuantumResult {
            counts,
            shots: config.shots,
            success: true,
            error_message: None,
        })
    }

    fn is_available(&self) -> bool {
        true // Native backend is always available
    }

    fn available_devices(&self) -> Vec<String> {
        vec![
            "native_simulator".to_string(),
            "native_statevector".to_string(),
        ]
    }

    fn optimize_circuit(&self, circuit: &HardwareCircuit) -> HardwareCircuit {
        // Basic optimization: merge consecutive single-qubit gates
        let mut optimized_gates = Vec::new();

        for gate in &circuit.gates {
            // Simple pass-through for now
            // You can add optimization logic here:
            // - Gate fusion
            // - Commutation rules
            // - Cancellation of inverse gates
            optimized_gates.push(gate.clone());
        }

        HardwareCircuit {
            num_qubits: circuit.num_qubits,
            gates: optimized_gates,
            measurements: circuit.measurements.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard_gate() {
        let mut sim = NativeSimulator::new(1);
        sim.apply_gate("h", &[0], &[]).unwrap();

        let prob0 = sim.get_probability(0);
        let prob1 = sim.get_probability(1);

        assert!((prob0 - 0.5).abs() < 1e-10);
        assert!((prob1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cnot_gate() {
        let mut sim = NativeSimulator::new(2);
        sim.apply_gate("x", &[0], &[]).unwrap();
        sim.apply_gate("cnot", &[0, 1], &[]).unwrap();

        // Should be in |11⟩ state
        let prob3 = sim.get_probability(3); // Binary: 11
        assert!((prob3 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state() {
        let mut sim = NativeSimulator::new(2);
        sim.apply_gate("h", &[0], &[]).unwrap();
        sim.apply_gate("cnot", &[0, 1], &[]).unwrap();

        // Should have equal probability for |00⟩ and |11⟩
        let prob0 = sim.get_probability(0);
        let prob3 = sim.get_probability(3);

        assert!((prob0 - 0.5).abs() < 1e-10);
        assert!((prob3 - 0.5).abs() < 1e-10);
    }
}
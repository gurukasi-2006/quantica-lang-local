// src/quantum_backend/native_simulator.rs

use super::{HardwareCircuit, QuantumBackend, QuantumConfig, QuantumResult};
use std::collections::HashMap;
use rayon::prelude::*;


#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Complex {
    pub fn new(real: f64, imag: f64) -> Self { Complex { real, imag } }
    pub fn zero() -> Self { Complex::new(0.0, 0.0) }
    pub fn one() -> Self { Complex::new(1.0, 0.0) }
    pub fn i() -> Self { Complex::new(0.0, 1.0) }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Complex::new(r * theta.cos(), r * theta.sin())
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }

    pub fn scale(&self, scalar: f64) -> Complex {
        Complex::new(self.real * scalar, self.imag * scalar)
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, other: Self) -> Self { Complex::new(self.real + other.real, self.imag + other.imag) }
}
impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Complex::new(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )
    }
}


#[derive(Clone)]
struct StateGroup {
    qubits: Vec<usize>,
    state: Vec<Complex>,
}

impl StateGroup {
    fn new(qubit_id: usize) -> Self {
        StateGroup {
            qubits: vec![qubit_id],
            state: vec![Complex::one(), Complex::zero()],
        }
    }

    fn get_shift(&self, global_id: usize) -> usize {
        let idx = self.qubits.iter().position(|&id| id == global_id).unwrap();
        self.qubits.len() - 1 - idx
    }

    fn merge(&mut self, other: StateGroup) {
        let mut new_state = Vec::with_capacity(self.state.len() * other.state.len());
        for amp_a in &self.state {
            for amp_b in &other.state {
                new_state.push(*amp_a * *amp_b);
            }
        }
        self.qubits.extend(other.qubits);
        self.state = new_state;
    }
}


pub struct NativeSimulator {
    groups: Vec<StateGroup>,
    qubit_map: HashMap<usize, usize>,
}

impl NativeSimulator {
    pub fn new(_total_qubits: usize) -> Self {
        NativeSimulator {
            groups: Vec::new(),
            qubit_map: HashMap::new(),
        }
    }

    fn ensure_qubit(&mut self, q: usize) {
        if !self.qubit_map.contains_key(&q) {
            let idx = self.groups.len();
            self.groups.push(StateGroup::new(q));
            self.qubit_map.insert(q, idx);
        }
    }

    fn ensure_entangled(&mut self, q1: usize, q2: usize) {
        self.ensure_qubit(q1);
        self.ensure_qubit(q2);

        let g1 = self.qubit_map[&q1];
        let g2 = self.qubit_map[&q2];

        if g1 != g2 {
            let (dest_idx, src_idx) = if g1 < g2 { (g1, g2) } else { (g2, g1) };
            let src_group = self.groups.remove(src_idx);

            for (_, grp_idx) in self.qubit_map.iter_mut() {
                if *grp_idx > src_idx { *grp_idx -= 1; }
            }

            for q in &src_group.qubits {
                self.qubit_map.insert(*q, dest_idx);
            }

            self.groups[dest_idx].merge(src_group);
        }
    }


    pub fn apply_gate(&mut self, name: &str, qubits: &[usize], params: &[f64], is_dagger: bool) -> Result<(), String> {
        match qubits.len() {
            1 => self.ensure_qubit(qubits[0]),
            2 => self.ensure_entangled(qubits[0], qubits[1]),
            3 => {
                self.ensure_entangled(qubits[0], qubits[1]);
                self.ensure_entangled(qubits[0], qubits[2]);
            }
            _ => return Err("Gates with >3 qubits not supported natively".to_string()),
        }

        let mat = Self::get_matrix(name, params, is_dagger)?;

        let group_idx = self.qubit_map[&qubits[0]];
        let group = &mut self.groups[group_idx];

        // 3. Apply
        if qubits.len() == 1 {
            let shift = group.get_shift(qubits[0]);
            Self::apply_1q(group, shift, mat)
        } else if qubits.len() == 2 {
            let s1 = group.get_shift(qubits[0]);
            let s2 = group.get_shift(qubits[1]);
            Self::apply_2q(group, s1, s2, mat)
        } else if qubits.len() == 3 {
             let s0 = group.get_shift(qubits[0]);
             let s1 = group.get_shift(qubits[1]);
             let s2 = group.get_shift(qubits[2]);
             Self::apply_3q_gate(group, s0, s1, s2, name, params, is_dagger)
        } else {
            Err("Gate logic error".to_string())
        }
    }

    pub fn measure_single(&mut self, qubit: usize) -> u32 {
        self.ensure_qubit(qubit);
        let group_idx = self.qubit_map[&qubit];
        let group = &mut self.groups[group_idx];

        let shift = group.get_shift(qubit);
        let mask = 1 << shift;

        let mut prob0 = 0.0;
        for (i, amp) in group.state.iter().enumerate() {
            if (i & mask) == 0 {
                prob0 += amp.magnitude_squared();
            }
        }

        let mut rng = rand::thread_rng();
        use rand::Rng;
        let result = if rng.gen::<f64>() < prob0 { 0 } else { 1 };

        let norm = if result == 0 {
            if prob0 > 0.0 { 1.0 / prob0.sqrt() } else { 0.0 }
        } else {
            if prob0 < 1.0 { 1.0 / (1.0 - prob0).sqrt() } else { 0.0 }
        };

        for (i, amp) in group.state.iter_mut().enumerate() {
            let bit = (i & mask) != 0;
            if (bit as u32) == result {
                *amp = amp.scale(norm);
            } else {
                *amp = Complex::zero();
            }
        }

        result
    }

    pub fn get_probability(&self, global_index: usize) -> f64 {
        let mut total_prob = 1.0;
        for group in &self.groups {
            let mut local_idx = 0;
            for (vec_idx, &q_id) in group.qubits.iter().enumerate() {
                if (global_index >> q_id) & 1 == 1 {
                    let shift = group.qubits.len() - 1 - vec_idx;
                    local_idx |= 1 << shift;
                }
            }
            if local_idx < group.state.len() {
                total_prob *= group.state[local_idx].magnitude_squared();
            } else {
                return 0.0;
            }
        }
        total_prob
    }


    fn apply_1q(group: &mut StateGroup, shift: usize, mat: Vec<Complex>) -> Result<(), String> {
        let size = group.state.len();
        let mask = 1 << shift;
        let old_state = group.state.clone();
        let mut new_state = vec![Complex::zero(); size];

        let m00 = mat[0]; let m01 = mat[1];
        let m10 = mat[2]; let m11 = mat[3];

        for i in 0..size {
            if (i & mask) == 0 {
                let i0 = i;
                let i1 = i | mask;

                let a0 = old_state[i0];
                let a1 = old_state[i1];

                new_state[i0] = m00 * a0 + m01 * a1;
                new_state[i1] = m10 * a0 + m11 * a1;
            }
        }
        group.state = new_state;
        Ok(())
    }

    fn apply_2q(group: &mut StateGroup, s1: usize, s2: usize, mat: Vec<Complex>) -> Result<(), String> {
        let size = group.state.len();
        let m1 = 1 << s1;
        let m2 = 1 << s2;
        let old_state = group.state.clone();
        let mut new_state = vec![Complex::zero(); size];

        for i in 0..size {
            if (i & m1) == 0 && (i & m2) == 0 {
                let i00 = i;
                let i01 = i | m2;
                let i10 = i | m1;
                let i11 = i | m1 | m2;

                let a00 = old_state[i00];
                let a01 = old_state[i01];
                let a10 = old_state[i10];
                let a11 = old_state[i11];

                let amps = [a00, a01, a10, a11];

                // 4x4 Matrix
                for row in 0..4 {
                    let mut sum = Complex::zero();
                    for col in 0..4 {
                        sum = sum + mat[row * 4 + col] * amps[col];
                    }

                    match row {
                        0 => new_state[i00] = sum,
                        1 => new_state[i01] = sum,
                        2 => new_state[i10] = sum,
                        3 => new_state[i11] = sum,
                        _ => {}
                    }
                }
            }
        }
        group.state = new_state;
        Ok(())
    }

    fn apply_3q_gate(group: &mut StateGroup, c1: usize, c2: usize, t: usize, name: &str, params: &[f64], is_dagger: bool) -> Result<(), String> {
        let m1 = 1 << c1;
        let m2 = 1 << c2;
        let mt = 1 << t;

        let target_gate_name = match name {
            "ccx" | "toffoli" => "x",
            "cch" | "cchadamard" => "h",
            "ccrx" => "rx",
            _ => return Err(format!("3-qubit gate {} not supported", name)),
        };

        let mat = Self::get_matrix(target_gate_name, params, is_dagger)?;
        let m00 = mat[0]; let m01 = mat[1];
        let m10 = mat[2]; let m11 = mat[3];

        let old_state = group.state.clone();
        let mut new_state = old_state.clone();

        for i in 0..group.state.len() {
            // Apply only if controls are 1
            if (i & m1) != 0 && (i & m2) != 0 {
                if (i & mt) == 0 {
                    let i0 = i;
                    let i1 = i | mt;

                    let a0 = old_state[i0];
                    let a1 = old_state[i1];

                    new_state[i0] = m00 * a0 + m01 * a1;
                    new_state[i1] = m10 * a0 + m11 * a1;
                }
            }
        }
        group.state = new_state;
        Ok(())
    }

    fn get_matrix(name: &str, params: &[f64], is_dagger: bool) -> Result<Vec<Complex>, String> {
        let z = Complex::zero();
        let o = Complex::one();
        let i = Complex::i();
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        let h = Complex::new(inv_sqrt2, 0.0);
        let d_mul = if is_dagger { -1.0 } else { 1.0 };

        match name {
            // Single Qubit
            "x" | "not" => Ok(vec![z, o, o, z]),
            "y" => Ok(vec![z, i.scale(-1.0), i, z]),
            "z" => Ok(vec![o, z, z, Complex::new(-1.0, 0.0)]),
            "h" | "hadamard" => Ok(vec![h, h, h, h.scale(-1.0)]),
            "s" => {
                let phase = if is_dagger { i.scale(-1.0) } else { i };
                Ok(vec![o, z, z, phase])
            },
            "t" => {
                let angle = std::f64::consts::FRAC_PI_4 * d_mul;
                Ok(vec![o, z, z, Complex::from_polar(1.0, angle)])
            },

            // Two Qubit
            "cnot" | "cx" | "controlled_x" => Ok(vec![o, z, z, z, z, o, z, z, z, z, z, o, z, z, o, z]),
            "swap" => Ok(vec![o, z, z, z, z, z, o, z, z, o, z, z, z, z, z, o]),
            "cz" | "controlled_z" => Ok(vec![o, z, z, z, z, o, z, z, z, z, o, z, z, z, z, Complex::new(-1.0, 0.0)]),

            // Controlled Phase Gates (4x4)
            "ch" | "chadamard" | "controlled_hadamard" => Ok(vec![
                o, z, z, z, z, o, z, z, z, z, h, h, z, z, h, h.scale(-1.0)
            ]),
            "cs" | "controlled_s" => {
                let phase = if is_dagger { i.scale(-1.0) } else { i };
                Ok(vec![o, z, z, z, z, o, z, z, z, z, o, z, z, z, z, phase])
            },
            "ct" | "controlled_t" => {
                let angle = std::f64::consts::FRAC_PI_4 * d_mul;
                let phase = Complex::from_polar(1.0, angle);
                Ok(vec![o, z, z, z, z, o, z, z, z, z, o, z, z, z, z, phase])
            },

            // Parameterized
            "rx" => {
                let theta = (params.get(0).unwrap_or(&0.0) / 2.0) * d_mul;
                let c = Complex::new(theta.cos(), 0.0);
                let s = Complex::new(0.0, -theta.sin());
                Ok(vec![c, s, s, c])
            },
            "ry" => {
                let theta = (params.get(0).unwrap_or(&0.0) / 2.0) * d_mul;
                let c = Complex::new(theta.cos(), 0.0);
                let s = Complex::new(theta.sin(), 0.0);
                Ok(vec![c, s.scale(-1.0), s, c])
            },
            "rz" => {
                let theta = (params.get(0).unwrap_or(&0.0) / 2.0) * d_mul;
                let a = Complex::from_polar(1.0, -theta);
                let b = Complex::from_polar(1.0, theta);
                Ok(vec![a, z, z, b])
            },
            "cphase" => {
                let phi = params.get(0).unwrap_or(&0.0) * d_mul;
                Ok(vec![o, z, z, z, z, o, z, z, z, z, o, z, z, z, z, Complex::from_polar(1.0, phi)])
            },
            "u" => {
                if params.len() < 3 { return Err("U gate requires 3 parameters".to_string()); }
                let theta = params[0];
                let phi = params[1];
                let lambda = params[2];
                let (t, p, l) = if is_dagger { (-theta, -lambda, -phi) } else { (theta, phi, lambda) };

                let half_t = t / 2.0;
                let cos_t = half_t.cos();
                let sin_t = half_t.sin();
                let a00 = Complex::new(cos_t, 0.0);
                let a01 = Complex::from_polar(-sin_t, l);
                let a10 = Complex::from_polar(sin_t, p);
                let a11 = Complex::from_polar(cos_t, p + l);
                Ok(vec![a00, a01, a10, a11])
            },

            "crx" | "controlled_rx" => {
                let theta = (params.get(0).unwrap_or(&0.0) / 2.0) * d_mul;
                let c = Complex::new(theta.cos(), 0.0);
                let s = Complex::new(0.0, -theta.sin());
                Ok(vec![o, z, z, z, z, o, z, z, z, z, c, s, z, z, s, c])
            },
            "cry" | "controlled_ry" => {
                let theta = (params.get(0).unwrap_or(&0.0) / 2.0) * d_mul;
                let c = Complex::new(theta.cos(), 0.0);
                let s = Complex::new(theta.sin(), 0.0);
                Ok(vec![o, z, z, z, z, o, z, z, z, z, c, s.scale(-1.0), z, z, s, c])
            },
            "crz" | "controlled_rz" => {
                let theta = (params.get(0).unwrap_or(&0.0) / 2.0) * d_mul;
                let a = Complex::from_polar(1.0, -theta);
                let b = Complex::from_polar(1.0, theta);
                Ok(vec![o, z, z, z, z, o, z, z, z, z, a, z, z, z, z, b])
            },


            "ccx" | "toffoli" | "cch" | "cchadamard" | "ccrx" => Ok(vec![]),

            _ => Err(format!("Matrix for {} not implemented", name))
        }
    }
    pub fn dump_state(&self, qubits: &[usize]) {
        // Collect unique groups involved
        let mut seen_groups = Vec::new();
        for &q in qubits {
            if let Some(&g_idx) = self.qubit_map.get(&q) {
                if !seen_groups.contains(&g_idx) {
                    seen_groups.push(g_idx);
                }
            }
        }

        if seen_groups.is_empty() {
            println!("(No active quantum state found for these qubits)");
            return;
        }

        for &g_idx in &seen_groups {
            let group = &self.groups[g_idx];
            println!("--- Real Simulator State (Qubits {:?}) ---", group.qubits);

            for (i, amp) in group.state.iter().enumerate() {
                if amp.magnitude_squared() > 1e-6 {
                    // Print binary representation
                    let width = group.qubits.len();
                    println!("  |{:0width$b}> : {:.6} + {:.6}i", i, amp.real, amp.imag, width=width);
                }
            }
            println!("------------------------------------------");
        }
    }
}

// Wrapper for the Backend Trait
pub struct NativeBackend;
impl NativeBackend { pub fn new() -> Self { NativeBackend } }

impl QuantumBackend for NativeBackend {
    fn execute(&self, circuit: &HardwareCircuit, config: &QuantumConfig) -> Result<QuantumResult, String> {
        let mut counts = HashMap::new();
        let shots = if config.shots > 0 { config.shots } else { 1 };

        // Run the simulation 'shots' times
        for _ in 0..shots {

            let mut sim = NativeSimulator::new(circuit.num_qubits);


            for gate in &circuit.gates {
                sim.apply_gate(&gate.name, &gate.qubits, &gate.params, gate.is_dagger)?;
            }


            let mut bitstring = String::new();
            if circuit.measurements.is_empty() {
            } else {
                for &q_idx in &circuit.measurements {
                    let bit = sim.measure_single(q_idx);
                    bitstring.push_str(&bit.to_string());
                }
            }

            // 4. Record the result
            if !bitstring.is_empty() {
                *counts.entry(bitstring).or_insert(0) += 1;
            }
        }

        Ok(QuantumResult {
            success: true,
            counts,
            shots,
            error_message: None,
        })
    }

    fn is_available(&self) -> bool { true }
    fn available_devices(&self) -> Vec<String> { vec!["native_simulator".to_string()] }
    fn optimize_circuit(&self, c: &HardwareCircuit) -> HardwareCircuit { c.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard_gate() {
        let mut sim = NativeSimulator::new(1);
        sim.apply_gate("h", &[0], &[], false).unwrap();
        let prob0 = sim.get_probability(0);
        assert!((prob0 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cnot_gate() {
        let mut sim = NativeSimulator::new(2);
        sim.apply_gate("x", &[0], &[], false).unwrap();
        sim.apply_gate("cnot", &[0, 1], &[], false).unwrap();
        let prob3 = sim.get_probability(3);
        assert!((prob3 - 1.0).abs() < 1e-10);
    }
}
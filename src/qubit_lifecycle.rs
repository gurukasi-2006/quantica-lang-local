//src/qubit_lifecycle.rs
use std::collections::{HashMap, HashSet};
use crate::parser::ast::Loc;


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QubitState {

    Classical(bool),


    Superposition,


    Measured(bool),


    Entangled(HashSet<QubitId>),


    Invalid,


    Reset,
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QubitId {
    pub register_name: String,
    pub index: usize,
}

impl QubitId {
    pub fn new(register_name: String, index: usize) -> Self {
        QubitId { register_name, index }
    }
}


#[derive(Debug, Clone, PartialEq)]
pub enum QubitOperation {
    Initialize,
    ApplyGate(String),
    ApplyControlledGate(String),
    Measure,
    Reset,
    Entangle(Vec<QubitId>),
}


#[derive(Debug, Clone)]
pub enum LifecycleError {

    UsedAfterMeasurement {
        qubit: QubitId,
        loc: Loc,
        measured_at: Loc,
    },


    DoubleMeasurement {
        qubit: QubitId,
        loc: Loc,
        first_measurement: Loc,
    },


    UninitializedQubit {
        qubit: QubitId,
        loc: Loc,
    },


    InvalidEntangledOperation {
        qubit: QubitId,
        entangled_with: Vec<QubitId>,
        operation: String,
        loc: Loc,
    },


    InvalidState {
        qubit: QubitId,
        state: String,
        operation: String,
        loc: Loc,
    },


    ClassicalQuantumMixing {
        qubit: QubitId,
        loc: Loc,
        explanation: String,
    },
}

impl LifecycleError {
    pub fn to_string(&self) -> String {
        match self {
            LifecycleError::UsedAfterMeasurement { qubit, loc, measured_at } => {
                format!(
                    "Lifecycle Error at {}: Qubit '{}[{}]' was used after being measured (measured at {}). \
                    Once a qubit is measured, it collapses to a classical state and cannot be used in quantum operations.",
                    loc, qubit.register_name, qubit.index, measured_at
                )
            }
            LifecycleError::DoubleMeasurement { qubit, loc, first_measurement } => {
                format!(
                    "Lifecycle Error at {}: Qubit '{}[{}]' was measured twice (first measurement at {}). \
                    A qubit can only be measured once per computation path.",
                    loc, qubit.register_name, qubit.index, first_measurement
                )
            }
            LifecycleError::UninitializedQubit { qubit, loc } => {
                format!(
                    "Lifecycle Error at {}: Qubit '{}[{}]' is used before initialization. \
                    All qubits must be initialized before use.",
                    loc, qubit.register_name, qubit.index
                )
            }
            LifecycleError::InvalidEntangledOperation { qubit, entangled_with, operation, loc } => {
                let entangled_list: Vec<String> = entangled_with
                    .iter()
                    .map(|q| format!("{}[{}]", q.register_name, q.index))
                    .collect();
                format!(
                    "Lifecycle Error at {}: Cannot perform '{}' on qubit '{}[{}]' because it is entangled with {}. \
                    Operations on entangled qubits must consider the entire entangled system.",
                    loc, operation, qubit.register_name, qubit.index, entangled_list.join(", ")
                )
            }
            LifecycleError::InvalidState { qubit, state, operation, loc } => {
                format!(
                    "Lifecycle Error at {}: Cannot perform '{}' on qubit '{}[{}]' in state '{}'. \
                    The qubit is in an invalid state for this operation.",
                    loc, operation, qubit.register_name, qubit.index, state
                )
            }
            LifecycleError::ClassicalQuantumMixing { qubit, loc, explanation } => {
                format!(
                    "Lifecycle Error at {}: Incorrect mixing of classical and quantum operations on qubit '{}[{}]'. {}",
                    loc, qubit.register_name, qubit.index, explanation
                )
            }
        }
    }
}


pub struct QubitLifecycleManager {

    qubit_states: HashMap<QubitId, QubitState>,


    measurement_locations: HashMap<QubitId, Loc>,


    operation_history: HashMap<QubitId, Vec<(QubitOperation, Loc)>>,


    entanglement_groups: Vec<HashSet<QubitId>>,


    strict_mode: bool,
}

impl QubitLifecycleManager {
    pub fn new(strict_mode: bool) -> Self {
        QubitLifecycleManager {
            qubit_states: HashMap::new(),
            measurement_locations: HashMap::new(),
            operation_history: HashMap::new(),
            entanglement_groups: Vec::new(),
            strict_mode,
        }
    }


    pub fn register_qubits(
        &mut self,
        register_name: &str,
        size: usize,
        initial_state: QubitState,
    ) {
        for i in 0..size {
            let qubit_id = QubitId::new(register_name.to_string(), i);
            self.qubit_states.insert(qubit_id.clone(), initial_state.clone());
            self.operation_history.insert(qubit_id, Vec::new());
        }
    }


    pub fn check_operation(
        &self,
        qubit: &QubitId,
        operation: &QubitOperation,
        loc: Loc,
    ) -> Result<(), LifecycleError> {
        let state = self.qubit_states.get(qubit).ok_or_else(|| {
            LifecycleError::UninitializedQubit {
                qubit: qubit.clone(),
                loc,
            }
        })?;

        match (state, operation) {

            (QubitState::Measured(_), QubitOperation::ApplyGate(gate_name)) => {
                if let Some(measured_loc) = self.measurement_locations.get(qubit) {
                    return Err(LifecycleError::UsedAfterMeasurement {
                        qubit: qubit.clone(),
                        loc,
                        measured_at: *measured_loc,
                    });
                }
                Err(LifecycleError::InvalidState {
                    qubit: qubit.clone(),
                    state: "measured".to_string(),
                    operation: gate_name.clone(),
                    loc,
                })
            }


            (QubitState::Measured(_), QubitOperation::Measure) => {
                if let Some(first_loc) = self.measurement_locations.get(qubit) {
                    return Err(LifecycleError::DoubleMeasurement {
                        qubit: qubit.clone(),
                        loc,
                        first_measurement: *first_loc,
                    });
                }
                Ok(())
            }


            (QubitState::Invalid, _) => Err(LifecycleError::InvalidState {
                qubit: qubit.clone(),
                state: "invalid".to_string(),
                operation: format!("{:?}", operation),
                loc,
            }),

            /*
            (QubitState::Entangled(entangled_set), QubitOperation::ApplyGate(gate_name))
                if self.strict_mode =>
            {
                Err(LifecycleError::InvalidEntangledOperation {
                    qubit: qubit.clone(),
                    entangled_with: entangled_set.iter().cloned().collect(),
                    operation: gate_name.clone(),
                    loc,
                })
            }*/


            _ => Ok(()),
        }
    }


    pub fn record_operation(
        &mut self,
        qubit: &QubitId,
        operation: QubitOperation,
        loc: Loc,
    ) -> Result<(), LifecycleError> {

        self.check_operation(qubit, &operation, loc)?;


        match &operation {
            QubitOperation::ApplyGate(gate_name) => {

                if gate_name.to_lowercase() == "x" || gate_name.to_lowercase() == "not" {

                    if let Some(QubitState::Classical(bit)) = self.qubit_states.get(qubit) {
                        self.qubit_states.insert(qubit.clone(), QubitState::Classical(!bit));
                    } else {
                        self.qubit_states.insert(qubit.clone(), QubitState::Superposition);
                    }
                } else {
                    self.qubit_states.insert(qubit.clone(), QubitState::Superposition);
                }
            }

            QubitOperation::Measure => {

                self.qubit_states.insert(qubit.clone(), QubitState::Measured(false));
                self.measurement_locations.insert(qubit.clone(), loc);
            }

            QubitOperation::Reset => {

                self.qubit_states.insert(qubit.clone(), QubitState::Reset);
                self.measurement_locations.remove(qubit);
            }

            QubitOperation::Entangle(qubits) => {

                let mut group = HashSet::new();
                for q in qubits {
                    group.insert(q.clone());
                    self.qubit_states.insert(q.clone(), QubitState::Entangled(group.clone()));
                }
                self.entanglement_groups.push(group);
            }

            _ => {}
        }


        if let Some(history) = self.operation_history.get_mut(qubit) {
            history.push((operation, loc));
        }

        Ok(())
    }


    pub fn record_controlled_gate(
        &mut self,
        control_qubits: &[QubitId],
        target_qubits: &[QubitId],
        gate_name: &str,
        loc: Loc,
    ) -> Result<(), LifecycleError> {

        for qubit in control_qubits.iter().chain(target_qubits.iter()) {
            self.check_operation(
                qubit,
                &QubitOperation::ApplyControlledGate(gate_name.to_string()),
                loc,
            )?;
        }


        let mut all_qubits: Vec<QubitId> = control_qubits.to_vec();
        all_qubits.extend(target_qubits.iter().cloned());

        let entangle_op = QubitOperation::Entangle(all_qubits.clone());
        for qubit in &all_qubits {
            self.record_operation(qubit, entangle_op.clone(), loc)?;
        }

        Ok(())
    }


    pub fn get_state(&self, qubit: &QubitId) -> Option<&QubitState> {
        self.qubit_states.get(qubit)
    }


    pub fn get_history(&self, qubit: &QubitId) -> Option<&Vec<(QubitOperation, Loc)>> {
        self.operation_history.get(qubit)
    }


    pub fn is_measured(&self, qubit: &QubitId) -> bool {
        matches!(self.qubit_states.get(qubit), Some(QubitState::Measured(_)))
    }


    pub fn is_entangled(&self, qubit: &QubitId) -> bool {
        matches!(self.qubit_states.get(qubit), Some(QubitState::Entangled(_)))
    }


    pub fn get_entangled_qubits(&self, qubit: &QubitId) -> Vec<QubitId> {
        if let Some(QubitState::Entangled(group)) = self.qubit_states.get(qubit) {
            group.iter().cloned().collect()
        } else {
            Vec::new()
        }
    }


    pub fn print_summary(&self) {
        println!("=== Qubit Lifecycle Summary ===");

        let mut qubits: Vec<_> = self.qubit_states.keys().collect();
        qubits.sort_by(|a, b| {
            a.register_name.cmp(&b.register_name)
                .then(a.index.cmp(&b.index))
        });

        for qubit in qubits {
            if let Some(state) = self.qubit_states.get(qubit) {
                println!("{}[{}]: {:?}", qubit.register_name, qubit.index, state);

                if let Some(history) = self.operation_history.get(qubit) {
                    if !history.is_empty() {
                        println!("  History: {} operations", history.len());
                    }
                }
            }
        }

        if !self.entanglement_groups.is_empty() {
            println!("\nEntanglement Groups:");
            for (i, group) in self.entanglement_groups.iter().enumerate() {
                let qubits: Vec<String> = group
                    .iter()
                    .map(|q| format!("{}[{}]", q.register_name, q.index))
                    .collect();
                println!("  Group {}: {}", i + 1, qubits.join(", "));
            }
        }

        println!("===============================");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_lifecycle() {
        let mut manager = QubitLifecycleManager::new(true);


        manager.register_qubits("q", 1, QubitState::Classical(false));

        let q0 = QubitId::new("q".to_string(), 0);
        let loc = Loc { line: 1, column: 1 };


        assert!(manager.record_operation(&q0, QubitOperation::ApplyGate("H".to_string()), loc).is_ok());


        assert!(manager.record_operation(&q0, QubitOperation::Measure, loc).is_ok());


        assert!(manager.record_operation(&q0, QubitOperation::ApplyGate("X".to_string()), loc).is_err());
    }

    #[test]
    fn test_double_measurement() {
        let mut manager = QubitLifecycleManager::new(true);
        manager.register_qubits("q", 1, QubitState::Classical(false));

        let q0 = QubitId::new("q".to_string(), 0);
        let loc = Loc { line: 1, column: 1 };


        assert!(manager.record_operation(&q0, QubitOperation::Measure, loc).is_ok());


        assert!(manager.record_operation(&q0, QubitOperation::Measure, loc).is_err());
    }

    #[test]
    fn test_entanglement() {
        let mut manager = QubitLifecycleManager::new(true);
        manager.register_qubits("q", 2, QubitState::Classical(false));

        let q0 = QubitId::new("q".to_string(), 0);
        let q1 = QubitId::new("q".to_string(), 1);
        let loc = Loc { line: 1, column: 1 };


        assert!(manager.record_controlled_gate(&[q0.clone()], &[q1.clone()], "CNOT", loc).is_ok());


        assert!(manager.is_entangled(&q0));
        assert!(manager.is_entangled(&q1));
    }
}
// Quantica Automated Regression Suite v1.1

let PI = 3.14159265
print("\n🚀 STARTING AUTOMATED QUANTUM TEST SUITE\n")


// 1. Basic Pauli Gates (X, Y, Z)

print("TEST 1: Pauli Gates...")
quantum q_pauli[3]

// Test X: |0> -> |1>
apply X(q_pauli[0])
let m_x = measure(q_pauli[0])
assert(m_x == 1, "❌ Pauli X failed: Expected 1")

// Test Y: |0> -> i|1> (Magnitude is 1)
apply Y(q_pauli[1])
let m_y = measure(q_pauli[1])
assert(m_y == 1, "❌ Pauli Y failed: Expected 1")

// Test Z: H -> Z -> H (Should act like X)
apply Hadamard(q_pauli[2])
apply Z(q_pauli[2])
apply Hadamard(q_pauli[2])
let m_z = measure(q_pauli[2])
assert(m_z == 1, "❌ Pauli Z (Phase Flip) failed")

print("   ✅ Pauli Gates Passed")


// 2. Reversibility (Hadamard & Swap)

print("TEST 2: Reversibility (H & Swap)...")

// --- Subtest A: Hadamard Reversibility ---
quantum q_rev_h[1]
apply Hadamard(q_rev_h[0])
apply Hadamard(q_rev_h[0])
let m_h = measure(q_rev_h[0])
assert(m_h == 0, "❌ H-H Reversibility failed")

// --- Subtest B: Swap Gate ---
quantum q_swap[2]
// Swap |10> -> |01>
apply X(q_swap[0]) // Set q0 to 1
apply Swap(q_swap[0], q_swap[1]) // Swap q0 and q1

let m_s0 = measure(q_swap[0])
let m_s1 = measure(q_swap[1])
assert(m_s0 == 0, "❌ Swap q0 failed (Should be 0)")
assert(m_s1 == 1, "❌ Swap q1 failed (Should be 1)")

print("   ✅ Reversibility Passed")


// 3. Phase Gates (S, T, RZ) via Interference

print("TEST 3: Phase Gates (S, T)...")
quantum q_phase[2]

// Test S: S*S = Z. So H*S*S*H = X -> |1>
apply Hadamard(q_phase[0])
apply S(q_phase[0])
apply S(q_phase[0])
apply Hadamard(q_phase[0])
let m_s = measure(q_phase[0])
assert(m_s == 1, "❌ S Gate (Phase) failed")

// Test T: T*T = S. So T^4 = Z. H*T^4*H = |1>
apply Hadamard(q_phase[1])
apply T(q_phase[1])
apply T(q_phase[1])
apply T(q_phase[1])
apply T(q_phase[1])
apply Hadamard(q_phase[1])
let m_t = measure(q_phase[1])
assert(m_t == 1, "❌ T Gate (Phase) failed")

print("   ✅ Phase Gates Passed")


// 4. Entanglement (Bell States)

print("TEST 4: Entanglement (Bell Φ+)...")
quantum bell[2]

// Create Bell State |Φ+> = (|00> + |11>)/√2
apply Hadamard(bell[0])
apply CNOT(bell[0], bell[1])

// Verify Entanglement (Deterministic Uncompute)
// Inverse of CNOT is CNOT. Inverse of H is H.
// Applying them in reverse should yield |00>
apply CNOT(bell[0], bell[1])
apply Hadamard(bell[0])

let b0 = measure(bell[0])
let b1 = measure(bell[1])

assert(b0 == 0, "❌ Bell Uncompute q0 failed")
assert(b1 == 0, "❌ Bell Uncompute q1 failed")

print("   ✅ Bell State Passed")


// 5. Multi-Qubit Logic (Toffoli & GHZ)

print("TEST 5: Multi-Qubit (Toffoli & GHZ)...")

// Toffoli (CCX): |110> -> |111>
quantum tof[3]
apply X(tof[0])
apply X(tof[1])
apply CCX(tof[0], tof[1], tof[2])

let t0 = measure(tof[0])
let t1 = measure(tof[1])
let t2 = measure(tof[2])

assert(t0 == 1, "❌ Toffoli Control 1 corrupted")
assert(t1 == 1, "❌ Toffoli Control 2 corrupted")
assert(t2 == 1, "❌ Toffoli Target failed")

// GHZ: |000> -> (|000> + |111>) -> Uncompute -> |000>
quantum ghz[3]
// Create
apply Hadamard(ghz[0])
apply CNOT(ghz[0], ghz[1])
apply CNOT(ghz[0], ghz[2])
// Uncompute (Inverse Order)
apply CNOT(ghz[0], ghz[2])
apply CNOT(ghz[0], ghz[1])
apply Hadamard(ghz[0])

let g0 = measure(ghz[0])
let g1 = measure(ghz[1])
let g2 = measure(ghz[2])

assert(g0 == 0, "❌ GHZ q0 failed")
assert(g1 == 0, "❌ GHZ q1 failed")
assert(g2 == 0, "❌ GHZ q2 failed")

print("   ✅ Multi-Qubit Passed")


// 6. Advanced Modifiers (Controlled & Dagger)

print("TEST 6: Advanced Modifiers...")
quantum q_adv[2]

// Test Controlled-Hadamard (CH)
// |10> -> |1> ⊗ H|0> -> |1+>
// Uncompute: Apply CH again -> |10>
apply X(q_adv[0]) // Control = 1
apply controlled(Hadamard)(q_adv[0], q_adv[1]) // Target enters superposition
apply controlled(Hadamard)(q_adv[0], q_adv[1]) // Target exits superposition

let c0 = measure(q_adv[0])
let c1 = measure(q_adv[1])
assert(c0 == 1, "❌ Controlled-H q0 failed")
assert(c1 == 0, "❌ Controlled-H q1 failed")

// Test Dagger(S)
// S * Dagger(S) = I
quantum q_dag[1]
apply S(q_dag[0])
apply dagger(S)(q_dag[0])
let m_dag = measure(q_dag[0])
assert(m_dag == 0, "❌ Dagger(S) failed")

print("   ✅ Modifiers Passed")


print("\n🎉 ALL TESTS PASSED SUCCESSFULLY! 🎉")
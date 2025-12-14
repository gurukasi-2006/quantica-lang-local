// examples/classes_example.qc
// Demonstrating classes and objects in Quantica

print("=== QUANTICA CLASSES & OBJECTS ===\n")

// --- Example 1: Basic Class ---
print("--- Example 1: Basic Point Class ---")

class Point:
    let x: Float = 0.0
    let y: Float = 0.0

    func init(x_val: Float, y_val: Float):
        x = x_val
        y = y_val

    func distance_from_origin() -> Float:
        return sqrt(x * x + y * y)

    func move(dx: Float, dy: Float):
        x = x + dx
        y = y + dy

    func display():
        print("Point(" + to_string(x) + ", " + to_string(y) + ")")

let p1 = new Point(3.0, 4.0)
p1.display()
print("Distance from origin:", p1.distance_from_origin())

p1.move(1.0, 1.0)
print("After moving:")
p1.display()

// --- Example 2: Quantum Circuit Class ---
print("\n--- Example 2: Quantum Circuit Class ---")

class QuantumCircuit:
    let num_qubits: Int
    let qreg: QuantumRegister

    func init(n: Int):
        num_qubits = n
        quantum temp[n]
        qreg = temp

    func apply_hadamard(index: Int):
        apply Hadamard(qreg[index])

    func apply_cnot(control: Int, target: Int):
        apply CNOT(qreg[control], qreg[target])

    func measure_all():
        for i in 0..num_qubits:
            let result = measure(qreg[i])
            print("Qubit", i, ":", result)

    func create_bell_state():
        apply_hadamard(0)
        apply_cnot(0, 1)

let circuit = new QuantumCircuit(2)
circuit.create_bell_state()
print("Bell state created!")
circuit.measure_all()

// --- Example 3: Inheritance (if supported) ---
print("\n--- Example 3: Vector Class with Inheritance ---")

class Vector2D:
    public let x: Float
    public let y: Float

    func init(x_val: Float, y_val: Float):
        x = x_val
        y = y_val

    func magnitude() -> Float:
        return sqrt(x * x + y * y)

    func dot(other: Vector2D) -> Float:
        return x * other.x + y * other.y

class Vector3D(Vector2D):
    public let z: Float

    func init(x_val: Float, y_val: Float, z_val: Float):
        super.init(x_val, y_val)
        z = z_val

    func magnitude() -> Float:
        return sqrt(x * x + y * y + z * z)

    func cross(other: Vector3D) -> Vector3D:
        let cx = y * other.z - z * other.y
        let cy = z * other.x - x * other.z
        let cz = x * other.y - y * other.x
        return new Vector3D(cx, cy, cz)

let v1 = new Vector3D(1.0, 0.0, 0.0)
let v2 = new Vector3D(0.0, 1.0, 0.0)
print("v1 magnitude:", v1.magnitude())
print("v1 dot v2:", v1.dot(v2))

// --- Example 4: Counter Class with Static Method ---
print("\n--- Example 4: Counter Class ---")

class Counter:
    private mut count: Int = 0
    static let max_count: Int = 100

    func increment():
        if count < max_count:
            count = count + 1

    func decrement():
        if count > 0:
            count = count - 1

    func get_value() -> Int:
        return count

    func reset():
        count = 0

let counter = new Counter()
counter.increment()
counter.increment()
counter.increment()
print("Counter value:", counter.get_value())

counter.decrement()
print("After decrement:", counter.get_value())

counter.reset()
print("After reset:", counter.get_value())

// --- Example 5: Quantum State Manager ---
print("\n--- Example 5: Quantum State Manager ---")

class QuantumStateManager:
    private let states: Array
    private mut current_index: Int

    func init():
        states = []
        current_index = 0

    func add_state(name: String, qreg: QuantumRegister):
        // Store state information
        states[current_index] = {"name": name, "register": qreg}
        current_index = current_index + 1

    func get_state_count() -> Int:
        return current_index

    func debug_all():
        for i in 0..current_index:
            let state = states[i]
            print("State:", state["name"])
            debug_state(state["register"])

let manager = new QuantumStateManager()

quantum q1[2]
apply Hadamard(q1[0])
manager.add_state("superposition", q1)

quantum q2[2]
apply X(q2[0])
apply CNOT(q2[0], q2[1])
manager.add_state("bell_state", q2)

print("\nStored states:", manager.get_state_count())
manager.debug_all()

print("\n=== CLASSES DEMONSTRATION COMPLETE ===")
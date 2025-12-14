// tests/Classes_test.qc
print("--- Running Classes & Objects Tests ---")

// --- Test 1: Basic Class Declaration and Instantiation ---
Class Testclass1:
    let value: Int = 42
    func get_value() -> Int:
        return value

let obj1 = new Testclass1()
assert(obj1.value == 42, "Class field access")
assert(obj1.get_value() == 42, "Method call on instance")
print("✓ Test 1: Basic Class passed")

// --- Test 2: Constructor ---
Class TestClass2:
    let x: Int
    let y: Int

    func init(a: Int, b: Int):
        x = a
        y = b

    func sum() -> Int:
        return x + y

let obj2 = new TestClass2(10, 20)
assert(obj2.x == 10, "Constructor parameter 1")
assert(obj2.y == 20, "Constructor parameter 2")
assert(obj2.sum() == 30, "Method using constructor values")
print("✓ Test 2: Constructor passed")

// --- Test 3: Multiple Instances ---
Class TestClass3:
    mut counter: Int = 0

    func increment():
        counter = counter + 1

    func get_counter() -> Int:
        return counter

let obj3a = new TestClass3()
let obj3b = new TestClass3()

obj3a.increment()
obj3a.increment()
obj3b.increment()

assert(obj3a.get_counter() == 2, "First instance counter")
assert(obj3b.get_counter() == 1, "Second instance counter")
print("✓ Test 3: Multiple instances passed")

// --- Test 4: Method Chaining (if supported) ---
Class TestClass4:
    mut value: Int = 0

    func set_value(v: Int):
        value = v

    func add(v: Int):
        value = value + v

    func multiply(v: Int):
        value = value * v

    func get_value() -> Int:
        return value

let obj4 = new TestClass4()
obj4.set_value(5)
obj4.add(3)
obj4.multiply(2)
assert(obj4.get_value() == 16, "Method chaining")
print("✓ Test 4: Method chaining passed")

// --- Test 5: Private/Public Fields ---
Class TestClass5:
    public let public_val: Int = 100
    private let private_val: Int = 200

    func get_private() -> Int:
        return private_val

let obj5 = new TestClass5()
assert(obj5.public_val == 100, "Public field access")
assert(obj5.get_private() == 200, "Private field via method")
print("✓ Test 5: Access modifiers passed")

// --- Test 6: Quantum Circuit in Class ---
Class QuantumTest:
    let qreg: QuantumRegister

    func init():
        quantum q[2]
        qreg = q

    func create_entanglement():
        apply Hadamard(qreg[0])
        apply CNOT(qreg[0], qreg[1])

    func measure_first() -> Int:
        return measure(qreg[0])

let qtest = new QuantumTest()
qtest.create_entanglement()
let measurement = qtest.measure_first()
assert(measurement == 0 Or measurement == 1, "Quantum measurement in Class")
print("✓ Test 6: Quantum operations in Class passed")

// --- Test 7: Nested Object References ---
Class Inner:
    let val: Int = 999

    func get_val() -> Int:
        return val

Class Outer:
    let inner: Inner

    func init():
        inner = new Inner()

    func get_inner_val() -> Int:
        return inner.get_val()

let outer = new Outer()
assert(outer.get_inner_val() == 999, "Nested object reference")
print("✓ Test 7: Nested objects passed")

print("\n--- All Classes & Objects Tests Passed! ---")
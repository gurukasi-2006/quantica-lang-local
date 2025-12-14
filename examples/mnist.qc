import "stdlib/ai.qc" as ai
import "stdlib/math.qc" as math

print("=== QUANTICA MNIST TRAINING ===")

/* --- Helper: One-Hot Encode Labels --- */
func one_hot(label: Int) -> Float[]:
    mut target = []
    mut i = 0
    while i < 10:
        if i == label:
            target = target + [1.0]
        else:
            target = target + [0.0]
        i = i + 1
    return target

/* --- Helper: Load CSV Data --- */
func load_mnist(filename: String, max_samples: Int) -> ai.Dataset:
    print("DEBUG: Reading " + filename + "...")
    let content = file_read(filename)
    let lines = split(content, "\n")

    let dataset = new ai.Dataset()

    mut count = max_samples
    if len(lines) < max_samples:
        count = len(lines)
    if max_samples <= 0:
        count = len(lines)

    print("DEBUG: Parsing " + to_string(count) + " samples...")

    mut i = 0
    mut samples_loaded = 0

    while i < count:
        let line = lines[i]

        if len(line) > 10:
            let values = split(line, ",")

            // Header Check
            if values[0] == "label":
                i = i + 1
                count = count + 1
                if count > len(lines):
                    count = len(lines)
            else:
                let label = to_int(values[0])
                let target = one_hot(label)

                mut input = []
                mut p = 1
                while p < 785:
                    let pixel = to_float(values[p]) / 255.0
                    input = input + [pixel]
                    p = p + 1

                dataset.add_sample(input, target)
                samples_loaded = samples_loaded + 1

        if samples_loaded > 0:
            if (samples_loaded % 500) == 0:
                print("DEBUG: Loaded " + to_string(samples_loaded) + " samples...")

        i = i + 1

    print("SUCCESS: Loaded " + to_string(dataset.size) + " samples.")
    return dataset

/* --- MAIN TRAINING --- */

// Loading Data
print("--- 1. Loading Training Set ---")
let train_data = load_mnist("mnist_train.csv", 1000)

print("\n--- 2. Loading Test Set ---")
let test_data = load_mnist("mnist_test.csv", 100)

// Creating Network
print("\n--- 3. Initializing Network ---")
let nn = new ai.NeuralNetwork(0.1)
nn.add_layer(784, 128, "relu")
nn.add_layer(128, 10, "sigmoid")

// Train with Timer
print("\n--- 4. Starting Training ---")

// Start timer
let start_time = time()

ai.train_network(nn, train_data, 6, 1, True)

// Stop timer
let end_time = time()
let duration = end_time - start_time
print(">> Total Training Time: " + to_string(duration) + " seconds")

//Evaluate
print("\n--- 5. Evaluating ---")
let acc = ai.evaluate_network(nn, test_data, 0.5)
print(">> TEST ACCURACY: " + to_string(acc * 100.0) + "%")

// Save
print("\n--- 6. Saving Model ---")
nn.save("mnist_brain.qmodel")
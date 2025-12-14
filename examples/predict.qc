import "stdlib/ai.qc" as ai
import "stdlib/math.qc" as math

print("=== QUANTICA DIGIT PREDICTOR ===")

/* --- Helper: Draw Digit as ASCII Art --- */
func draw_digit(pixels: Float[]):
    print("--- IMAGE ---")
    mut row = 0
    while row < 28:
        mut line = ""
        mut col = 0
        while col < 28:
            let idx = row * 28 + col
            let val = pixels[idx]

            if val > 0.5:
                line = line + " # "
            elif val > 0.2:
                line = line + " . "
            else:
                line = line + "   "
            col = col + 1
        print(line)
        row = row + 1
    print("-------------")

/* --- Helper: Find index of max value --- */
func argmax(values: Float[]) -> Int:
    mut max_idx = 0
    mut max_val = values[0]
    mut i = 1
    while i < len(values):
        if values[i] > max_val:
            max_val = values[i]
            max_idx = i
        i = i + 1
    return max_idx

/* --- Helper: Load CSV Data (Single Sample) --- */
func load_sample(filename: String, line_index: Int) -> Dict:
    let content = file_read(filename)
    let lines = split(content, "\n")
    mut real_index = line_index
    let first = split(lines[0], ",")
    if first[0] == "label":
        real_index = real_index + 1

    let line = lines[real_index]
    let values = split(line, ",")

    let label = to_int(values[0])

    mut input = []
    mut p = 1
    while p < 785:
        let pixel = to_float(values[p]) / 255.0
        input = input + [pixel]
        p = p + 1

    return {"input": input, "label": label}

/* --- MAIN PREDICTION LOGIC --- */

print("Initializing Network Structure...")
let nn = new ai.NeuralNetwork(0.1)
nn.add_layer(784, 128, "relu")
nn.add_layer(128, 10, "sigmoid")

print("Loading 'mnist_brain.qmodel'...")
nn.load("mnist_brain.qmodel")

print("\n--- TESTING MULTIPLE DIGITS ---")

mut i = 0
while i < 5:
    print("\n[ Test Sample #" + to_string(i) + " ]")

    // Load
    let sample = load_sample("mnist_test.csv", i)

    // Predict
    let prediction_probs = nn.predict(sample["input"])
    let predicted_digit = argmax(prediction_probs)

    // Visualize
    draw_digit(sample["input"])

    // Result
    print("TRUE LABEL:    " + to_string(sample["label"]))
    print("PREDICTION:    " + to_string(predicted_digit))

    if sample["label"] == predicted_digit:
        print("RESULT:        ✅ PASS")
    else:
        print("RESULT:        ❌ FAIL")

    i = i + 1
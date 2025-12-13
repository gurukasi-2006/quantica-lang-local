import "stdlib/ai.qc" as ai

print("--- 1. Training a Model ---")
let nn = new ai.NeuralNetwork(0.5)
nn.add_layer(2, 2, "sigmoid")
nn.add_layer(2, 1, "sigmoid")

// Train on XOR (briefly)
let dataset = new ai.Dataset()
dataset.add_sample([0.0, 0.0], [0.0])
dataset.add_sample([0.0, 1.0], [1.0])
dataset.add_sample([1.0, 0.0], [1.0])
dataset.add_sample([1.0, 1.0], [0.0])

ai.train_network(nn, dataset, 500, 1, False)

print("Prediction BEFORE save (0,1): " + to_string(nn.predict([0.0, 1.0])[0]))

// Save
nn.save("xor_model.qmodel")

print("\n--- 2. Loading into NEW Model ---")
let nn2 = new ai.NeuralNetwork(0.1) // Dummy LR
nn2.load("xor_model.qmodel")

print("Prediction AFTER load (0,1):  " + to_string(nn2.predict([0.0, 1.0])[0]))

// Verify
let diff = nn.predict([0.0, 1.0])[0] - nn2.predict([0.0, 1.0])[0]
if diff < 0.00001:
    print("✅ SUCCESS: Model restored perfectly.")
else:
    print("❌ FAILURE: Models do not match.")
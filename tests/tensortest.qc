print("=== QUANTICA GENERAL TENSOR TEST ===\n")
print("--- 1. Finance: Calculating Total Cost ---")
let quantities = [2.0, 5.0, 3.0]
let prices = [0.50, 0.20, 0.80]
let total_cost = quantities *** prices

print("Quantities: " + to_string(quantities))
print("Prices:     " + to_string(prices))
print("Total Cost: $" + to_string(total_cost))
assert(total_cost == 4.4, "Shopping bill calculation")

print("\n--- 2. Physics: Work Calculation ---")
let force = [10.0, 20.0, 5.0]
let displacement = [2.0, 0.0, 4.0]
let work_done = force *** displacement

print("Force Vector: " + to_string(force))
print("Move Vector:  " + to_string(displacement))
print("Work Done:    " + to_string(work_done) + " Joules")

assert(work_done == 40.0, "Work calculation")





print("\n--- 3. Grading: Weighted Average ---")
let scores = [90.0, 80.0, 95.0]
let weights = [0.2, 0.3, 0.5]

let final_grade = scores *** weights

print("Scores:  " + to_string(scores))
print("Weights: " + to_string(weights))
print("Weighted Grade: " + to_string(final_grade))

assert(final_grade == 89.5, "Weighted grade calculation")
print("\n ALL GENERAL TENSOR TESTS PASSED!")
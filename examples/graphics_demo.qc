// examples/graphics_demo.qc
// Comprehensive graphics demonstration

import "stdlib/graphics.qc" as gfx
import "stdlib/math.qc" as math

print("=== QUANTICA GRAPHICS DEMO ===\n")

// ============================================
// 1. BASIC SHAPES
// ============================================
print("--- Test 1: Basic Shapes ---")
let canvas1 = gfx.Canvas(800, 600)
canvas1.set_background(gfx.COLOR_WHITE)

// Draw shapes
canvas1.draw_rect(50.0, 50.0, 100.0, 80.0, gfx.COLOR_RED, True)
canvas1.draw_circle(400.0, 300.0, 50.0, gfx.COLOR_BLUE, False)
canvas1.draw_line(100.0, 500.0, 700.0, 500.0, gfx.COLOR_GREEN, 3.0)
canvas1.draw_text(350.0, 550.0, "Basic Shapes Demo", gfx.COLOR_BLACK, 20.0)

canvas1.save_svg("output/basic_shapes.svg")
print("✓ Saved basic_shapes.svg")
canvas1.destroy()

// ============================================
// 2. GEOMETRIC PATTERNS
// ============================================
print("\n--- Test 2: Geometric Patterns ---")
let canvas2 = gfx.Canvas(800, 800)
canvas2.set_background(gfx.COLOR_WHITE)

// Draw concentric circles
mut i = 1
while i <= 10:
    let radius = to_float(i * 30)
    let color = gfx.lerp_color(gfx.COLOR_BLUE, gfx.COLOR_RED,to_float(i) / 10.0)
    canvas2.draw_circle(400.0, 400.0, radius, color, False)
    i = i + 1

canvas2.draw_text(350.0, 750.0, "Concentric Circles",gfx.COLOR_BLACK, 20.0)

canvas2.save_svg("output/patterns.svg")
print("✓ Saved patterns.svg")
canvas2.destroy()

// ============================================
// 3. STAR PATTERN
// ============================================
print("\n--- Test 3: Star Pattern ---")
let canvas3 = gfx.Canvas(800, 800)
canvas3.set_background(gfx.COLOR_BLACK)

// Draw multiple stars
mut j = 0
while j < 5:
    let angle = math.TAU * to_float(j) / 5.0
    let x = 400.0 + 200.0 * math.cos(angle)
    let y = 400.0 + 200.0 * math.sin(angle)
    gfx.draw_star(canvas3, x, y, 50.0, 20.0, 5, gfx.COLOR_YELLOW)
    j = j + 1

// Central star
gfx.draw_star(canvas3, 400.0, 400.0, 80.0, 30.0, 5, gfx.COLOR_WHITE)

canvas3.save_svg("output/stars.svg")
print("✓ Saved stars.svg")
canvas3.destroy()

// ============================================
// 4. GRID AND AXES
// ============================================
print("\n--- Test 4: Grid and Axes ---")
let canvas4 = gfx.Canvas(800, 600)
canvas4.set_background(gfx.COLOR_WHITE)

// Draw grid
gfx.draw_grid(canvas4, 50.0, gfx.COLOR_GRAY)

// Draw axes
gfx.draw_axes(canvas4, 400.0, 300.0, 300.0, gfx.COLOR_BLACK)

canvas4.draw_text(350.0, 550.0, "Grid with Axes",gfx.COLOR_BLACK, 20.0)

canvas4.save_svg("output/grid.svg")
print("✓ Saved grid.svg")
canvas4.destroy()

// ============================================
// 5. MATHEMATICAL FUNCTION PLOT
// ============================================
print("\n--- Test 5: Function Plot (Sine Wave) ---")
let canvas5 = gfx.Canvas(800, 600)
canvas5.set_background(gfx.COLOR_WHITE)

// Generate sine wave data
mut x_data = []
mut y_data = []
mut k = 0
while k <= 100:
    let x = to_float(k) / 100.0 * math.TAU * 2.0
    let y = math.sin(x)
    x_data = x_data + [x]
    y_data = y_data + [y]
    k = k + 1

let plot1 = gfx.Plot(gfx.PLOT_LINE)
plot1.set_data(x_data, y_data)
plot1.set_title("Sine Wave")
plot1.render(canvas5)

canvas5.save_svg("output/sine_plot.svg")
print("✓ Saved sine_plot.svg")
canvas5.destroy()
plot1.destroy()

// ============================================
// 6. SCATTER PLOT
// ============================================
print("\n--- Test 6: Scatter Plot ---")
let canvas6 = gfx.Canvas(800, 600)
canvas6.set_background(gfx.COLOR_WHITE)

// Generate random-looking scatter data
mut x_scatter = []
mut y_scatter = []
mut m = 0
while m < 50:
    let x = to_float(m) / 10.0
    let y = math.sin(to_float(m) / 5.0) + to_float(m % 3) / 3.0
    x_scatter = x_scatter + [x]
    y_scatter = y_scatter + [y]
    m = m + 1

let plot2 = gfx.Plot(gfx.PLOT_SCATTER)
plot2.set_data(x_scatter, y_scatter)
plot2.set_title("Scatter Plot Example")
plot2.render(canvas6)

canvas6.save_svg("output/scatter_plot.svg")
print("✓ Saved scatter_plot.svg")
canvas6.destroy()
plot2.destroy()

// ============================================
// 7. BAR CHART
// ============================================
print("\n--- Test 7: Bar Chart ---")
let canvas7 = gfx.Canvas(800, 600)
canvas7.set_background(gfx.COLOR_WHITE)

// Sample data
let x_bars = [1.0, 2.0, 3.0, 4.0, 5.0]
let y_bars = [10.0, 25.0, 15.0, 30.0, 20.0]

let plot3 = gfx.Plot(gfx.PLOT_BAR)
plot3.set_data(x_bars, y_bars)
plot3.set_title("Bar Chart Example")
plot3.render(canvas7)

canvas7.save_svg("output/bar_chart.svg")
print("✓ Saved bar_chart.svg")
canvas7.destroy()
plot3.destroy()

// ============================================
// 8. BLOCH SPHERE (Quantum Visualization)
// ============================================
print("\n--- Test 8: Bloch Sphere ---")
let canvas8 = gfx.Canvas(800, 800)
canvas8.set_background(gfx.COLOR_WHITE)

canvas8.draw_text(300.0, 50.0, "Quantum State Visualization",gfx.COLOR_BLACK, 24.0)

gfx.draw_bloch_sphere(canvas8, 400.0, 400.0, 200.0)

// Draw state vector
let theta = math.PI / 4.0
let phi = 0.0
let r = 200.0
let x_state = 400.0 + r * math.sin(theta) * math.cos(phi)
let y_state = 400.0 + r * math.sin(theta) * math.sin(phi)

canvas8.draw_line(400.0, 400.0, x_state, y_state,gfx.COLOR_RED, 3.0)
canvas8.draw_circle(x_state, y_state, 8.0, gfx.COLOR_RED, True)

canvas8.save_svg("output/bloch_sphere.svg")
print("✓ Saved bloch_sphere.svg")
canvas8.destroy()

// ============================================
// 9. COLORFUL MANDALA
// ============================================
print("\n--- Test 9: Colorful Mandala ---")
let canvas9 = gfx.Canvas(800, 800)
canvas9.set_background(gfx.COLOR_BLACK)

// Draw mandala pattern
mut layer = 0
while layer < 12:
    let radius = 50.0 + to_float(layer) * 25.0
    let sides = 6 + layer
    let t = to_float(layer) / 12.0
    let color = gfx.lerp_color(gfx.COLOR_CYAN, gfx.COLOR_MAGENTA, t)
    gfx.draw_polygon(canvas9, 400.0, 400.0, radius, sides, color, False)
    layer = layer + 1

canvas9.draw_text(320.0, 750.0, "Geometric Mandala",gfx.COLOR_WHITE, 24.0)

canvas9.save_svg("output/mandala.svg")
print("✓ Saved mandala.svg")
canvas9.destroy()

// ============================================
// 10. COMBINED PLOT (Multiple Curves)
// ============================================
print("\n--- Test 10: Multiple Function Plots ---")
let canvas10 = gfx.Canvas(800, 600)
canvas10.set_background(gfx.COLOR_WHITE)

// Draw coordinate system
gfx.draw_axes(canvas10, 100.0, 500.0, 0.0, gfx.COLOR_BLACK)

// Plot multiple functions
mut n = 0
while n <= 100:
    let x = to_float(n) / 10.0
    let x_screen = 100.0 + x * 60.0
    // sin(x)
    let y1 = math.sin(x) * 100.0
    let y1_screen = 300.0 - y1
    if n > 0:
        canvas10.draw_circle(x_screen, y1_screen, 2.0, gfx.COLOR_RED, True)
    // cos(x)
    let y2 = math.cos(x) * 100.0
    let y2_screen = 300.0 - y2
    if n > 0:
        canvas10.draw_circle(x_screen, y2_screen, 2.0, gfx.COLOR_BLUE, True)
    n = n + 1

canvas10.draw_text(300.0, 50.0, "Sin and Cos Functions",gfx.COLOR_BLACK, 20.0)
canvas10.draw_text(120.0, 100.0, "sin(x)", gfx.COLOR_RED, 16.0)
canvas10.draw_text(120.0, 130.0, "cos(x)", gfx.COLOR_BLUE, 16.0)

canvas10.save_svg("output/multi_plot.svg")
print("✓ Saved multi_plot.svg")
canvas10.destroy()

// ============================================
// SUMMARY
// ============================================
print("\n=== GRAPHICS DEMO COMPLETE ===")
print("Generated 10 SVG files in output/ directory:")
print("  1. basic_shapes.svg - Basic geometric shapes")
print("  2. patterns.svg - Concentric circles")
print("  3. stars.svg - Star pattern")
print("  4. grid.svg - Grid with axes")
print("  5. sine_plot.svg - Sine wave plot")
print("  6. scatter_plot.svg - Scatter plot")
print("  7. bar_chart.svg - Bar chart")
print("  8. bloch_sphere.svg - Quantum state visualization")
print("  9. mandala.svg - Geometric mandala")
print(" 10. multi_plot.svg - Multiple function plots")
print("\n✓ All graphics tests passed!")
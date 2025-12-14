// test_draw.qc
import "stdlib/graphics.qc" as graphics

print("--- Starting Graphics Test ---")

// 1. Create a canvas (800x600)
// This calls Canvas.init -> _graphics_create_canvas
let canvas = graphics.Canvas(800, 600)
print("Canvas created with ID: " + to_string(canvas.id))

// 2. Set background to white
// This calls Canvas.set_background -> _graphics_set_background
let white = graphics.COLOR_WHITE
canvas.set_background(white)
print("Background set to white")

// 3. Draw some shapes
// Red Rectangle
let red = graphics.COLOR_RED
canvas.draw_rect(50.0, 50.0, 200.0, 100.0, red, True)
print("Drawn red rectangle")

// Blue Circle (Outline)
let blue = graphics.COLOR_BLUE
canvas.draw_circle(400.0, 300.0, 50.0, blue, False)
print("Drawn blue circle outline")

// Green Line
let green = graphics.COLOR_GREEN
canvas.draw_line(10.0, 10.0, 790.0, 590.0, green, 5.0)
print("Drawn green diagonal line")

// 4. Save the result
// This calls Canvas.save_svg -> _graphics_save_svg
let filename = "output_test.svg"
if canvas.save_svg(filename):
    print("Success! Image saved to: " + filename)
else:
    print("Error: Failed to save image.")

// 5. Cleanup
canvas.destroy()
print("Canvas destroyed")
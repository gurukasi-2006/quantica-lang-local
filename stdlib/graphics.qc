// stdlib/graphics.qc
/*
   Quantica Standard Graphics Library
   Provides 2D/3D graphics, plotting, and visualization
*/

/* ============================================
   COLOR CONSTANTS
   ============================================ */
let COLOR_BLACK = {r: 0, g: 0, b: 0, a: 255}
let COLOR_WHITE = {r: 255, g: 255, b: 255, a: 255}
let COLOR_RED = {r: 255, g: 0, b: 0, a: 255}
let COLOR_GREEN = {r: 0, g: 255, b: 0, a: 255}
let COLOR_BLUE = {r: 0, g: 0, b: 255, a: 255}
let COLOR_YELLOW = {r: 255, g: 255, b: 0, a: 255}
let COLOR_CYAN = {r: 0, g: 255, b: 255, a: 255}
let COLOR_MAGENTA = {r: 255, g: 0, b: 255, a: 255}
let COLOR_GRAY = {r: 128, g: 128, b: 128, a: 255}
let COLOR_ORANGE = {r: 255, g: 165, b: 0, a: 255}
let COLOR_PURPLE = {r: 128, g: 0, b: 128, a: 255}

/* ============================================
   PLOT TYPES
   ============================================ */
let PLOT_LINE = 0
let PLOT_SCATTER = 1
let PLOT_BAR = 2
let PLOT_HISTOGRAM = 3
let PLOT_HEATMAP = 4

/* ============================================
   CANVAS CLASS (2D Drawing)
   ============================================ */

/* Canvas represents a 2D drawing surface */
Class Canvas:
    let width: Int
    let height: Int
    let id: Int

    func init(w: Int, h: Int):
        self.width = w
        self.height = h
        // Call native function to create canvas
        self.id = _graphics_create_canvas(w, h)

    func set_background(color: Dict):
        _graphics_set_background(
            self.id,
            color["r"], color["g"], color["b"], color["a"]
        )

    func clear():
        _graphics_clear(self.id)

    func draw_line(x1: Float, y1: Float, x2: Float, y2: Float,color: Dict, width: Float):
        _graphics_draw_line(
            self.id, x1, y1, x2, y2,
            color["r"], color["g"], color["b"], color["a"],
            width
        )

    func draw_rect(x: Float, y: Float, w: Float, h: Float,color: Dict, filled: Bool):
        let fill_int = if filled: 1 else: 0
        _graphics_draw_rect(
            self.id, x, y, w, h,
            color["r"], color["g"], color["b"], color["a"],
            fill_int
        )

    func draw_circle(x: Float, y: Float, radius: Float,color: Dict, filled: Bool):
        let fill_int = if filled: 1 else: 0
        _graphics_draw_circle(
            self.id, x, y, radius,
            color["r"], color["g"], color["b"], color["a"],
            fill_int
        )

    func draw_text(x: Float, y: Float, text: String,color: Dict, size: Float):
        _graphics_draw_text(
            self.id, x, y, text,
            color["r"], color["g"], color["b"], color["a"],
            size
        )

    func save_svg(filename: String) -> Bool:
        return _graphics_save_svg(self.id, filename) == 0

    func save_png(filename: String) -> Bool:
        return _graphics_save_png(self.id, filename) == 0

    func destroy():
        _graphics_destroy_canvas(self.id)

/* ============================================
   PLOT CLASS (Data Visualization)
   ============================================ */

Class Plot:
    let id: Int
    let plot_type: Int

    func init(ptype: Int):
        self.plot_type = ptype
        self.id = _graphics_create_plot(ptype)

    func set_data(x_data: Float[], y_data: Float[]):
        _graphics_plot_set_data(self.id, x_data, y_data, len(x_data))

    func set_title(title: String):
        _graphics_plot_set_title(self.id, title)

    func render(canvas: Canvas) -> Bool:
        return _graphics_plot_render(self.id, canvas.id) == 0

    func destroy():
        _graphics_destroy_plot(self.id)

/* ============================================
   HELPER FUNCTIONS
   ============================================ */

/* Create a color from RGB values */
func rgb(r: Int, g: Int, b: Int) -> Dict:
    return {r: r, g: g, b: b, a: 255}

/* Create a color from RGBA values */
func rgba(r: Int, g: Int, b: Int, a: Int) -> Dict:
    return {r: r, g: g, b: b, a: a}

/* Create a color from hex string */
func hex_to_color(hex: String) -> Dict:
    // Simple implementation for #RRGGBB format
    // TODO: Implement proper hex parsing
    return COLOR_BLACK

/* Linear interpolation between two colors */
func lerp_color(c1: Dict, c2: Dict, t: Float) -> Dict:
    let r = to_int(c1["r"] * (1.0 - t) + c2["r"] * t)
    let g = to_int(c1["g"] * (1.0 - t) + c2["g"] * t)
    let b = to_int(c1["b"] * (1.0 - t) + c2["b"] * t)
    let a = to_int(c1["a"] * (1.0 - t) + c2["a"] * t)
    return {r: r, g: g, b: b, a: a}

/* ============================================
   DRAWING FUNCTIONS (Convenience)
   ============================================ */

/* Draw a regular polygon */
func draw_polygon(canvas: Canvas, x: Float, y: Float,radius: Float, sides: Int,color: Dict, filled: Bool):
    import "stdlib/math.qc" as math

    let angle_step = math.TAU / to_float(sides)
    mut i = 0

    while i < sides:
        let angle1 = angle_step * to_float(i)
        let angle2 = angle_step * to_float(i + 1)

        let x1 = x + radius * math.cos(angle1)
        let y1 = y + radius * math.sin(angle1)
        let x2 = x + radius * math.cos(angle2)
        let y2 = y + radius * math.sin(angle2)

        canvas.draw_line(x1, y1, x2, y2, color, 2.0)
        i = i + 1

/* Draw a star */
func draw_star(canvas: Canvas, x: Float, y: Float,outer_radius: Float, inner_radius: Float,points: Int, color: Dict):
    import "stdlib/math.qc" as math

    let angle_step = math.TAU / to_float(points * 2)
    mut i = 0

    while i < points * 2:
        let angle1 = angle_step * to_float(i)
        let angle2 = angle_step * to_float(i + 1)

        let r1 = if i % 2 == 0: outer_radius else: inner_radius
        let r2 = if (i + 1) % 2 == 0: outer_radius else: inner_radius

        let x1 = x + r1 * math.cos(angle1)
        let y1 = y + r1 * math.sin(angle1)
        let x2 = x + r2 * math.cos(angle2)
        let y2 = y + r2 * math.sin(angle2)

        canvas.draw_line(x1, y1, x2, y2, color, 2.0)
        i = i + 1

/* Draw a grid */
func draw_grid(canvas: Canvas, spacing: Float, color: Dict):
    let w = to_float(canvas.width)
    let h = to_float(canvas.height)

    // Vertical lines
    mut x = 0.0
    while x <= w:
        canvas.draw_line(x, 0.0, x, h, color, 1.0)
        x = x + spacing

    // Horizontal lines
    mut y = 0.0
    while y <= h:
        canvas.draw_line(0.0, y, w, y, color, 1.0)
        y = y + spacing

/* Draw axes */
func draw_axes(canvas: Canvas, origin_x: Float, origin_y: Float,length: Float, color: Dict):
    // X-axis
    canvas.draw_line(
        origin_x - length, origin_y,
        origin_x + length, origin_y,
        color, 2.0
    )

    // Y-axis
    canvas.draw_line(
        origin_x, origin_y - length,
        origin_x, origin_y + length,
        color, 2.0
    )

    // Arrow heads
    let arrow_size = 10.0
    canvas.draw_line(
        origin_x + length, origin_y,
        origin_x + length - arrow_size, origin_y - arrow_size / 2.0,
        color, 2.0
    )
    canvas.draw_line(
        origin_x + length, origin_y,
        origin_x + length - arrow_size, origin_y + arrow_size / 2.0,
        color, 2.0
    )

/* ============================================
   PLOTTING FUNCTIONS
   ============================================ */

/* Quick plot function for line plots */
func quick_plot(x_data: Float[], y_data: Float[],title: String, filename: String):
    let canvas = Canvas(800, 600)
    canvas.set_background(COLOR_WHITE)

    let plot = Plot(PLOT_LINE)
    plot.set_data(x_data, y_data)
    plot.set_title(title)
    plot.render(canvas)

    canvas.save_svg(filename)
    canvas.destroy()
    plot.destroy()

/* Scatter plot */
func scatter_plot(x_data: Float[], y_data: Float[],title: String, filename: String):
    let canvas = Canvas(800, 600)
    canvas.set_background(COLOR_WHITE)

    let plot = Plot(PLOT_SCATTER)
    plot.set_data(x_data, y_data)
    plot.set_title(title)
    plot.render(canvas)

    canvas.save_svg(filename)
    canvas.destroy()
    plot.destroy()

/* Bar chart */
func bar_chart(x_data: Float[], y_data: Float[],title: String, filename: String):
    let canvas = Canvas(800, 600)
    canvas.set_background(COLOR_WHITE)

    let plot = Plot(PLOT_BAR)
    plot.set_data(x_data, y_data)
    plot.set_title(title)
    plot.render(canvas)

    canvas.save_svg(filename)
    canvas.destroy()
    plot.destroy()

/* ============================================
   VISUALIZATION EXAMPLES
   ============================================ */

/* Visualize quantum state probabilities */
func visualize_quantum_state(state: QuantumRegister,filename: String):
    // Create canvas
    let canvas = Canvas(800, 600)
    canvas.set_background(COLOR_WHITE)

    // Draw title
    canvas.draw_text(400.0, 30.0, "Quantum State Probabilities",
                    COLOR_BLACK, 24.0)

    // TODO: Extract probabilities from quantum state
    // and visualize as bar chart

    canvas.save_svg(filename)
    canvas.destroy()

/* Draw Bloch sphere representation */
func draw_bloch_sphere(canvas: Canvas, x: Float, y: Float,radius: Float):
    // Draw sphere
    canvas.draw_circle(x, y, radius, COLOR_BLUE, False)

    // Draw axes
    canvas.draw_line(x, y - radius, x, y + radius, COLOR_BLACK, 2.0)
    canvas.draw_line(x - radius, y, x + radius, y, COLOR_BLACK, 2.0)

    // Labels
    canvas.draw_text(x, y - radius - 20.0, "|0⟩", COLOR_BLACK, 16.0)
    canvas.draw_text(x, y + radius + 20.0, "|1⟩", COLOR_BLACK, 16.0)
    canvas.draw_text(x + radius + 10.0, y, "|+⟩", COLOR_BLACK, 16.0)
    canvas.draw_text(x - radius - 30.0, y, "|-⟩", COLOR_BLACK, 16.0)

/* ============================================
   ANIMATION SUPPORT
   ============================================ */

/* Create animation frames */
func create_animation(num_frames: Int,frame_func: Function,output_dir: String):
    mut i = 0
    while i < num_frames:
        let canvas = Canvas(800, 600)
        frame_func(canvas, i, num_frames)

        let filename = output_dir + "/frame_" + to_string(i) + ".svg"
        canvas.save_svg(filename)
        canvas.destroy()

        i = i + 1

    print("Created " + to_string(num_frames) + " animation frames")
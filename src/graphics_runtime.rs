// src/graphics_runtime.rs

use crate::graphics::{Canvas, Color, Plot, PlotType, Point2D};
use std::collections::HashMap;
use std::sync::Mutex;
use std::os::raw::{c_char, c_int};
use std::ffi::CStr;

lazy_static::lazy_static! {
    static ref CANVASES: Mutex<HashMap<i32, Canvas>> = Mutex::new(HashMap::new());
    static ref PLOTS: Mutex<HashMap<i32, Plot>> = Mutex::new(HashMap::new());
    static ref NEXT_CANVAS_ID: Mutex<i32> = Mutex::new(0);
    static ref NEXT_PLOT_ID: Mutex<i32> = Mutex::new(0);
}

// Canvas functions
#[no_mangle]
pub extern "C" fn quantica_graphics_create_canvas(width: c_int, height: c_int) -> c_int {
    let canvas = Canvas::new(width as u32, height as u32);

    let mut canvases = CANVASES.lock().unwrap();
    let mut next_id = NEXT_CANVAS_ID.lock().unwrap();

    let id = *next_id;
    *next_id += 1;

    canvases.insert(id, canvas);
    id
}

#[no_mangle]
pub extern "C" fn quantica_graphics_set_background(
    canvas_id: c_int,
    r: c_int,
    g: c_int,
    b: c_int,
    a: c_int
) {
    let mut canvases = CANVASES.lock().unwrap();
    if let Some(canvas) = canvases.get_mut(&canvas_id) {
        canvas.set_background(Color::new(r as u8, g as u8, b as u8, a as u8));
    }
}

#[no_mangle]
pub extern "C" fn quantica_graphics_clear(canvas_id: c_int) {
    let mut canvases = CANVASES.lock().unwrap();
    if let Some(canvas) = canvases.get_mut(&canvas_id) {
        canvas.clear();
    }
}

#[no_mangle]
pub extern "C" fn quantica_graphics_draw_line(
    canvas_id: c_int,
    x1: f64, y1: f64,
    x2: f64, y2: f64,
    r: c_int, g: c_int, b: c_int, a: c_int,
    width: f32
) {
    let mut canvases = CANVASES.lock().unwrap();
    if let Some(canvas) = canvases.get_mut(&canvas_id) {
        let color = Color::new(r as u8, g as u8, b as u8, a as u8);
        canvas.draw_line(x1, y1, x2, y2, color, width);
    }
}

#[no_mangle]
pub extern "C" fn quantica_graphics_draw_rect(
    canvas_id: c_int,
    x: f64, y: f64,
    width: f64, height: f64,
    r: c_int, g: c_int, b: c_int, a: c_int,
    filled: c_int
) {
    let mut canvases = CANVASES.lock().unwrap();
    if let Some(canvas) = canvases.get_mut(&canvas_id) {
        let color = Color::new(r as u8, g as u8, b as u8, a as u8);
        canvas.draw_rectangle(x, y, width, height, color, filled != 0);
    }
}

#[no_mangle]
pub extern "C" fn quantica_graphics_draw_circle(
    canvas_id: c_int,
    x: f64, y: f64,
    radius: f64,
    r: c_int, g: c_int, b: c_int, a: c_int,
    filled: c_int
) {
    let mut canvases = CANVASES.lock().unwrap();
    if let Some(canvas) = canvases.get_mut(&canvas_id) {
        let color = Color::new(r as u8, g as u8, b as u8, a as u8);
        canvas.draw_circle(x, y, radius, color, filled != 0);
    }
}

#[no_mangle]
pub unsafe extern "C" fn quantica_graphics_draw_text(
    canvas_id: c_int,
    x: f64, y: f64,
    text: *const c_char,
    r: c_int, g: c_int, b: c_int, a: c_int,
    size: f32
) {
    if text.is_null() {
        return;
    }

    let text_str = unsafe {
        CStr::from_ptr(text).to_string_lossy().into_owned()
    };

    let mut canvases = CANVASES.lock().unwrap();
    if let Some(canvas) = canvases.get_mut(&canvas_id) {
        let color = Color::new(r as u8, g as u8, b as u8, a as u8);
        canvas.draw_text(x, y, text_str, color, size);
    }
}

#[no_mangle]
pub unsafe extern "C" fn quantica_graphics_save_svg(
    canvas_id: c_int,
    filename: *const c_char
) -> c_int {
    if filename.is_null() {
        return -1;
    }

    let filename_str = unsafe {
        CStr::from_ptr(filename).to_string_lossy().into_owned()
    };

    let canvases = CANVASES.lock().unwrap();
    if let Some(canvas) = canvases.get(&canvas_id) {
        match canvas.save_svg(&filename_str) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    } else {
        -1
    }
}

#[no_mangle]
pub unsafe extern "C" fn quantica_graphics_save_png(
    canvas_id: c_int,
    filename: *const c_char
) -> c_int {
    if filename.is_null() {
        return -1;
    }

    let filename_str = unsafe {
        CStr::from_ptr(filename).to_string_lossy().into_owned()
    };

    let canvases = CANVASES.lock().unwrap();
    if let Some(canvas) = canvases.get(&canvas_id) {
        match canvas.save_png(&filename_str) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    } else {
        -1
    }
}

// Plot functions
#[no_mangle]
pub extern "C" fn quantica_graphics_create_plot(plot_type: c_int) -> c_int {
    let ptype = match plot_type {
        0 => PlotType::Line,
        1 => PlotType::Scatter,
        2 => PlotType::Bar,
        3 => PlotType::Histogram,
        4 => PlotType::Heatmap,
        _ => PlotType::Line,
    };

    let plot = Plot::new(ptype);

    let mut plots = PLOTS.lock().unwrap();
    let mut next_id = NEXT_PLOT_ID.lock().unwrap();

    let id = *next_id;
    *next_id += 1;

    plots.insert(id, plot);
    id
}

#[no_mangle]
pub unsafe extern "C" fn quantica_graphics_plot_set_data(
    plot_id: c_int,
    x_data: *const f64,
    y_data: *const f64,
    len: c_int
) {
    if x_data.is_null() || y_data.is_null() || len <= 0 {
        return;
    }

    let x_slice = unsafe { std::slice::from_raw_parts(x_data, len as usize) };
    let y_slice = unsafe { std::slice::from_raw_parts(y_data, len as usize) };

    let mut plots = PLOTS.lock().unwrap();
    if let Some(plot) = plots.get_mut(&plot_id) {
        plot.set_data(x_slice.to_vec(), y_slice.to_vec());
    }
}

#[no_mangle]
pub unsafe extern "C" fn quantica_graphics_plot_set_title(
    plot_id: c_int,
    title: *const c_char
) {
    if title.is_null() {
        return;
    }

    let title_str = unsafe {
        CStr::from_ptr(title).to_string_lossy().into_owned()
    };

    let mut plots = PLOTS.lock().unwrap();
    if let Some(plot) = plots.get_mut(&plot_id) {
        plot.title = title_str;
    }
}

#[no_mangle]
pub extern "C" fn quantica_graphics_plot_render(
    plot_id: c_int,
    canvas_id: c_int
) -> c_int {
    let plots = PLOTS.lock().unwrap();
    let mut canvases = CANVASES.lock().unwrap();

    if let (Some(plot), Some(canvas)) = (plots.get(&plot_id), canvases.get_mut(&canvas_id)) {
        match plot.render_to_canvas(canvas) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    } else {
        -1
    }
}

#[no_mangle]
pub extern "C" fn quantica_graphics_destroy_canvas(canvas_id: c_int) {
    let mut canvases = CANVASES.lock().unwrap();
    canvases.remove(&canvas_id);
}

#[no_mangle]
pub extern "C" fn quantica_graphics_destroy_plot(plot_id: c_int) {
    let mut plots = PLOTS.lock().unwrap();
    plots.remove(&plot_id);
}
// src/graphics/mod.rs

use std::collections::HashMap;
use serde::{Deserialize, Serialize};


/// Color representation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Color { r, g, b, a }
    }

    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Color { r, g, b, a: 255 }
    }

    pub fn from_hex(hex: &str) -> Result<Self, String> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 && hex.len() != 8 {
            return Err("Invalid hex color format".to_string());
        }

        let r = u8::from_str_radix(&hex[0..2], 16).map_err(|e| e.to_string())?;
        let g = u8::from_str_radix(&hex[2..4], 16).map_err(|e| e.to_string())?;
        let b = u8::from_str_radix(&hex[4..6], 16).map_err(|e| e.to_string())?;
        let a = if hex.len() == 8 {
            u8::from_str_radix(&hex[6..8], 16).map_err(|e| e.to_string())?
        } else {
            255
        };

        Ok(Color { r, g, b, a })
    }

    // Predefined colors
    pub const BLACK: Color = Color { r: 0, g: 0, b: 0, a: 255 };
    pub const WHITE: Color = Color { r: 255, g: 255, b: 255, a: 255 };
    pub const RED: Color = Color { r: 255, g: 0, b: 0, a: 255 };
    pub const GREEN: Color = Color { r: 0, g: 255, b: 0, a: 255 };
    pub const BLUE: Color = Color { r: 0, g: 0, b: 255, a: 255 };
    pub const YELLOW: Color = Color { r: 255, g: 255, b: 0, a: 255 };
    pub const CYAN: Color = Color { r: 0, g: 255, b: 255, a: 255 };
    pub const MAGENTA: Color = Color { r: 255, g: 0, b: 255, a: 255 };
}

/// 2D Point
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

/// 3D Point
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Graphics Window
pub struct Window {
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub background: Color,
}

impl Window {
    pub fn new(width: u32, height: u32, title: String) -> Self {
        Window {
            width,
            height,
            title,
            background: Color::WHITE,
        }
    }
}

/// Shape types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Shape {
    Line {
        start: Point2D,
        end: Point2D,
        color: Color,
        width: f32,
    },
    Rectangle {
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        color: Color,
        filled: bool,
    },
    Circle {
        center: Point2D,
        radius: f64,
        color: Color,
        filled: bool,
    },
    Polygon {
        points: Vec<Point2D>,
        color: Color,
        filled: bool,
    },
    Text {
        position: Point2D,
        text: String,
        color: Color,
        size: f32,
    },
}

/// Canvas for 2D drawing
pub struct Canvas {
    pub width: u32,
    pub height: u32,
    shapes: Vec<Shape>,
    background: Color,
}

impl Canvas {
    pub fn new(width: u32, height: u32) -> Self {
        Canvas {
            width,
            height,
            shapes: Vec::new(),
            background: Color::WHITE,
        }
    }

    pub fn set_background(&mut self, color: Color) {
        self.background = color;
    }

    pub fn clear(&mut self) {
        self.shapes.clear();
    }

    pub fn draw_line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, color: Color, width: f32) {
        self.shapes.push(Shape::Line {
            start: Point2D { x: x1, y: y1 },
            end: Point2D { x: x2, y: y2 },
            color,
            width,
        });
    }

    pub fn draw_rectangle(&mut self, x: f64, y: f64, width: f64, height: f64, color: Color, filled: bool) {
        self.shapes.push(Shape::Rectangle {
            x, y, width, height, color, filled,
        });
    }

    pub fn draw_circle(&mut self, x: f64, y: f64, radius: f64, color: Color, filled: bool) {
        self.shapes.push(Shape::Circle {
            center: Point2D { x, y },
            radius,
            color,
            filled,
        });
    }

    pub fn draw_polygon(&mut self, points: Vec<Point2D>, color: Color, filled: bool) {
        self.shapes.push(Shape::Polygon {
            points,
            color,
            filled,
        });
    }

    pub fn draw_text(&mut self, x: f64, y: f64, text: String, color: Color, size: f32) {
        self.shapes.push(Shape::Text {
            position: Point2D { x, y },
            text,
            color,
            size,
        });
    }

    pub fn save_svg(&self, filename: &str) -> Result<(), String> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(filename).map_err(|e| e.to_string())?;

        writeln!(file, r#"<?xml version="1.0" encoding="UTF-8"?>"#).map_err(|e| e.to_string())?;
        writeln!(file, r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
                 self.width, self.height).map_err(|e| e.to_string())?;

        // Background
        writeln!(file, r#"  <rect width="100%" height="100%" fill="rgb({},{},{})"/>"#,
                 self.background.r, self.background.g, self.background.b).map_err(|e| e.to_string())?;

        // Draw shapes
        for shape in &self.shapes {
            match shape {
                Shape::Line { start, end, color, width } => {
                    writeln!(file, r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="rgb({},{},{})" stroke-width="{}"/>"#,
                             start.x, start.y, end.x, end.y, color.r, color.g, color.b, width).map_err(|e| e.to_string())?;
                }
                Shape::Rectangle { x, y, width, height, color, filled } => {
                    if *filled {
                        writeln!(file, r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="rgb({},{},{})"/>"#,
                                 x, y, width, height, color.r, color.g, color.b).map_err(|e| e.to_string())?;
                    } else {
                        writeln!(file, r#"  <rect x="{}" y="{}" width="{}" height="{}" stroke="rgb({},{},{})" fill="none"/>"#,
                                 x, y, width, height, color.r, color.g, color.b).map_err(|e| e.to_string())?;
                    }
                }
                Shape::Circle { center, radius, color, filled } => {
                    if *filled {
                        writeln!(file, r#"  <circle cx="{}" cy="{}" r="{}" fill="rgb({},{},{})"/>"#,
                                 center.x, center.y, radius, color.r, color.g, color.b).map_err(|e| e.to_string())?;
                    } else {
                        writeln!(file, r#"  <circle cx="{}" cy="{}" r="{}" stroke="rgb({},{},{})" fill="none"/>"#,
                                 center.x, center.y, radius, color.r, color.g, color.b).map_err(|e| e.to_string())?;
                    }
                }
                Shape::Polygon { points, color, filled } => {
                    let points_str: Vec<String> = points.iter().map(|p| format!("{},{}", p.x, p.y)).collect();
                    if *filled {
                        writeln!(file, r#"  <polygon points="{}" fill="rgb({},{},{})"/>"#,
                                 points_str.join(" "), color.r, color.g, color.b).map_err(|e| e.to_string())?;
                    } else {
                        writeln!(file, r#"  <polygon points="{}" stroke="rgb({},{},{})" fill="none"/>"#,
                                 points_str.join(" "), color.r, color.g, color.b).map_err(|e| e.to_string())?;
                    }
                }
                Shape::Text { position, text, color, size } => {
                    writeln!(file, r#"  <text x="{}" y="{}" font-size="{}" fill="rgb({},{},{})">{}</text>"#,
                             position.x, position.y, size, color.r, color.g, color.b, text).map_err(|e| e.to_string())?;
                }
            }
        }

        writeln!(file, "</svg>").map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn save_png(&self, filename: &str) -> Result<(), String> {
        // For PNG export, we'd use a library like image or cairo
        // This is a placeholder that generates a simple PPM format
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(filename).map_err(|e| e.to_string())?;

        // Simple PPM format (ASCII)
        writeln!(file, "P3").map_err(|e| e.to_string())?;
        writeln!(file, "{} {}", self.width, self.height).map_err(|e| e.to_string())?;
        writeln!(file, "255").map_err(|e| e.to_string())?;

        // Create pixel buffer
        let mut pixels = vec![self.background; (self.width * self.height) as usize];

        // Rasterize shapes (simple implementation)
        for shape in &self.shapes {
            match shape {
                Shape::Circle { center, radius, color, filled } => {
                    for y in 0..self.height {
                        for x in 0..self.width {
                            let dx = x as f64 - center.x;
                            let dy = y as f64 - center.y;
                            let dist = (dx * dx + dy * dy).sqrt();

                            if *filled && dist <= *radius {
                                pixels[(y * self.width + x) as usize] = *color;
                            } else if !*filled && (dist - *radius).abs() < 1.0 {
                                pixels[(y * self.width + x) as usize] = *color;
                            }
                        }
                    }
                }
                _ => {} // Implement other shapes as needed
            }
        }

        // Write pixels
        for pixel in pixels {
            writeln!(file, "{} {} {}", pixel.r, pixel.g, pixel.b).map_err(|e| e.to_string())?;
        }

        Ok(())
    }
}

/// Plot types
#[derive(Debug, Clone)]
pub enum PlotType {
    Line,
    Scatter,
    Bar,
    Histogram,
    Heatmap,
}

/// Plot data
pub struct Plot {
    pub plot_type: PlotType,
    pub x_data: Vec<f64>,
    pub y_data: Vec<f64>,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub color: Color,
}

impl Plot {
    pub fn new(plot_type: PlotType) -> Self {
        Plot {
            plot_type,
            x_data: Vec::new(),
            y_data: Vec::new(),
            title: String::new(),
            x_label: String::new(),
            y_label: String::new(),
            color: Color::BLUE,
        }
    }

    pub fn set_data(&mut self, x: Vec<f64>, y: Vec<f64>) {
        self.x_data = x;
        self.y_data = y;
    }

    pub fn render_to_canvas(&self, canvas: &mut Canvas) -> Result<(), String> {
        if self.x_data.len() != self.y_data.len() {
            return Err("X and Y data must have same length".to_string());
        }

        if self.x_data.is_empty() {
            return Err("No data to plot".to_string());
        }

        // Calculate bounds
        let margin = 50.0;
        let plot_width = canvas.width as f64 - 2.0 * margin;
        let plot_height = canvas.height as f64 - 2.0 * margin;

        let x_min = self.x_data.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = self.x_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = self.y_data.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = self.y_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        // Draw axes
        canvas.draw_line(margin, canvas.height as f64 - margin,
                        canvas.width as f64 - margin, canvas.height as f64 - margin,
                        Color::BLACK, 2.0);
        canvas.draw_line(margin, margin, margin, canvas.height as f64 - margin,
                        Color::BLACK, 2.0);

        // Draw title
        if !self.title.is_empty() {
            canvas.draw_text(canvas.width as f64 / 2.0, 30.0,
                           self.title.clone(), Color::BLACK, 20.0);
        }

        // Draw labels
        if !self.x_label.is_empty() {
            canvas.draw_text(canvas.width as f64 / 2.0, canvas.height as f64 - 10.0,
                           self.x_label.clone(), Color::BLACK, 14.0);
        }

        // Plot data
        match self.plot_type {
            PlotType::Line => {
                for i in 0..self.x_data.len() - 1 {
                    let x1 = margin + (self.x_data[i] - x_min) / x_range * plot_width;
                    let y1 = canvas.height as f64 - margin - (self.y_data[i] - y_min) / y_range * plot_height;
                    let x2 = margin + (self.x_data[i + 1] - x_min) / x_range * plot_width;
                    let y2 = canvas.height as f64 - margin - (self.y_data[i + 1] - y_min) / y_range * plot_height;

                    canvas.draw_line(x1, y1, x2, y2, self.color, 2.0);
                }
            }
            PlotType::Scatter => {
                for i in 0..self.x_data.len() {
                    let x = margin + (self.x_data[i] - x_min) / x_range * plot_width;
                    let y = canvas.height as f64 - margin - (self.y_data[i] - y_min) / y_range * plot_height;

                    canvas.draw_circle(x, y, 3.0, self.color, true);
                }
            }
            PlotType::Bar => {
                let bar_width = plot_width / self.x_data.len() as f64 * 0.8;
                for i in 0..self.x_data.len() {
                    let x = margin + (self.x_data[i] - x_min) / x_range * plot_width - bar_width / 2.0;
                    let height = (self.y_data[i] - y_min) / y_range * plot_height;
                    let y = canvas.height as f64 - margin - height;

                    canvas.draw_rectangle(x, y, bar_width, height, self.color, true);
                }
            }
            _ => return Err(format!("Plot type {:?} not yet implemented", self.plot_type)),
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_creation() {
        let c = Color::rgb(255, 0, 0);
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 0);
    }

    #[test]
    fn test_canvas_drawing() {
        let mut canvas = Canvas::new(800, 600);
        canvas.draw_line(0.0, 0.0, 100.0, 100.0, Color::RED, 2.0);
        canvas.draw_circle(400.0, 300.0, 50.0, Color::BLUE, true);
        assert_eq!(canvas.shapes.len(), 2);
    }
}
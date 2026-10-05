extern crate nalgebra as na;

pub mod app;
pub mod axis_map;
pub mod demo;
pub mod geometry;
#[cfg(not(target_arch = "wasm32"))]
pub mod headless;
#[cfg(not(target_arch = "wasm32"))]
pub mod native;
pub mod render;
pub mod scene;
#[cfg(target_arch = "wasm32")]
pub mod web;

pub use demo::run;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{run_native, NativeArgs};

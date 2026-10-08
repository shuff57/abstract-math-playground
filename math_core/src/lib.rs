//! `math_core`: the pure-logic math engine.
//!
//! No GPU, windowing or browser dependencies, so everything here unit-tests natively and
//! compiles unchanged to WASM. Pipeline: [`parse`] -> [`analyze`] -> [`compile`].

pub mod actions;
pub mod analyze;
pub mod ast;
pub mod calculus;
pub mod compile;
pub mod complex;
pub mod drag;
pub mod doc;
pub mod intersect;
pub mod interval;
pub mod list;
pub mod mesh;
pub mod mesh_contour;
pub mod mesh_height;
pub mod mesh_param;
pub mod mesh_surface;
pub mod mesh_touch;
pub mod param;
pub mod parse;
pub mod print;
pub mod reg_family;
pub mod regress;
pub mod resolve;
pub mod slice;
pub mod special;
pub mod stats;
pub mod table;
pub mod view;
pub mod wgsl;
pub mod wgsl_complex;

pub use ast::{BinOp, Expr, Rel};
pub use compile::{compile, Angle, CompileError, Program};
pub use complex::{compile_complex, parse_complex, CProgram, C64};
pub use interval::Interval;
pub use parse::{parse, parse_with, ParseCtx, ParseError};
pub use resolve::{Defs, ResolveError};

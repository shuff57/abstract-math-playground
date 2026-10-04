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
pub mod doc;
pub mod interval;
pub mod list;
pub mod mesh;
pub mod parse;
pub mod regress;
pub mod print;
pub mod resolve;
pub mod slice;
pub mod stats;
pub mod table;
pub mod view;
pub mod wgsl;
pub mod wgsl_complex;

pub use ast::{BinOp, Expr, Rel};
pub use compile::{compile, Angle, CompileError, Program};
pub use complex::{compile_complex, parse_complex, CProgram, C64};
pub use interval::Interval;
pub use resolve::{Defs, ResolveError};
pub use parse::{parse, parse_with, ParseCtx, ParseError};

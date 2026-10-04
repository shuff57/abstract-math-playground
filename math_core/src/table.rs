//! Data tables: columns of expression cells, each column named like a list variable
//! (`x_1`, `y_1`). The scene turns every column into a list definition, so `histogram(x_1)` or a
//! regression against `y_1` work exactly as with a hand-typed list, and plots the first two
//! columns as points (optionally joined by a line).
//!
//! Pure data and validation here; drawing lives in `math_playground::scene`.

use crate::ast::Expr;
use crate::parse::{parse_with, ParseCtx};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const MAX_COLUMNS: usize = 12;
pub const MAX_ROWS: usize = 500;
pub const MAX_CELL_CHARS: usize = 120;
pub const MAX_NAME_CHARS: usize = 16;

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TableStyle {
    /// First two columns drawn as dots (the default).
    #[default]
    Points,
    /// Dots joined by a polyline in row order.
    Line,
    /// Not drawn; the column lists are still defined.
    Hidden,
}

impl TableStyle {
    pub fn parse(s: &str) -> Option<TableStyle> {
        match s {
            "points" => Some(TableStyle::Points),
            "line" => Some(TableStyle::Line),
            "hidden" => Some(TableStyle::Hidden),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            TableStyle::Points => "points",
            TableStyle::Line => "line",
            TableStyle::Hidden => "hidden",
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Column {
    /// List variable name, e.g. `x_1`.
    pub name: String,
    /// Cell source text, one per row (empty string = blank cell).
    pub cells: Vec<String>,
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Table {
    pub columns: Vec<Column>,
    #[serde(default)]
    pub style: TableStyle,
}

/// Is `name` usable as a column (list) name: it must lex as ONE variable and not collide with a
/// spatial/special symbol.
pub fn valid_column_name(name: &str) -> bool {
    if name.is_empty() || name.chars().count() > MAX_NAME_CHARS {
        return false;
    }
    if matches!(name, "x" | "y" | "z" | "t" | "r" | "e" | "i" | "theta" | "pi" | "tau") {
        return false;
    }
    matches!(parse_with(name, &ParseCtx::new()), Ok(Expr::Var(ref n)) if n == name)
}

/// The cells of one column, parsed (row-aligned: blank or unparsable cells are `None`).
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedColumn {
    pub name: String,
    pub cells: Vec<Option<Expr>>,
}

impl ParsedColumn {
    /// The defined list: the non-blank cells in order.
    pub fn list(&self) -> Expr {
        Expr::List(self.cells.iter().flatten().cloned().collect())
    }
}

impl Table {
    /// A new table with `rows` blank rows and the given column names (default `x_1`, `y_1`).
    pub fn new(names: &[String], rows: usize) -> Table {
        let names: Vec<String> = if names.is_empty() { vec!["x_1".into(), "y_1".into()] } else { names.to_vec() };
        Table {
            columns: names.into_iter().map(|name| Column { name, cells: vec![String::new(); rows] }).collect(),
            style: TableStyle::Points,
        }
    }

    pub fn rows(&self) -> usize {
        self.columns.iter().map(|c| c.cells.len()).max().unwrap_or(0)
    }

    fn pad(&mut self) {
        let n = self.rows();
        for c in &mut self.columns {
            c.cells.resize(n, String::new());
        }
    }

    pub fn set_cell(&mut self, row: usize, col: usize, value: &str) -> Result<(), String> {
        if col >= self.columns.len() {
            return Err(format!("no column {col}"));
        }
        if row >= MAX_ROWS {
            return Err(format!("too many rows (max {MAX_ROWS})"));
        }
        if value.chars().count() > MAX_CELL_CHARS {
            return Err(format!("cell longer than {MAX_CELL_CHARS} characters"));
        }
        if row >= self.rows() {
            for c in &mut self.columns {
                c.cells.resize(row + 1, String::new());
            }
        }
        self.columns[col].cells[row] = value.to_string();
        Ok(())
    }

    /// Inserts a blank row at `at` (appends when `None` or past the end).
    pub fn add_row(&mut self, at: Option<usize>) -> Result<(), String> {
        self.pad();
        let n = self.rows();
        if n >= MAX_ROWS {
            return Err(format!("too many rows (max {MAX_ROWS})"));
        }
        let at = at.unwrap_or(n).min(n);
        for c in &mut self.columns {
            c.cells.insert(at, String::new());
        }
        Ok(())
    }

    pub fn remove_row(&mut self, row: usize) -> Result<(), String> {
        self.pad();
        if row >= self.rows() {
            return Err(format!("no row {row}"));
        }
        for c in &mut self.columns {
            c.cells.remove(row);
        }
        Ok(())
    }

    /// The first unused `<letter>_<n>` name: continues the naming of the last column
    /// (`x_1`, `y_1` -> `x_2`, then `y_2`...) so headers stay readable.
    pub fn next_name(&self) -> String {
        let used: HashSet<&str> = self.columns.iter().map(|c| c.name.as_str()).collect();
        let letters = ["x", "y", "a", "b", "c", "d", "u", "v", "w", "p", "q", "s"];
        for n in 1..100 {
            for l in letters {
                let cand = format!("{l}_{n}");
                if !used.contains(cand.as_str()) {
                    return cand;
                }
            }
        }
        "w_100".into()
    }

    pub fn add_column(&mut self, name: Option<&str>) -> Result<(), String> {
        if self.columns.len() >= MAX_COLUMNS {
            return Err(format!("too many columns (max {MAX_COLUMNS})"));
        }
        let name = match name {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => self.next_name(),
        };
        if !valid_column_name(&name) {
            return Err(format!("'{name}' is not a valid column name (use a letter and a subscript, like x_1)"));
        }
        if self.columns.iter().any(|c| c.name == name) {
            return Err(format!("a column named '{name}' already exists"));
        }
        let rows = self.rows();
        self.columns.push(Column { name, cells: vec![String::new(); rows] });
        Ok(())
    }

    pub fn remove_column(&mut self, col: usize) -> Result<(), String> {
        if col >= self.columns.len() {
            return Err(format!("no column {col}"));
        }
        self.columns.remove(col);
        Ok(())
    }

    pub fn rename_column(&mut self, col: usize, name: &str) -> Result<(), String> {
        if col >= self.columns.len() {
            return Err(format!("no column {col}"));
        }
        if !valid_column_name(name) {
            return Err(format!("'{name}' is not a valid column name (use a letter and a subscript, like x_1)"));
        }
        if self.columns.iter().enumerate().any(|(i, c)| i != col && c.name == name) {
            return Err(format!("a column named '{name}' already exists"));
        }
        self.columns[col].name = name.to_string();
        Ok(())
    }

    /// Structural checks used by document validation.
    pub fn validate(&self) -> Result<(), String> {
        if self.columns.len() > MAX_COLUMNS {
            return Err(format!("too many columns (max {MAX_COLUMNS})"));
        }
        let mut seen = HashSet::new();
        for c in &self.columns {
            if !valid_column_name(&c.name) {
                return Err(format!("bad column name {:?}", c.name.chars().take(20).collect::<String>()));
            }
            if !seen.insert(c.name.as_str()) {
                return Err(format!("duplicate column name {:?}", c.name));
            }
            if c.cells.len() > MAX_ROWS {
                return Err(format!("too many rows (max {MAX_ROWS})"));
            }
            if c.cells.iter().any(|s| s.chars().count() > MAX_CELL_CHARS) {
                return Err(format!("cell longer than {MAX_CELL_CHARS} characters"));
            }
        }
        Ok(())
    }

    /// Parses every cell. Errors are `(column index, row, message)`; such cells become `None`.
    pub fn parse(&self, ctx: &ParseCtx) -> (Vec<ParsedColumn>, Vec<(usize, usize, String)>) {
        let mut errs = Vec::new();
        let cols = self
            .columns
            .iter()
            .enumerate()
            .map(|(ci, c)| ParsedColumn {
                name: c.name.clone(),
                cells: c
                    .cells
                    .iter()
                    .enumerate()
                    .map(|(ri, s)| {
                        if s.trim().is_empty() {
                            return None;
                        }
                        match parse_with(s, ctx) {
                            Ok(e) => Some(e),
                            Err(e) => {
                                errs.push((ci, ri, e.to_string()));
                                None
                            }
                        }
                    })
                    .collect(),
            })
            .collect();
        (cols, errs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t() -> Table {
        Table::new(&[], 2)
    }

    #[test]
    fn default_table_has_x1_y1() {
        let t = t();
        assert_eq!(t.columns.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["x_1", "y_1"]);
        assert_eq!(t.rows(), 2);
    }

    #[test]
    fn names_must_lex_as_one_variable() {
        for ok in ["x_1", "y_2", "a", "u_10", "L"] {
            assert!(valid_column_name(ok), "{ok}");
        }
        for bad in ["", "x", "y", "t", "e", "pi", "height", "x y", "1x", "sin", "x_1+1"] {
            assert!(!valid_column_name(bad), "{bad}");
        }
    }

    #[test]
    fn cell_edits_grow_rows_and_respect_limits() {
        let mut t = t();
        t.set_cell(4, 0, "3").unwrap();
        assert_eq!(t.rows(), 5);
        assert_eq!(t.columns[1].cells.len(), 5);
        assert!(t.set_cell(0, 9, "1").is_err());
        assert!(t.set_cell(MAX_ROWS, 0, "1").is_err());
        assert!(t.set_cell(0, 0, &"1".repeat(MAX_CELL_CHARS + 1)).is_err());
    }

    #[test]
    fn rows_add_remove() {
        let mut t = t();
        t.set_cell(0, 0, "1").unwrap();
        t.add_row(Some(0)).unwrap();
        assert_eq!(t.columns[0].cells, ["", "1", ""]);
        t.remove_row(0).unwrap();
        assert_eq!(t.columns[0].cells, ["1", ""]);
        assert!(t.remove_row(7).is_err());
        t.add_row(None).unwrap();
        assert_eq!(t.rows(), 3);
    }

    #[test]
    fn columns_add_remove_rename() {
        let mut t = t();
        t.add_column(None).unwrap();
        assert_eq!(t.columns[2].name, "a_1");
        assert_eq!(t.columns[2].cells.len(), 2);
        assert!(t.add_column(Some("x_1")).is_err());
        assert!(t.add_column(Some("height")).is_err());
        t.rename_column(2, "w_1").unwrap();
        assert!(t.rename_column(2, "y_1").is_err());
        t.remove_column(0).unwrap();
        assert_eq!(t.columns.len(), 2);
        assert!(t.remove_column(5).is_err());
        while t.columns.len() < MAX_COLUMNS {
            t.add_column(None).unwrap();
        }
        assert!(t.add_column(None).is_err());
    }

    #[test]
    fn parse_reports_bad_cells_and_keeps_alignment() {
        let mut t = t();
        t.set_cell(0, 0, "1").unwrap();
        t.set_cell(1, 0, "2+").unwrap();
        t.set_cell(2, 0, "3").unwrap();
        let (cols, errs) = t.parse(&ParseCtx::new());
        assert_eq!(errs.len(), 1);
        assert_eq!((errs[0].0, errs[0].1), (0, 1));
        assert_eq!(cols[0].cells.len(), 3);
        assert!(cols[0].cells[1].is_none());
        assert_eq!(cols[0].list(), Expr::List(vec![Expr::Num(1.0), Expr::Num(3.0)]));
    }

    #[test]
    fn json_round_trip() {
        let mut t = t();
        t.set_cell(0, 1, "a+1").unwrap();
        t.style = TableStyle::Line;
        let s = serde_json::to_string(&t).unwrap();
        assert!(s.contains("\"style\":\"line\""));
        let back: Table = serde_json::from_str(&s).unwrap();
        assert_eq!(back, t);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn validate_rejects_duplicates_and_bad_names() {
        let mut t = t();
        t.columns[1].name = "x_1".into();
        assert!(t.validate().is_err());
        t.columns[1].name = "height".into();
        assert!(t.validate().is_err());
    }
}

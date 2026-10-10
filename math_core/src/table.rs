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
    pub fn is_points(&self) -> bool {
        *self == TableStyle::Points
    }

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

fn is_false(b: &bool) -> bool {
    !*b
}

fn is_true(b: &bool) -> bool {
    *b
}

fn yes() -> bool {
    true
}

/// How one y column of a table is drawn (every column after the first is its own point set
/// against the first column, as in Desmos). Every field is optional and serialised only when it
/// differs from the default, so documents written before per-column styles load unchanged.
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ColumnStyle {
    /// `#rrggbb`; absent: the next colour of the theme palette.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// Not drawn at all (the column list is still defined).
    #[serde(default, skip_serializing_if = "is_false")]
    pub hidden: bool,
    /// Draw the points (default on).
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub points: bool,
    /// Join the points with segments in row order.
    #[serde(default, skip_serializing_if = "is_false")]
    pub lines: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_style: Option<crate::doc::PointStyle>,
    /// Point diameter in pixels, `1..=40` (absent: the item's, else the built-in size).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_size: Option<f64>,
    /// `[0, 1]` (absent: the item's opacity, else opaque).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
    /// A thin background-coloured ring around each point.
    #[serde(default, skip_serializing_if = "is_false")]
    pub outline: bool,
    // TODO(drag): Desmos also has a per-column "Drag" toggle (move table points on the canvas,
    // writing the new values back into the cells). Not implemented yet; it needs a drag handle
    // per table point in `math_playground::app` and a `setCell` write-back.
}

impl Default for ColumnStyle {
    fn default() -> Self {
        ColumnStyle {
            color: None,
            hidden: false,
            points: true,
            lines: false,
            point_style: None,
            point_size: None,
            opacity: None,
            outline: false,
        }
    }
}

/// Keys accepted by [`ColumnStyle::merge`] (camelCase, as serialised).
pub const COLUMN_STYLE_KEYS: [&str; 8] =
    ["color", "hidden", "points", "lines", "pointStyle", "pointSize", "opacity", "outline"];

impl ColumnStyle {
    pub fn is_default(&self) -> bool {
        *self == ColumnStyle::default()
    }

    /// Range and format checks: colour `#rrggbb`, opacity in `[0, 1]`, point size in `[1, 40]`.
    pub fn validate(&self) -> Result<(), String> {
        let mut e = Vec::new();
        if self.color.as_deref().is_some_and(|c| !crate::doc::is_hex_color(c)) {
            e.push("color must match #rrggbb".to_string());
        }
        if self.opacity.is_some_and(|o| !(o.is_finite() && (0.0..=1.0).contains(&o))) {
            e.push("opacity must be in [0,1]".to_string());
        }
        if self
            .point_size
            .is_some_and(|s| !(s.is_finite() && crate::doc::POINT_SIZE_RANGE.contains(&s)))
        {
            e.push("pointSize must be in [1,40]".to_string());
        }
        if e.is_empty() {
            Ok(())
        } else {
            Err(e.join("; "))
        }
    }

    /// This style with `patch` merged in (a `null` value resets that key to its default),
    /// validated. Unknown keys are an error.
    pub fn merge(&self, patch: &serde_json::Map<String, serde_json::Value>) -> Result<ColumnStyle, String> {
        if let Some(k) = patch.keys().find(|k| !COLUMN_STYLE_KEYS.contains(&k.as_str())) {
            return Err(format!(
                "unknown column style key '{k}' (expected one of {})",
                COLUMN_STYLE_KEYS.join(", ")
            ));
        }
        let mut obj = match serde_json::to_value(self) {
            Ok(serde_json::Value::Object(m)) => m,
            _ => serde_json::Map::new(),
        };
        for (k, v) in patch {
            if v.is_null() {
                obj.remove(k);
            } else {
                obj.insert(k.clone(), v.clone());
            }
        }
        let merged: ColumnStyle =
            serde_json::from_value(serde_json::Value::Object(obj)).map_err(|e| format!("column style: {e}"))?;
        merged.validate()?;
        Ok(merged)
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Column {
    /// List variable name, e.g. `x_1`.
    pub name: String,
    /// Cell source text, one per row (empty string = blank cell).
    pub cells: Vec<String>,
    /// How the column is drawn (ignored for the first, x, column). Additive field.
    #[serde(default, skip_serializing_if = "ColumnStyle::is_default")]
    pub style: ColumnStyle,
}

impl Column {
    /// A column with the default style.
    pub fn new(name: impl Into<String>, cells: Vec<String>) -> Column {
        Column { name: name.into(), cells, style: ColumnStyle::default() }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Table {
    pub columns: Vec<Column>,
    /// Legacy table-wide style. Documents load with it moved into every column's
    /// [`ColumnStyle`] (see [`Table::migrate_style`]), after which it is always `Points` and is
    /// no longer serialised.
    #[serde(default, skip_serializing_if = "TableStyle::is_points")]
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
            columns: names.into_iter().map(|name| Column::new(name, vec![String::new(); rows])).collect(),
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
        self.columns.push(Column::new(name, vec![String::new(); rows]));
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

    /// Moves a legacy table-wide [`TableStyle`] into every column (`Line` turns lines on,
    /// `Hidden` hides each column) and resets it to `Points`, so an old document draws exactly as
    /// before. Returns whether anything changed. Idempotent.
    pub fn migrate_style(&mut self) -> bool {
        let old = std::mem::take(&mut self.style);
        if old == TableStyle::Points {
            return false;
        }
        self.apply_table_style(old);
        true
    }

    /// Applies a table-wide style to every column (the legacy `setTableStyle` command).
    pub fn apply_table_style(&mut self, st: TableStyle) {
        self.style = TableStyle::Points;
        for c in &mut self.columns {
            match st {
                TableStyle::Points => {
                    c.style.hidden = false;
                    c.style.lines = false;
                }
                TableStyle::Line => {
                    c.style.hidden = false;
                    c.style.lines = true;
                }
                TableStyle::Hidden => c.style.hidden = true,
            }
        }
    }

    /// The table-wide style the columns add up to, for shells that only know the legacy
    /// control: `Hidden` when every y column is hidden, `Line` when every shown one has lines.
    pub fn summary_style(&self) -> TableStyle {
        let ys: Vec<&ColumnStyle> = self.columns.iter().skip(1).map(|c| &c.style).collect();
        if !ys.is_empty() && ys.iter().all(|c| c.hidden) {
            TableStyle::Hidden
        } else if ys.iter().any(|c| !c.hidden) && ys.iter().filter(|c| !c.hidden).all(|c| c.lines) {
            TableStyle::Line
        } else {
            TableStyle::Points
        }
    }

    /// Merges a style patch into column `col` (see [`ColumnStyle::merge`]).
    pub fn set_column_style(
        &mut self,
        col: usize,
        patch: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), String> {
        let c = self.columns.get_mut(col).ok_or_else(|| format!("no column {col}"))?;
        c.style = c.style.merge(patch)?;
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
            c.style.validate().map_err(|m| format!("column {:?}: {m}", c.name))?;
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

    fn patch(v: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn default_column_style_is_not_serialised() {
        let t = t();
        let s = serde_json::to_string(&t).unwrap();
        assert!(!s.contains("\"style\""), "{s}");
        assert_eq!(s, r#"{"columns":[{"name":"x_1","cells":["",""]},{"name":"y_1","cells":["",""]}]}"#);
    }

    #[test]
    fn column_style_round_trip() {
        let mut t = t();
        t.set_column_style(
            1,
            &patch(serde_json::json!({
                "color": "#c74440", "lines": true, "points": false, "pointStyle": "cross",
                "pointSize": 12, "opacity": 0.5, "outline": true, "hidden": true
            })),
        )
        .unwrap();
        let s = serde_json::to_string(&t).unwrap();
        assert!(s.contains(r##""style":{"color":"#c74440","hidden":true,"points":false,"lines":true,"pointStyle":"cross","pointSize":12.0,"opacity":0.5,"outline":true}"##), "{s}");
        let back: Table = serde_json::from_str(&s).unwrap();
        assert_eq!(back, t);
        assert!(back.validate().is_ok());
        // Null resets a key; the column returns to the default (and is not serialised again).
        let all_null: serde_json::Map<_, _> =
            COLUMN_STYLE_KEYS.iter().map(|k| (k.to_string(), serde_json::Value::Null)).collect();
        t.set_column_style(1, &all_null).unwrap();
        assert!(t.columns[1].style.is_default());
    }

    #[test]
    fn column_style_rejects_bad_values_and_keys() {
        let mut t = t();
        for bad in [
            serde_json::json!({"color": "red"}),
            serde_json::json!({"opacity": 2}),
            serde_json::json!({"pointSize": 0}),
            serde_json::json!({"pointStyle": "hexagon"}),
            serde_json::json!({"lines": "yes"}),
            serde_json::json!({"wobble": true}),
        ] {
            assert!(t.set_column_style(1, &patch(bad.clone())).is_err(), "{bad}");
        }
        assert!(t.columns[1].style.is_default(), "errors change nothing");
        assert!(t.set_column_style(7, &patch(serde_json::json!({"lines": true}))).is_err());
        // Hostile data in a loaded document fails validation.
        t.columns[1].style.opacity = Some(f64::NAN);
        assert!(t.validate().is_err());
    }

    #[test]
    fn legacy_table_style_moves_into_columns() {
        for (old, hidden, lines) in [("points", false, false), ("line", false, true), ("hidden", true, false)] {
            let json = format!(r#"{{"columns":[{{"name":"x_1","cells":["1"]}},{{"name":"y_1","cells":["2"]}}],"style":"{old}"}}"#);
            let mut t: Table = serde_json::from_str(&json).unwrap();
            assert_eq!(t.migrate_style(), old != "points");
            assert_eq!(t.style, TableStyle::Points);
            for c in &t.columns {
                assert_eq!((c.style.hidden, c.style.lines), (hidden, lines), "{old}");
            }
            assert!(!t.migrate_style(), "idempotent");
            assert!(!serde_json::to_string(&t).unwrap().contains(r#""style":""#));
        }
    }

    #[test]
    fn column_ops_keep_styles() {
        let mut t = t();
        t.set_column_style(1, &patch(serde_json::json!({"color": "#2d70b3"}))).unwrap();
        t.add_column(None).unwrap();
        assert!(t.columns[2].style.is_default());
        t.rename_column(1, "w_1").unwrap();
        assert_eq!(t.columns[1].style.color.as_deref(), Some("#2d70b3"));
        t.remove_column(0).unwrap();
        assert_eq!(t.columns[0].style.color.as_deref(), Some("#2d70b3"));
        assert_eq!(t.summary_style(), TableStyle::Points);
        t.apply_table_style(TableStyle::Line);
        assert!(t.columns.iter().all(|c| c.style.lines && !c.style.hidden));
        assert_eq!(t.summary_style(), TableStyle::Line);
        t.apply_table_style(TableStyle::Hidden);
        assert_eq!(t.summary_style(), TableStyle::Hidden);
        t.columns[1].style.hidden = false;
        assert_eq!(t.summary_style(), TableStyle::Line, "the shown column has lines");
        assert_eq!(t.columns[0].style.color.as_deref(), Some("#2d70b3"), "colours survive");
    }
}

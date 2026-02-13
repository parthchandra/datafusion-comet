// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Query execution context for error reporting
//!
//! This module provides QueryContext which mirrors Spark's SQLQueryContext
//! for providing SQL text, line/position information, and error location
//! pointers in exception messages.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Mirrors Spark's SQLQueryContext for error reporting.
///
/// Contains information about where an error occurred in a SQL query,
/// including the full SQL text, line/column positions, and object context.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QueryContext {
    /// Full SQL query text
    #[serde(rename = "sqlText")]
    pub sql_text: Arc<String>,

    /// Start offset in SQL text (0-based, character index)
    #[serde(rename = "startIndex")]
    pub start_index: i32,

    /// Stop offset in SQL text (0-based, character index, inclusive)
    #[serde(rename = "stopIndex")]
    pub stop_index: i32,

    /// Object type (e.g., "VIEW", "Project", "Filter")
    #[serde(rename = "objectType", skip_serializing_if = "Option::is_none")]
    pub object_type: Option<String>,

    /// Object name (e.g., view name, column name)
    #[serde(rename = "objectName", skip_serializing_if = "Option::is_none")]
    pub object_name: Option<String>,

    /// Line number in SQL query (1-based)
    pub line: i32,

    /// Column position within the line (0-based)
    #[serde(rename = "startPosition")]
    pub start_position: i32,
}

impl QueryContext {
    /// Creates a new QueryContext.
    ///
    /// # Arguments
    /// * `sql_text` - Full SQL query text
    /// * `start_index` - Start character offset (0-based)
    /// * `stop_index` - Stop character offset (0-based, inclusive)
    /// * `object_type` - Optional object type (e.g., "VIEW")
    /// * `object_name` - Optional object name
    /// * `line` - Line number (1-based)
    /// * `start_position` - Column position (0-based)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sql_text: String,
        start_index: i32,
        stop_index: i32,
        object_type: Option<String>,
        object_name: Option<String>,
        line: i32,
        start_position: i32,
    ) -> Self {
        Self {
            sql_text: Arc::new(sql_text),
            start_index,
            stop_index,
            object_type,
            object_name,
            line,
            start_position,
        }
    }

    /// Generate a summary string showing SQL fragment with error location.
    ///
    /// Format example:
    /// ```text
    /// == SQL of VIEW v1 (line 1, position 8) ==
    /// SELECT a/b FROM t
    ///        ^^^
    /// ```
    pub fn format_summary(&self) -> String {
        let start_idx = self.start_index.max(0) as usize;
        let stop_idx = (self.stop_index + 1).max(0) as usize;

        // Extract the problematic fragment
        let fragment = if start_idx < self.sql_text.len() && stop_idx <= self.sql_text.len() {
            &self.sql_text[start_idx..stop_idx]
        } else {
            ""
        };

        // Build the header line
        let mut summary = String::from("== SQL");

        if let Some(obj_type) = &self.object_type {
            if !obj_type.is_empty() {
                summary.push_str(" of ");
                summary.push_str(obj_type);

                if let Some(obj_name) = &self.object_name {
                    if !obj_name.is_empty() {
                        summary.push(' ');
                        summary.push_str(obj_name);
                    }
                }
            }
        }

        summary.push_str(&format!(
            " (line {}, position {}) ==\n",
            self.line,
            self.start_position + 1 // Convert 0-based to 1-based for display
        ));

        // Add the SQL text with fragment highlighted
        summary.push_str(&self.sql_text);
        summary.push('\n');

        // Add caret pointer
        let caret_position = self.start_position.max(0) as usize;
        summary.push_str(&" ".repeat(caret_position));
        summary.push_str(&"^".repeat(fragment.len().max(1)));

        summary
    }

    /// Returns the SQL fragment that caused the error.
    pub fn fragment(&self) -> String {
        let start_idx = self.start_index.max(0) as usize;
        let stop_idx = (self.stop_index + 1).max(0) as usize;

        if start_idx < self.sql_text.len() && stop_idx <= self.sql_text.len() {
            self.sql_text[start_idx..stop_idx].to_string()
        } else {
            String::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_context_creation() {
        let ctx = QueryContext::new(
            "SELECT a/b FROM t".to_string(),
            7,
            9,
            Some("Divide".to_string()),
            Some("a/b".to_string()),
            1,
            7,
        );

        assert_eq!(*ctx.sql_text, "SELECT a/b FROM t");
        assert_eq!(ctx.start_index, 7);
        assert_eq!(ctx.stop_index, 9);
        assert_eq!(ctx.object_type, Some("Divide".to_string()));
        assert_eq!(ctx.object_name, Some("a/b".to_string()));
        assert_eq!(ctx.line, 1);
        assert_eq!(ctx.start_position, 7);
    }

    #[test]
    fn test_query_context_serialization() {
        let ctx = QueryContext::new(
            "SELECT a/b FROM t".to_string(),
            7,
            9,
            Some("Divide".to_string()),
            Some("a/b".to_string()),
            1,
            7,
        );

        let json = serde_json::to_string(&ctx).unwrap();
        let deserialized: QueryContext = serde_json::from_str(&json).unwrap();

        assert_eq!(ctx, deserialized);
    }

    #[test]
    fn test_format_summary() {
        let ctx = QueryContext::new(
            "SELECT a/b FROM t".to_string(),
            7,
            9,
            Some("VIEW".to_string()),
            Some("v1".to_string()),
            1,
            7,
        );

        let summary = ctx.format_summary();

        assert!(summary.contains("== SQL of VIEW v1 (line 1, position 8) =="));
        assert!(summary.contains("SELECT a/b FROM t"));
        assert!(summary.contains("^^^")); // Three carets for "a/b"
    }

    #[test]
    fn test_format_summary_without_object() {
        let ctx = QueryContext::new(
            "SELECT a/b FROM t".to_string(),
            7,
            9,
            None,
            None,
            1,
            7,
        );

        let summary = ctx.format_summary();

        assert!(summary.contains("== SQL (line 1, position 8) =="));
        assert!(summary.contains("SELECT a/b FROM t"));
    }

    #[test]
    fn test_fragment() {
        let ctx = QueryContext::new(
            "SELECT a/b FROM t".to_string(),
            7,
            9,
            None,
            None,
            1,
            7,
        );

        assert_eq!(ctx.fragment(), "a/b");
    }

    #[test]
    fn test_arc_string_sharing() {
        let ctx1 = QueryContext::new(
            "SELECT a/b FROM t".to_string(),
            7,
            9,
            None,
            None,
            1,
            7,
        );

        let ctx2 = ctx1.clone();

        // Arc should share the same allocation
        assert!(Arc::ptr_eq(&ctx1.sql_text, &ctx2.sql_text));
    }

    #[test]
    fn test_json_with_optional_fields() {
        let ctx = QueryContext::new(
            "SELECT a/b FROM t".to_string(),
            7,
            9,
            None,
            None,
            1,
            7,
        );

        let json = serde_json::to_string(&ctx).unwrap();

        // Should not serialize objectType and objectName when None
        assert!(!json.contains("objectType"));
        assert!(!json.contains("objectName"));
    }
}

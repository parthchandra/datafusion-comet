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

//! Query context registry for error reporting in spark-expr
//!
//! This module provides QueryContextRegistry which stores QueryContext
//! information for expressions during execution, enabling rich error messages
//! with SQL text and position information.




use crate::QueryContext;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Registry that maps expression IDs to their QueryContext.
///
/// This registry is populated during plan deserialization and accessed
/// during error creation to attach SQL context to exceptions.
#[derive(Debug)]
pub struct QueryContextRegistry {
    /// Map from expression ID to QueryContext
    contexts: RwLock<HashMap<u64, Arc<QueryContext>>>,
}

impl QueryContextRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            contexts: RwLock::new(HashMap::new()),
        }
    }

    /// Register a QueryContext for an expression ID.
    ///
    /// If the expression ID already exists, it will be replaced.
    ///
    /// # Arguments
    /// * `expr_id` - Unique expression identifier from protobuf
    /// * `context` - QueryContext containing SQL text and position info
    pub fn register(&self, expr_id: u64, context: QueryContext) {
        let mut contexts = self.contexts.write().unwrap();
        contexts.insert(expr_id, Arc::new(context));
    }

    /// Get the QueryContext for an expression ID.
    ///
    /// Returns None if no context is registered for this expression.
    ///
    /// # Arguments
    /// * `expr_id` - Expression identifier to look up
    pub fn get(&self, expr_id: u64) -> Option<Arc<QueryContext>> {
        let contexts = self.contexts.read().unwrap();
        contexts.get(&expr_id).cloned()
    }

    /// Clear all registered contexts.
    ///
    /// This is typically called after plan execution completes to free memory.
    pub fn clear(&self) {
        let mut contexts = self.contexts.write().unwrap();
        contexts.clear();
    }

    /// Return the number of registered contexts (for debugging/testing)
    pub fn len(&self) -> usize {
        let contexts = self.contexts.read().unwrap();
        contexts.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for QueryContextRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-safe singleton for global access
// Note: In the future, this should be replaced with SessionState extension
use once_cell::sync::Lazy;

static GLOBAL_REGISTRY: Lazy<Arc<QueryContextRegistry>> =
    Lazy::new(|| Arc::new(QueryContextRegistry::new()));

/// Get the global QueryContextRegistry instance.
///
/// This provides access to the singleton registry. In a future enhancement,
/// this should be replaced with per-session registries stored in SessionState.
pub fn get_global_query_context_registry() -> Arc<QueryContextRegistry> {
    Arc::clone(&GLOBAL_REGISTRY)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_register_and_get() {
        let registry = QueryContextRegistry::new();

        let ctx = QueryContext::new(
            "SELECT a/b FROM t".to_string(),
            7,
            9,
            None,
            None,
            1,
            7,
        );

        registry.register(1, ctx.clone());

        let retrieved = registry.get(1).unwrap();
        assert_eq!(*retrieved.sql_text, "SELECT a/b FROM t");
        assert_eq!(retrieved.start_index, 7);
    }

    #[test]
    fn test_registry_get_nonexistent() {
        let registry = QueryContextRegistry::new();
        assert!(registry.get(999).is_none());
    }

    #[test]
    fn test_registry_clear() {
        let registry = QueryContextRegistry::new();

        let ctx = QueryContext::new(
            "SELECT a/b FROM t".to_string(),
            7,
            9,
            None,
            None,
            1,
            7,
        );

        registry.register(1, ctx);
        assert_eq!(registry.len(), 1);

        registry.clear();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }
}




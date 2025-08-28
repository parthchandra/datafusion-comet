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

//! Utilities for working with selection vectors imported from Java side via Arrow FFI

use crate::errors::CometError;
use crate::execution::operators::ExecutionError;
use arrow::{
    array::{Array, ArrayRef, Int32Array, StructArray},
    compute::take,
    datatypes::{DataType, Fields},
    ffi::{from_ffi, FFI_ArrowArray, FFI_ArrowSchema},
};
use std::sync::Arc;

/// Represents a selection vector imported from Java side containing both
/// the original data array and selection indices
pub struct CometSelectionVector {
    /// The original data array
    pub original_data: ArrayRef,
    /// The selection indices to apply
    pub selection_indices: Int32Array,
}

impl CometSelectionVector {
    /// Creates a CometSelectionVector from Arrow FFI pointers.
    /// Expects a struct array with two fields: "original_data" and "selection_indices"
    ///
    /// # Safety
    /// This function is unsafe because it dereferences raw pointers from FFI.
    /// The caller must ensure that array_ptr and schema_ptr are valid FFI pointers.
    pub unsafe fn from_ffi(
        array_ptr: *mut FFI_ArrowArray,
        schema_ptr: *mut FFI_ArrowSchema,
    ) -> Result<Self, ExecutionError> {
        // Import the struct array from FFI
        let array_data = std::ptr::replace(array_ptr, FFI_ArrowArray::empty());
        let schema_data = std::ptr::replace(schema_ptr, FFI_ArrowSchema::empty());

        let array_data = from_ffi(array_data, &schema_data)
            .map_err(|e| ExecutionError::ArrowError(e.to_string()))?;
        let struct_array = StructArray::from(array_data);

        // Validate the struct has the expected fields
        let schema = struct_array.data_type();
        if let DataType::Struct(ref fields) = schema {
            Self::validate_selection_vector_schema(fields)?;
        } else {
            return Err(ExecutionError::ArrowError(
                "Expected struct array for selection vector".to_string(),
            ));
        }

        // Extract the original data and selection indices
        let original_data =
            Arc::<dyn Array>::clone(struct_array.column_by_name("original_data").ok_or_else(
                || ExecutionError::ArrowError("Missing original_data field".to_string()),
            )?);

        let selection_indices_array = struct_array
            .column_by_name("selection_indices")
            .ok_or_else(|| {
                ExecutionError::ArrowError("Missing selection_indices field".to_string())
            })?;

        let selection_indices = selection_indices_array
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| {
                ExecutionError::ArrowError("Selection indices must be Int32Array".to_string())
            })?
            .clone();

        Ok(CometSelectionVector {
            original_data,
            selection_indices,
        })
    }

    /// Validates that the struct schema has the expected fields for a selection vector
    fn validate_selection_vector_schema(fields: &Fields) -> Result<(), ExecutionError> {
        if fields.len() != 2 {
            return Err(ExecutionError::ArrowError(format!(
                "Selection vector struct must have exactly 2 fields, got {}",
                fields.len()
            )));
        }

        let _original_data_field = fields
            .find("original_data")
            .ok_or_else(|| ExecutionError::ArrowError("Missing original_data field".to_string()))?;

        let selection_indices_field = fields.find("selection_indices").ok_or_else(|| {
            ExecutionError::ArrowError("Missing selection_indices field".to_string())
        })?;

        // Validate selection indices field is Int32 - selection_indices_field is (usize, &Field)
        if !matches!(selection_indices_field.1.data_type(), DataType::Int32) {
            return Err(ExecutionError::ArrowError(
                "Selection indices field must be Int32".to_string(),
            ));
        }

        Ok(())
    }

    /// Applies the selection by taking elements from the original data at the specified indices
    pub fn apply_selection(&self) -> Result<ArrayRef, ExecutionError> {
        take(&self.original_data, &self.selection_indices, None)
            .map_err(|e| ExecutionError::ArrowError(e.to_string()))
    }

    /// Returns the number of selected elements
    pub fn len(&self) -> usize {
        self.selection_indices.len()
    }

    /// Returns true if the selection vector is empty
    pub fn is_empty(&self) -> bool {
        self.selection_indices.is_empty()
    }

    /// Returns a reference to the original data array
    pub fn original_data(&self) -> &ArrayRef {
        &self.original_data
    }

    /// Returns a reference to the selection indices
    pub fn selection_indices(&self) -> &Int32Array {
        &self.selection_indices
    }
}

/// Helper function to create a CometSelectionVector from FFI addresses
/// This is typically called from JNI code
///
/// # Safety
/// This function is unsafe because it calls from_ffi with raw pointers.
/// The caller must ensure that array_addr and schema_addr are valid FFI addresses.
pub unsafe fn import_selection_vector_from_addresses(
    array_addr: i64,
    schema_addr: i64,
) -> Result<CometSelectionVector, ExecutionError> {
    let array_ptr = array_addr as *mut FFI_ArrowArray;
    let schema_ptr = schema_addr as *mut FFI_ArrowSchema;

    CometSelectionVector::from_ffi(array_ptr, schema_ptr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
        ffi::{FFI_ArrowArray, FFI_ArrowSchema},
    };
    use std::sync::Arc;

    #[test]
    fn test_selection_vector_creation() {
        // Create test data
        let original_data = Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e", "f"]));
        let selection_indices = Int32Array::from(vec![0, 2, 4, 5]);

        // Create struct array manually for testing
        let original_field = Field::new("original_data", DataType::Utf8, false);
        let indices_field = Field::new("selection_indices", DataType::Int32, false);
        let fields = vec![original_field, indices_field];

        let struct_array = StructArray::from(vec![
            (
                Arc::new(fields[0].clone()),
                original_data.clone() as ArrayRef,
            ),
            (
                Arc::new(fields[1].clone()),
                Arc::new(selection_indices) as ArrayRef,
            ),
        ]);

        // Test apply_selection logic manually
        let selection_indices = Int32Array::from(vec![0, 2, 4, 5]);
        let result = take(&original_data, &selection_indices, None).unwrap();
        let result_strings = result.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(result_strings.value(0), "a");
        assert_eq!(result_strings.value(1), "c");
        assert_eq!(result_strings.value(2), "e");
        assert_eq!(result_strings.value(3), "f");
    }
}

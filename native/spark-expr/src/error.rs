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

use arrow::error::ArrowError;
use datafusion::common::DataFusionError;

#[derive(thiserror::Error, Debug, Clone)]
pub enum SparkError {
    // ==================== Cast and Conversion Errors ====================

    // Note that this message format is based on Spark 3.4 and is more detailed than the message
    // returned by Spark 3.3
    #[error("[CAST_INVALID_INPUT] The value '{value}' of the type \"{from_type}\" cannot be cast to \"{to_type}\" \
        because it is malformed. Correct the value as per the syntax, or change its target type. \
        Use `try_cast` to tolerate malformed input and return NULL instead. If necessary \
        set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error.")]
    CastInvalidValue {
        value: String,
        from_type: String,
        to_type: String,
    },

    #[error("[NUMERIC_VALUE_OUT_OF_RANGE] {value} cannot be represented as Decimal({precision}, {scale}). If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error, and return NULL instead.")]
    NumericValueOutOfRange {
        value: String,
        precision: u8,
        scale: i8,
    },

    #[error("[NUMERIC_OUT_OF_SUPPORTED_RANGE] The value {value} cannot be interpreted as a numeric since it has more than 38 digits.")]
    NumericOutOfRange { value: String },

    #[error("[CAST_OVERFLOW] The value {value} of the type \"{from_type}\" cannot be cast to \"{to_type}\" \
        due to an overflow. Use `try_cast` to tolerate overflow and return NULL instead. If necessary \
        set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error.")]
    CastOverFlow {
        value: String,
        from_type: String,
        to_type: String,
    },

    #[error("[CANNOT_PARSE_DECIMAL] Cannot parse decimal.")]
    CannotParseDecimal,

    // ==================== Arithmetic Errors ====================

    #[error("[ARITHMETIC_OVERFLOW] {from_type} overflow. If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error.")]
    ArithmeticOverflow { from_type: String },

    #[error("[DIVIDE_BY_ZERO] Division by zero. Use `try_divide` to tolerate divisor being 0 and return NULL instead. If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error.")]
    DivideByZero,

    #[error("[REMAINDER_BY_ZERO] Division by zero. Use `try_remainder` to tolerate divisor being 0 and return NULL instead. If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error.")]
    RemainderByZero,

    #[error("[INTERVAL_DIVIDED_BY_ZERO] Divide by zero in interval arithmetic.")]
    IntervalDividedByZero,

    #[error("[BINARY_ARITHMETIC_OVERFLOW] {value1} {symbol} {value2} caused overflow. Use `{function_name}` to tolerate overflow and return NULL instead. If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error.")]
    BinaryArithmeticOverflow {
        value1: String,
        symbol: String,
        value2: String,
        function_name: String,
    },

    #[error("[INTERVAL_ARITHMETIC_OVERFLOW] Interval arithmetic overflow. Use `{function_name}` to tolerate overflow and return NULL instead. If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error.")]
    IntervalArithmeticOverflowWithSuggestion {
        function_name: String,
    },

    #[error("[INTERVAL_ARITHMETIC_OVERFLOW] Interval arithmetic overflow. If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error.")]
    IntervalArithmeticOverflowWithoutSuggestion,

    #[error("[DATETIME_OVERFLOW] Datetime arithmetic overflow.")]
    DatetimeOverflow,

    // ==================== Array Index Errors ====================

    #[error("[INVALID_ARRAY_INDEX] The index {index_value} is out of bounds. The array has {array_size} elements. Use the SQL function `get(array, index)` or `try_element_at` instead. If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error.")]
    InvalidArrayIndex {
        index_value: i32,
        array_size: i32,
    },

    #[error("[INVALID_ARRAY_INDEX_IN_ELEMENT_AT] The index {index_value} is out of bounds. The array has {array_size} elements. If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error.")]
    InvalidElementAtIndex {
        index_value: i32,
        array_size: i32,
    },

    #[error("[INVALID_BITMAP_POSITION] The bit position {bit_position} is out of bounds. The bitmap has {bitmap_num_bytes} bytes ({bitmap_num_bits} bits).")]
    InvalidBitmapPosition {
        bit_position: i64,
        bitmap_num_bytes: i64,
        bitmap_num_bits: i64,
    },

    #[error("[INVALID_INDEX_OF_ZERO] The index 0 is invalid. An index shall be either < 0 or > 0 (the first element is at index 1).")]
    InvalidIndexOfZero,

    // ==================== Map/Collection Errors ====================

    #[error("[DUPLICATED_MAP_KEY] Cannot create map with duplicate keys: {key}.")]
    DuplicatedMapKey {
        key: String,
    },

    #[error("[NULL_MAP_KEY] Cannot use null as map key.")]
    NullMapKey,

    #[error("[MAP_KEY_VALUE_DIFF_SIZES] The key array and value array of a map must have the same length.")]
    MapKeyValueDiffSizes,

    #[error("[EXCEED_LIMIT_LENGTH] Cannot create a map with {size} elements which exceeds the limit {max_size}.")]
    ExceedMapSizeLimit {
        size: i32,
        max_size: i32,
    },

    #[error("[COLLECTION_SIZE_LIMIT_EXCEEDED] Cannot create array with {num_elements} elements which exceeds the limit {max_elements}.")]
    CollectionSizeLimitExceeded {
        num_elements: i64,
        max_elements: i64,
    },

    // ==================== Null Validation Errors ====================

    #[error("[NOT_NULL_ASSERT_VIOLATION] The field `{field_name}` cannot be null.")]
    NotNullAssertViolation {
        field_name: String,
    },

    #[error("[VALUE_IS_NULL] The value of field `{field_name}` at row {row_index} is null.")]
    ValueIsNull {
        field_name: String,
        row_index: i32,
    },

    // ==================== DateTime Errors ====================

    #[error("[CANNOT_PARSE_TIMESTAMP] Cannot parse timestamp: {message}. Try using `{suggested_func}` instead.")]
    CannotParseTimestamp {
        message: String,
        suggested_func: String,
    },

    #[error("[INVALID_FRACTION_OF_SECOND] The fraction of second {value} is invalid. Valid values are in the range [0, 60]. If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error.")]
    InvalidFractionOfSecond {
        value: f64,
    },

    // ==================== String/UTF8 Errors ====================

    #[error("[INVALID_UTF8_STRING] Invalid UTF-8 string: {hex_string}.")]
    InvalidUtf8String {
        hex_string: String,
    },

    // ==================== Function Parameter Errors ====================

    #[error("[UNEXPECTED_POSITIVE_VALUE] The {parameter_name} parameter must be less than or equal to 0. The actual value is {actual_value}.")]
    UnexpectedPositiveValue {
        parameter_name: String,
        actual_value: i32,
    },

    #[error("[UNEXPECTED_NEGATIVE_VALUE] The {parameter_name} parameter must be greater than or equal to 0. The actual value is {actual_value}.")]
    UnexpectedNegativeValue {
        parameter_name: String,
        actual_value: i32,
    },

    // ==================== Regex Errors ====================

    #[error("[INVALID_PARAMETER_VALUE] Invalid regex group index {group_index} in function `{function_name}`. Group count is {group_count}.")]
    InvalidRegexGroupIndex {
        function_name: String,
        group_count: i32,
        group_index: i32,
    },

    // ==================== Unsupported Operation Errors ====================

    #[error("[DATATYPE_CANNOT_ORDER] Cannot order by type: {data_type}.")]
    DatatypeCannotOrder {
        data_type: String,
    },

    // ==================== Subquery Errors ====================

    #[error("[SCALAR_SUBQUERY_TOO_MANY_ROWS] Scalar subquery returned more than one row.")]
    ScalarSubqueryTooManyRows,

    // ==================== Generic Errors ====================

    #[error("ArrowError: {0}.")]
    Arrow(#[source] ArrowError),

    #[error("InternalError: {0}.")]
    Internal(String),
}

/// Metadata for creating Spark exceptions with proper error classes and parameters
#[derive(Debug, Clone)]
pub struct SparkExceptionInfo {
    /// The fully qualified Java exception class name
    pub exception_class: String,
    /// The Spark error class (e.g., "DIVIDE_BY_ZERO")
    pub error_class: String,
    /// Message parameters for the error template
    pub message_parameters: Vec<(String, String)>,
}

impl SparkError {
    /// Returns the appropriate Spark exception class for this error
    pub fn exception_class(&self) -> &'static str {
        match self {
            // ArithmeticException
            SparkError::DivideByZero
            | SparkError::RemainderByZero
            | SparkError::IntervalDividedByZero
            | SparkError::NumericValueOutOfRange { .. }
            | SparkError::ArithmeticOverflow { .. }
            | SparkError::BinaryArithmeticOverflow { .. }
            | SparkError::IntervalArithmeticOverflowWithSuggestion { .. }
            | SparkError::IntervalArithmeticOverflowWithoutSuggestion
            | SparkError::DatetimeOverflow => "org/apache/spark/SparkArithmeticException",

            // CastOverflow gets special handling with CastOverflowException
            SparkError::CastOverFlow { .. } => "org/apache/spark/sql/comet/CastOverflowException",

            // ArrayIndexOutOfBoundsException
            SparkError::InvalidArrayIndex { .. }
            | SparkError::InvalidElementAtIndex { .. }
            | SparkError::InvalidBitmapPosition { .. }
            | SparkError::InvalidIndexOfZero => "org/apache/spark/SparkArrayIndexOutOfBoundsException",

            // RuntimeException
            SparkError::CastInvalidValue { .. }
            | SparkError::CannotParseDecimal
            | SparkError::DuplicatedMapKey { .. }
            | SparkError::NullMapKey
            | SparkError::MapKeyValueDiffSizes
            | SparkError::ExceedMapSizeLimit { .. }
            | SparkError::CollectionSizeLimitExceeded { .. }
            | SparkError::NotNullAssertViolation { .. }
            | SparkError::ValueIsNull { .. }
            | SparkError::InvalidUtf8String { .. }
            | SparkError::UnexpectedPositiveValue { .. }
            | SparkError::UnexpectedNegativeValue { .. }
            | SparkError::InvalidRegexGroupIndex { .. }
            | SparkError::ScalarSubqueryTooManyRows => "org/apache/spark/SparkRuntimeException",

            // DateTimeException
            SparkError::CannotParseTimestamp { .. }
            | SparkError::InvalidFractionOfSecond { .. } => "org/apache/spark/SparkDateTimeException",

            // IllegalArgumentException
            SparkError::DatatypeCannotOrder { .. } => "org/apache/spark/SparkIllegalArgumentException",

            // Generic errors
            SparkError::Arrow(_) | SparkError::Internal(_) => "org/apache/spark/SparkException",
        }
    }

    /// Returns the Spark error class code for this error
    pub fn error_class(&self) -> Option<&'static str> {
        match self {
            // Cast errors
            SparkError::CastInvalidValue { .. } => Some("CAST_INVALID_INPUT"),
            SparkError::CastOverFlow { .. } => Some("CAST_OVERFLOW"),
            SparkError::NumericValueOutOfRange { .. } => Some("NUMERIC_VALUE_OUT_OF_RANGE"),
            SparkError::CannotParseDecimal => Some("CANNOT_PARSE_DECIMAL"),

            // Arithmetic errors
            SparkError::DivideByZero => Some("DIVIDE_BY_ZERO"),
            SparkError::RemainderByZero => Some("REMAINDER_BY_ZERO"),
            SparkError::IntervalDividedByZero => Some("INTERVAL_DIVIDED_BY_ZERO"),
            SparkError::ArithmeticOverflow { .. } => Some("ARITHMETIC_OVERFLOW"),
            SparkError::BinaryArithmeticOverflow { .. } => Some("BINARY_ARITHMETIC_OVERFLOW"),
            SparkError::IntervalArithmeticOverflowWithSuggestion { .. } => {
                Some("INTERVAL_ARITHMETIC_OVERFLOW")
            }
            SparkError::IntervalArithmeticOverflowWithoutSuggestion => {
                Some("INTERVAL_ARITHMETIC_OVERFLOW")
            }
            SparkError::DatetimeOverflow => Some("DATETIME_OVERFLOW"),

            // Array index errors
            SparkError::InvalidArrayIndex { .. } => Some("INVALID_ARRAY_INDEX"),
            SparkError::InvalidElementAtIndex { .. } => Some("INVALID_ARRAY_INDEX_IN_ELEMENT_AT"),
            SparkError::InvalidBitmapPosition { .. } => Some("INVALID_BITMAP_POSITION"),
            SparkError::InvalidIndexOfZero => Some("INVALID_INDEX_OF_ZERO"),

            // Map/Collection errors
            SparkError::DuplicatedMapKey { .. } => Some("DUPLICATED_MAP_KEY"),
            SparkError::NullMapKey => Some("NULL_MAP_KEY"),
            SparkError::MapKeyValueDiffSizes => Some("MAP_KEY_VALUE_DIFF_SIZES"),
            SparkError::ExceedMapSizeLimit { .. } => Some("EXCEED_LIMIT_LENGTH"),
            SparkError::CollectionSizeLimitExceeded { .. } => {
                Some("COLLECTION_SIZE_LIMIT_EXCEEDED")
            }

            // Null validation errors
            SparkError::NotNullAssertViolation { .. } => Some("NOT_NULL_ASSERT_VIOLATION"),
            SparkError::ValueIsNull { .. } => Some("VALUE_IS_NULL"),

            // DateTime errors
            SparkError::CannotParseTimestamp { .. } => Some("CANNOT_PARSE_TIMESTAMP"),
            SparkError::InvalidFractionOfSecond { .. } => Some("INVALID_FRACTION_OF_SECOND"),

            // String/UTF8 errors
            SparkError::InvalidUtf8String { .. } => Some("INVALID_UTF8_STRING"),

            // Function parameter errors
            SparkError::UnexpectedPositiveValue { .. } => Some("UNEXPECTED_POSITIVE_VALUE"),
            SparkError::UnexpectedNegativeValue { .. } => Some("UNEXPECTED_NEGATIVE_VALUE"),

            // Regex errors
            SparkError::InvalidRegexGroupIndex { .. } => Some("INVALID_PARAMETER_VALUE"),

            // Unsupported operation errors
            SparkError::DatatypeCannotOrder { .. } => Some("DATATYPE_CANNOT_ORDER"),

            // Subquery errors
            SparkError::ScalarSubqueryTooManyRows => Some("SCALAR_SUBQUERY_TOO_MANY_ROWS"),

            // Generic errors (no error class)
            SparkError::Arrow(_) | SparkError::Internal(_) => None,
        }
    }
}

pub type SparkResult<T> = Result<T, SparkError>;

impl From<ArrowError> for SparkError {
    fn from(value: ArrowError) -> Self {
        SparkError::Arrow(value)
    }
}

impl From<SparkError> for DataFusionError {
    fn from(value: SparkError) -> Self {
        DataFusionError::External(Box::new(value))
    }
}

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

use crate::execution::operators::ExecutionError;
use arrow::{
    array::{cast::AsArray, types::Int32Type, Array, ArrayRef},
    compute::{cast_with_options, take, CastOptions},
    util::display::FormatOptions,
};
use arrow_array::{DictionaryArray, StructArray};
use arrow_schema::DataType;
use datafusion_comet_spark_expr::utils::array_with_timezone;
use datafusion_comet_spark_expr::EvalMode;
use datafusion_common::{Result as DataFusionResult, ScalarValue};
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_expr::ColumnarValue;
use object_store::path::Path;
use object_store::{parse_url, ObjectStore};
use std::collections::HashMap;
use std::{fmt::Debug, hash::Hash, sync::Arc};
use url::Url;

static TIMESTAMP_FORMAT: Option<&str> = Some("%Y-%m-%d %H:%M:%S%.f");

static PARQUET_OPTIONS: CastOptions = CastOptions {
    safe: true,
    format_options: FormatOptions::new()
        .with_timestamp_tz_format(TIMESTAMP_FORMAT)
        .with_timestamp_format(TIMESTAMP_FORMAT),
};

/// Spark cast options
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SparkParquetOptions {
    /// Spark evaluation mode
    pub eval_mode: EvalMode,
    /// When cast from/to timezone related types, we need timezone, which will be resolved with
    /// session local timezone by an analyzer in Spark.
    // TODO we should change timezone to Tz to avoid repeated parsing
    pub timezone: String,
    /// Allow casts that are supported but not guaranteed to be 100% compatible
    pub allow_incompat: bool,
    /// Support casting unsigned ints to signed ints (used by Parquet SchemaAdapter)
    pub allow_cast_unsigned_ints: bool,
    /// We also use the cast logic for adapting Parquet schemas, so this flag is used
    /// for that use case
    pub is_adapting_schema: bool,
    /// Whether to always represent decimals using 128 bits. If false, the native reader may represent decimals using 32 or 64 bits, depending on the precision.
    pub use_decimal_128: bool,
    /// Whether to read dates/timestamps that were written in the legacy hybrid Julian + Gregorian calendar as it is. If false, throw exceptions instead. If the spark type is TimestampNTZ, this should be true.
    pub use_legacy_date_timestamp_or_ntz: bool,
}

impl SparkParquetOptions {
    pub fn new(eval_mode: EvalMode, timezone: &str, allow_incompat: bool) -> Self {
        Self {
            eval_mode,
            timezone: timezone.to_string(),
            allow_incompat,
            allow_cast_unsigned_ints: false,
            is_adapting_schema: false,
            use_decimal_128: false,
            use_legacy_date_timestamp_or_ntz: false,
        }
    }

    pub fn new_without_timezone(eval_mode: EvalMode, allow_incompat: bool) -> Self {
        Self {
            eval_mode,
            timezone: "".to_string(),
            allow_incompat,
            allow_cast_unsigned_ints: false,
            is_adapting_schema: false,
            use_decimal_128: false,
            use_legacy_date_timestamp_or_ntz: false,
        }
    }
}

/// Spark-compatible cast implementation. Defers to DataFusion's cast where that is known
/// to be compatible, and returns an error when a not supported and not DF-compatible cast
/// is requested.
pub fn spark_parquet_convert(
    arg: ColumnarValue,
    data_type: &DataType,
    parquet_options: &SparkParquetOptions,
) -> DataFusionResult<ColumnarValue> {
    match arg {
        ColumnarValue::Array(array) => Ok(ColumnarValue::Array(cast_array(
            array,
            data_type,
            parquet_options,
        )?)),
        ColumnarValue::Scalar(scalar) => {
            // Note that normally CAST(scalar) should be fold in Spark JVM side. However, for
            // some cases e.g., scalar subquery, Spark will not fold it, so we need to handle it
            // here.
            let array = scalar.to_array()?;
            let scalar =
                ScalarValue::try_from_array(&cast_array(array, data_type, parquet_options)?, 0)?;
            Ok(ColumnarValue::Scalar(scalar))
        }
    }
}

fn cast_array(
    array: ArrayRef,
    to_type: &DataType,
    parquet_options: &SparkParquetOptions,
) -> DataFusionResult<ArrayRef> {
    use DataType::*;
    let array = match to_type {
        Timestamp(_, None) => array, // array_with_timezone does not support to_type of NTZ.
        _ => array_with_timezone(array, parquet_options.timezone.clone(), Some(to_type))?,
    };
    let from_type = array.data_type().clone();

    let array = match &from_type {
        Dictionary(key_type, value_type)
            if key_type.as_ref() == &Int32
                && (value_type.as_ref() == &Utf8 || value_type.as_ref() == &LargeUtf8) =>
        {
            let dict_array = array
                .as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()
                .expect("Expected a dictionary array");

            let casted_dictionary = DictionaryArray::<Int32Type>::new(
                dict_array.keys().clone(),
                cast_array(Arc::clone(dict_array.values()), to_type, parquet_options)?,
            );

            let casted_result = match to_type {
                Dictionary(_, _) => Arc::new(casted_dictionary.clone()),
                _ => take(casted_dictionary.values().as_ref(), dict_array.keys(), None)?,
            };
            return Ok(casted_result);
        }
        _ => array,
    };
    let from_type = array.data_type();

    match (from_type, to_type) {
        (Struct(_), Struct(_)) => Ok(cast_struct_to_struct(
            array.as_struct(),
            from_type,
            to_type,
            parquet_options,
        )?),
        _ => Ok(cast_with_options(&array, to_type, &PARQUET_OPTIONS)?),
    }
}

/// Cast between struct types based on logic in
/// `org.apache.spark.sql.catalyst.expressions.Cast#castStruct`.
fn cast_struct_to_struct(
    array: &StructArray,
    from_type: &DataType,
    to_type: &DataType,
    parquet_options: &SparkParquetOptions,
) -> DataFusionResult<ArrayRef> {
    match (from_type, to_type) {
        (DataType::Struct(from_fields), DataType::Struct(to_fields)) => {
            // TODO some of this logic may be specific to converting Parquet to Spark
            let mut field_name_to_index_map = HashMap::new();
            for (i, field) in from_fields.iter().enumerate() {
                field_name_to_index_map.insert(field.name(), i);
            }
            assert_eq!(field_name_to_index_map.len(), from_fields.len());
            let mut cast_fields: Vec<ArrayRef> = Vec::with_capacity(to_fields.len());
            for i in 0..to_fields.len() {
                let from_index = field_name_to_index_map[to_fields[i].name()];
                let cast_field = cast_array(
                    Arc::clone(array.column(from_index)),
                    to_fields[i].data_type(),
                    parquet_options,
                )?;
                cast_fields.push(cast_field);
            }
            Ok(Arc::new(StructArray::new(
                to_fields.clone(),
                cast_fields,
                array.nulls().cloned(),
            )))
        }
        _ => unreachable!(),
    }
}

// Mirrors object_store::parse::parse_url for the hdfs object store
#[cfg(feature = "hdfs")]
fn parse_hdfs_url(url: &Url) -> Result<(Box<dyn ObjectStore>, Path), object_store::Error> {
    match datafusion_comet_objectstore_hdfs::object_store::hdfs::HadoopFileSystem::new(url.as_ref())
    {
        Some(object_store) => {
            let path = object_store.get_path(url.as_str());
            Ok((Box::new(object_store), path))
        }
        _ => {
            return Err(object_store::Error::Generic {
                store: "HadoopFileSystem",
                source: Box::new(datafusion_comet_objectstore_hdfs::object_store::hdfs::HadoopFileSystem::HdfsErr::Generic(
                    "Could not create hdfs object store".to_string(),
                )),
            });
        }
    }
}

#[cfg(not(feature = "hdfs"))]
fn parse_hdfs_url(_url: &Url) -> Result<(Box<dyn ObjectStore>, Path), object_store::Error> {
    Err(object_store::Error::Generic {
        store: "HadoopFileSystem",
        source: "Hdfs support is not enabled in this build".into(),
    })
}

/// Parses the url, registers the object store, and returns a tuple of the object store url and object store path
pub(crate) fn prepare_object_store(
    runtime_env: Arc<RuntimeEnv>,
    url: String,
) -> Result<(ObjectStoreUrl, Path), ExecutionError> {
    let mut url = Url::parse(url.as_str()).unwrap();
    let mut scheme = url.scheme();
    if scheme == "s3a" {
        scheme = "s3";
        url.set_scheme("s3")
            .expect("Could not convert scheme from s3a to s3");
    }
    let url_key = format!(
        "{}://{}",
        scheme,
        &url[url::Position::BeforeHost..url::Position::AfterPort],
    );

    let (object_store, object_store_path): (Box<dyn ObjectStore>, Path) = if scheme == "hdfs" {
        match parse_hdfs_url(&url) {
            Ok(r) => r,
            Err(e) => return Err(ExecutionError::GeneralError(e.to_string())),
        }
    } else {
        match parse_url(&url) {
            Ok(r) => r,
            Err(e) => return Err(ExecutionError::GeneralError(e.to_string())),
        }
    };
    let object_store_url = ObjectStoreUrl::parse(url_key.clone())?;
    // runtime_env.register_object_store(object_store_url.as_ref(), Arc::from(object_store));
    runtime_env.register_object_store(&url, Arc::from(object_store));
    Ok((object_store_url, object_store_path))
}

#[cfg(test)]
mod tests {
    use crate::execution::operators::ExecutionError;
    use crate::parquet::parquet_support::prepare_object_store;
    use datafusion_execution::object_store::ObjectStoreUrl;
    use datafusion_execution::runtime_env::RuntimeEnv;
    use object_store::path::Path;
    use std::sync::Arc;
    use url::Url;

    #[test]
    #[cfg(not(feature = "hdfs"))]
    fn test_prepare_object_store() {
        let local_file_system_url = "file:///comet/spark-warehouse/part-00000.snappy.parquet";
        let s3_url = "s3a://test_bucket/comet/spark-warehouse/part-00000.snappy.parquet";
        let hdfs_url = "hdfs://localhost:8020/comet/spark-warehouse/part-00000.snappy.parquet";

        let all_urls = [local_file_system_url, s3_url, hdfs_url];
        let expected: Vec<Result<(ObjectStoreUrl, Path), ExecutionError>> = vec![
            Ok((
                ObjectStoreUrl::parse("file://").unwrap(),
                Path::from("/comet/spark-warehouse/part-00000.snappy.parquet"),
            )),
            Ok((
                ObjectStoreUrl::parse("s3://test_bucket").unwrap(),
                Path::from("/comet/spark-warehouse/part-00000.snappy.parquet"),
            )),
            Err(ExecutionError::GeneralError(
                "Generic HadoopFileSystem error: Hdfs support is not enabled in this build"
                    .parse()
                    .unwrap(),
            )),
        ];

        for (i, url_str) in all_urls.iter().enumerate() {
            let url = &Url::parse(url_str).unwrap();
            let res = prepare_object_store(Arc::new(RuntimeEnv::default()), url.to_string());

            let expected = expected.get(i).unwrap();
            match expected {
                Ok((o, p)) => {
                    let (r_o, r_p) = res.unwrap();
                    assert_eq!(r_o, *o);
                    assert_eq!(r_p, *p);
                }
                Err(e) => {
                    assert!(res.is_err());
                    let Err(res_e) = res else {
                        panic!("test failed")
                    };
                    assert_eq!(e.to_string(), res_e.to_string())
                }
            }
        }
    }
}

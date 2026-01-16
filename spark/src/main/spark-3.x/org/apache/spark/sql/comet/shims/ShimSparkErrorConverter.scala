/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.spark.sql.comet.shims

/**
 * Spark 3.x stub implementation for converting error types to Spark exceptions.
 *
 * This stub always returns None, which causes SparkErrorConverter to fall back to the generic
 * Phase 1 behavior of returning SparkException with error class. This preserves backward
 * compatibility for Spark 3.x while allowing Spark 4.0+ to use proper typed exceptions.
 */
trait ShimSparkErrorConverter {

  /**
   * Convert error type string and parameters to appropriate Spark exception. This Spark 3.x stub
   * always returns None to preserve Phase 1 generic behavior.
   *
   * @param errorType
   *   The error type from JSON (e.g., "DivideByZero")
   * @param errorClass
   *   The Spark error class (e.g., "DIVIDE_BY_ZERO")
   * @param params
   *   Error parameters from JSON
   * @return
   *   Always None (triggers fallback to generic SparkException)
   */
  def convertErrorType(
      errorType: String,
      errorClass: String,
      params: Map[String, Any]): Option[Throwable] = {
    // Spark 3.x: Return None to use generic SparkException (Phase 1 behavior)
    None
  }
}

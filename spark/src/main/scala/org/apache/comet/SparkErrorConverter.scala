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

package org.apache.comet

import org.json4s._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.SparkException

import org.apache.comet.exceptions.CometQueryExecutionException

/**
 * Converts CometQueryExecutionException (with JSON payload) to appropriate Spark
 * QueryExecutionErrors.* exceptions.
 *
 * This converter parses the JSON-encoded error information from native execution and dispatches
 * to the corresponding QueryExecutionErrors factory method.
 *
 * NOTE: Phase 1 (MVP) implementation - returns generic SparkException with error class. Phase 2
 * will add full QueryExecutionErrors integration with version-specific shims.
 */
object SparkErrorConverter {

  implicit val formats: DefaultFormats.type = DefaultFormats

  case class ErrorJson(
      errorType: String,
      errorClass: Option[String],
      params: Option[Map[String, Any]])

  /**
   * Parse JSON from exception and convert to appropriate Spark exception.
   *
   * Phase 1 (MVP): Returns a SparkException with the error class and formatted message. Phase 2:
   * Will call specific QueryExecutionErrors.* methods per Spark version.
   *
   * @param e
   *   the CometQueryExecutionException with JSON message
   * @return
   *   the corresponding Spark exception, or the original exception if parsing fails
   */
  def convertToSparkException(e: CometQueryExecutionException): Throwable = {
    try {
      if (!e.isJsonMessage()) {
        // Not JSON, return original exception
        return e
      }

      val json = parse(e.getMessage)
      val errorJson = json.extract[ErrorJson]
      val params = errorJson.params.getOrElse(Map.empty)
      val errorClass = errorJson.errorClass.getOrElse("_LEGACY_ERROR_TEMP_COMET")

      // Phase 1 (MVP): Create generic SparkException with error class and message
      // This ensures errors are properly categorized and can be caught by error class
      val message = formatErrorMessage(errorJson.errorType, errorClass, params)

      new SparkException(
        errorClass = errorClass,
        messageParameters = paramsToStringMap(params),
        cause = null)
    } catch {
      case _: Exception =>
        // JSON parsing failed, return original exception
        e
    }
  }

  /**
   * Format error message from error type and parameters.
   */
  private def formatErrorMessage(
      errorType: String,
      errorClass: String,
      params: Map[String, Any]): String = {
    // Create a human-readable message from the error type and parameters
    val paramStr = if (params.nonEmpty) {
      params.map { case (k, v) => s"$k=$v" }.mkString(", ")
    } else {
      ""
    }

    s"[$errorClass] $errorType${if (paramStr.nonEmpty) s": $paramStr" else ""}"
  }

  /**
   * Convert parameter map to string-keyed map for SparkException.
   */
  private def paramsToStringMap(params: Map[String, Any]): Map[String, String] = {
    params.map { case (k, v) => (k, v.toString) }
  }
}

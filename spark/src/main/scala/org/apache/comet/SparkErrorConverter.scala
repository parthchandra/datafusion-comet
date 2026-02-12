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

import org.apache.spark.{QueryContext, SparkException}
import org.apache.spark.sql.catalyst.trees.SQLQueryContext
import org.apache.spark.sql.comet.shims.ShimSparkErrorConverter

import org.apache.comet.exceptions.CometQueryExecutionException

/**
 * Converts CometQueryExecutionException (with JSON payload) to appropriate Spark
 * QueryExecutionErrors.* exceptions using version-specific shims.
 *
 * This converter parses the JSON-encoded error information from native execution and delegates to
 * the version-specific ShimSparkErrorConverter trait for conversion to proper Spark exception
 * types.
 *
 * For Spark 4.0+, this returns properly typed exceptions (SparkArithmeticException,
 * SparkArrayIndexOutOfBoundsException, etc.). For Spark 3.x, falls back to generic
 * SparkException.
 */
object SparkErrorConverter extends ShimSparkErrorConverter {

  implicit val formats: DefaultFormats.type = DefaultFormats

  case class QueryContextJson(
      sqlText: String,
      startIndex: Int,
      stopIndex: Int,
      objectType: Option[String],
      objectName: Option[String],
      line: Int,
      startPosition: Int)

  case class ErrorJson(
      errorType: String,
      errorClass: Option[String],
      params: Option[Map[String, Any]],
      context: Option[QueryContextJson],
      summary: Option[String])

  /**
   * Parse JSON from exception and convert to appropriate Spark exception.
   *
   * Delegates to version-specific ShimSparkErrorConverter.convertErrorType() to call the proper
   * QueryExecutionErrors.* method for the Spark version.
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

      // Build Spark SQLQueryContext if context is present
      val sparkContext: Array[QueryContext] = errorJson.context match {
        case Some(ctx) =>
          Array(
            new SQLQueryContext(
              sqlText = Some(ctx.sqlText),
              line = Some(ctx.line),
              startPosition = Some(ctx.startPosition),
              originStartIndex = Some(ctx.startIndex),
              originStopIndex = Some(ctx.stopIndex),
              originObjectType = ctx.objectType,
              originObjectName = ctx.objectName))
        case None => null // No context available
      }

      val summary: String = errorJson.summary.orNull

      // Delegate to version-specific shim
      convertErrorType(errorJson.errorType, errorClass, params, sparkContext, summary) match {
        case Some(exception) =>
          // Shim successfully converted - return the proper typed exception
          exception

        case None =>
          // Unknown error type - fallback to generic SparkException (Phase 1 behavior)
          new SparkException(
            errorClass = errorClass,
            messageParameters = paramsToStringMap(params),
            cause = null)
      }
    } catch {
      case _: Exception =>
        // JSON parsing failed, return original exception
        e
    }
  }

  /**
   * Convert parameter map to string-keyed map for SparkException.
   */
  private def paramsToStringMap(params: Map[String, Any]): Map[String, String] = {
    params.map { case (k, v) => (k, v.toString) }
  }
}

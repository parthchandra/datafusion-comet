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

package org.apache.comet.serde

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.catalyst.expressions.objects.StaticInvoke
import org.apache.spark.sql.catalyst.optimizer.{BuildRight, NormalizeNaNAndZero}
import org.apache.spark.sql.catalyst.plans._
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.catalyst.util.CharVarcharCodegenUtils
import org.apache.spark.sql.comet.{CometBroadcastExchangeExec, CometHashAggregateExec, CometPlan, CometSinkPlaceHolder, DecimalPrecision}
import org.apache.spark.sql.comet.execution.shuffle.CometShuffleExchangeExec
import org.apache.spark.sql.execution
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.adaptive.{BroadcastQueryStageExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeExec, ReusedExchangeExec, ShuffleExchangeExec}
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, HashJoin, ShuffledHashJoinExec, SortMergeJoinExec}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

import org.apache.comet.{CometExplainInfo, CometExplainSubtreeIsNotNative}
import org.apache.comet.CometSparkSessionExtensions.{isCometOperatorEnabled, isCometScan, isSpark32, isSpark34Plus}
import org.apache.comet.serde.ExprOuterClass.{AggExpr, DataType => ProtoDataType, Expr, ScalarFunc}
import org.apache.comet.serde.ExprOuterClass.DataType.{DataTypeInfo, DecimalInfo, ListInfo, MapInfo, StructInfo}
import org.apache.comet.serde.OperatorOuterClass.{AggregateMode => CometAggregateMode, JoinType, Operator}
import org.apache.comet.shims.ShimQueryPlanSerde

/**
 * An utility object for query plan and expression serialization.
 */
object QueryPlanSerde extends Logging with ShimQueryPlanSerde {
  def emitWarning(reason: String): Unit = {
    logWarning(s"Comet native execution is disabled due to: $reason")
  }

  def unsupported[T: ClassTag, R: ClassTag](
      op: String,
      info: T): (Option[R], CometExplainInfo) = {
    info match {
      case s: Seq[CometExplainInfo] if s.nonEmpty =>
        (None, CometExplainInfo(op, s))
      case _: CometExplainSubtreeIsNotNative =>
        (None, CometExplainInfo.subTreeIsNotNative)
      case i: CometExplainInfo =>
        (None, CometExplainInfo(op, i))
      case s: String =>
        (None, CometExplainInfo(s"$op is not supported ($s)"))
      case _ =>
        (None, CometExplainInfo(s"$op is not supported"))
    }
  }

  def supportedDataType(dt: DataType): Boolean = dt match {
    case _: ByteType | _: ShortType | _: IntegerType | _: LongType | _: FloatType |
        _: DoubleType | _: StringType | _: BinaryType | _: TimestampType | _: DecimalType |
        _: DateType | _: BooleanType | _: NullType =>
      true
    // `TimestampNTZType` is private in Spark 3.2.
    case dt if dt.typeName == "timestamp_ntz" => true
    case dt =>
      emitWarning(s"unsupported Spark data type: $dt")
      false
  }

  /**
   * Serializes Spark datatype to protobuf. Note that, a datatype can be serialized by this method
   * doesn't mean it is supported by Comet native execution, i.e., `supportedDataType` may return
   * false for it.
   */
  def serializeDataType(dt: DataType): Option[ExprOuterClass.DataType] = {
    val typeId = dt match {
      case _: BooleanType => 0
      case _: ByteType => 1
      case _: ShortType => 2
      case _: IntegerType => 3
      case _: LongType => 4
      case _: FloatType => 5
      case _: DoubleType => 6
      case _: StringType => 7
      case _: BinaryType => 8
      case _: TimestampType => 9
      case _: DecimalType => 10
      case dt if dt.typeName == "timestamp_ntz" => 11
      case _: DateType => 12
      case _: NullType => 13
      case _: ArrayType => 14
      case _: MapType => 15
      case _: StructType => 16
      case dt =>
        emitWarning(s"Cannot serialize Spark data type: $dt")
        return None
    }

    val builder = ProtoDataType.newBuilder()
    builder.setTypeIdValue(typeId)

    // Decimal
    val dataType = dt match {
      case t: DecimalType =>
        val info = DataTypeInfo.newBuilder()
        val decimal = DecimalInfo.newBuilder()
        decimal.setPrecision(t.precision)
        decimal.setScale(t.scale)
        info.setDecimal(decimal)
        builder.setTypeInfo(info.build()).build()

      case a: ArrayType =>
        val elementType = serializeDataType(a.elementType)

        if (elementType.isEmpty) {
          return None
        }

        val info = DataTypeInfo.newBuilder()
        val list = ListInfo.newBuilder()
        list.setElementType(elementType.get)
        list.setContainsNull(a.containsNull)

        info.setList(list)
        builder.setTypeInfo(info.build()).build()

      case m: MapType =>
        val keyType = serializeDataType(m.keyType)
        if (keyType.isEmpty) {
          return None
        }

        val valueType = serializeDataType(m.valueType)
        if (valueType.isEmpty) {
          return None
        }

        val info = DataTypeInfo.newBuilder()
        val map = MapInfo.newBuilder()
        map.setKeyType(keyType.get)
        map.setValueType(valueType.get)
        map.setValueContainsNull(m.valueContainsNull)

        info.setMap(map)
        builder.setTypeInfo(info.build()).build()

      case s: StructType =>
        val info = DataTypeInfo.newBuilder()
        val struct = StructInfo.newBuilder()

        val fieldNames = s.fields.map(_.name).toIterable.asJava
        val fieldDatatypes = s.fields.map(f => serializeDataType(f.dataType)).toSeq
        val fieldNullable = s.fields.map(f => Boolean.box(f.nullable)).toIterable.asJava

        if (fieldDatatypes.exists(_.isEmpty)) {
          return None
        }

        struct.addAllFieldNames(fieldNames)
        struct.addAllFieldDatatypes(fieldDatatypes.map(_.get).asJava)
        struct.addAllFieldNullable(fieldNullable)

        info.setStruct(struct)
        builder.setTypeInfo(info.build()).build()
      case _ => builder.build()
    }

    Some(dataType)
  }

  private def sumDataTypeSupported(dt: DataType): Boolean = {
    dt match {
      case _: NumericType => true
      case _ => false
    }
  }

  private def avgDataTypeSupported(dt: DataType): Boolean = {
    dt match {
      case _: NumericType => true
      // TODO: implement support for interval types
      case _ => false
    }
  }

  private def minMaxDataTypeSupported(dt: DataType): Boolean = {
    dt match {
      case _: NumericType | DateType | TimestampType | BooleanType => true
      case _ => false
    }
  }

  private def bitwiseAggTypeSupported(dt: DataType): Boolean = {
    dt match {
      case _: IntegerType | LongType | ShortType | ByteType => true
      case _ => false
    }
  }

  def aggExprToProto(
      aggExpr: AggregateExpression,
      inputs: Seq[Attribute],
      binding: Boolean): (Option[AggExpr], CometExplainInfo) = {
    aggExpr.aggregateFunction match {
      case s @ Sum(child, _) if sumDataTypeSupported(s.dataType) =>
        val (childExpr, info) = exprToProto(child, inputs, binding)
        val dataType = serializeDataType(s.dataType)

        if (childExpr.isDefined && dataType.isDefined) {
          val sumBuilder = ExprOuterClass.Sum.newBuilder()
          sumBuilder.setChild(childExpr.get)
          sumBuilder.setDatatype(dataType.get)
          sumBuilder.setFailOnError(getFailOnError(s))

          (
            Some(
              ExprOuterClass.AggExpr
                .newBuilder()
                .setSum(sumBuilder)
                .build()),
            CometExplainInfo.none)
        } else if (dataType.isEmpty) {
          unsupported("SUM", CometExplainInfo(s"datatype ${s.dataType} is not supported"))
        } else {
          unsupported("SUM", info)
        }
      case s @ Average(child, _) if avgDataTypeSupported(s.dataType) =>
        val (childExpr, info) = exprToProto(child, inputs, binding)
        val dataType = serializeDataType(s.dataType)

        val sumDataType = if (child.dataType.isInstanceOf[DecimalType]) {

          // This is input precision + 10 to be consistent with Spark
          val precision = Math.min(
            DecimalType.MAX_PRECISION,
            child.dataType.asInstanceOf[DecimalType].precision + 10)
          val newType =
            DecimalType.apply(precision, child.dataType.asInstanceOf[DecimalType].scale)
          serializeDataType(newType)
        } else {
          serializeDataType(child.dataType)
        }

        if (childExpr.isDefined && dataType.isDefined) {
          val builder = ExprOuterClass.Avg.newBuilder()
          builder.setChild(childExpr.get)
          builder.setDatatype(dataType.get)
          builder.setFailOnError(getFailOnError(s))
          builder.setSumDatatype(sumDataType.get)

          (
            Some(
              ExprOuterClass.AggExpr
                .newBuilder()
                .setAvg(builder)
                .build()),
            CometExplainInfo.none)
        } else if (dataType.isEmpty) {
          unsupported("AVERAGE", CometExplainInfo(s"datatype ${s.dataType} is not supported"))
        } else {
          unsupported("AVERAGE", info)
        }
      case Count(children) =>
        val (exprChildren, exprInfo) = children.map(exprToProto(_, inputs, binding)).unzip

        if (exprChildren.forall(_.isDefined)) {
          val countBuilder = ExprOuterClass.Count.newBuilder()
          countBuilder.addAllChildren(exprChildren.map(_.get).asJava)

          (
            Some(
              ExprOuterClass.AggExpr
                .newBuilder()
                .setCount(countBuilder)
                .build()),
            CometExplainInfo.none)
        } else {
          unsupported("COUNT", exprInfo)
        }
      case min @ Min(child) if minMaxDataTypeSupported(min.dataType) =>
        val (childExpr, info) = exprToProto(child, inputs, binding)
        val dataType = serializeDataType(min.dataType)

        if (childExpr.isDefined && dataType.isDefined) {
          val minBuilder = ExprOuterClass.Min.newBuilder()
          minBuilder.setChild(childExpr.get)
          minBuilder.setDatatype(dataType.get)

          (
            Some(
              ExprOuterClass.AggExpr
                .newBuilder()
                .setMin(minBuilder)
                .build()),
            CometExplainInfo.none)
        } else if (dataType.isEmpty) {
          unsupported("MIN", CometExplainInfo(s"datatype ${min.dataType} is not supported"))
        } else {
          unsupported("MIN", info)
        }
      case max @ Max(child) if minMaxDataTypeSupported(max.dataType) =>
        val (childExpr, info) = exprToProto(child, inputs, binding)
        val dataType = serializeDataType(max.dataType)

        if (childExpr.isDefined && dataType.isDefined) {
          val maxBuilder = ExprOuterClass.Max.newBuilder()
          maxBuilder.setChild(childExpr.get)
          maxBuilder.setDatatype(dataType.get)

          (
            Some(
              ExprOuterClass.AggExpr
                .newBuilder()
                .setMax(maxBuilder)
                .build()),
            CometExplainInfo.none)
        } else if (dataType.isEmpty) {
          unsupported("MAX", CometExplainInfo(s"datatype ${max.dataType} is not supported"))
        } else {
          unsupported("MAX", info)
        }
      case first @ First(child, ignoreNulls)
          if !ignoreNulls => // DataFusion doesn't support ignoreNulls true
        val (childExpr, info) = exprToProto(child, inputs, binding)
        val dataType = serializeDataType(first.dataType)

        if (childExpr.isDefined && dataType.isDefined) {
          val firstBuilder = ExprOuterClass.First.newBuilder()
          firstBuilder.setChild(childExpr.get)
          firstBuilder.setDatatype(dataType.get)

          (
            Some(
              ExprOuterClass.AggExpr
                .newBuilder()
                .setFirst(firstBuilder)
                .build()),
            CometExplainInfo.none)
        } else if (dataType.isEmpty) {
          unsupported("FIRST", CometExplainInfo(s"datatype ${first.dataType} is not supported"))
        } else {
          unsupported("FIRST", info)
        }
      case last @ Last(child, ignoreNulls)
          if !ignoreNulls => // DataFusion doesn't support ignoreNulls true
        val (childExpr, info) = exprToProto(child, inputs, binding)
        val dataType = serializeDataType(last.dataType)

        if (childExpr.isDefined && dataType.isDefined) {
          val lastBuilder = ExprOuterClass.Last.newBuilder()
          lastBuilder.setChild(childExpr.get)
          lastBuilder.setDatatype(dataType.get)

          (
            Some(
              ExprOuterClass.AggExpr
                .newBuilder()
                .setLast(lastBuilder)
                .build()),
            CometExplainInfo.none)
        } else if (dataType.isEmpty) {
          unsupported("LAST", CometExplainInfo(s"datatype ${last.dataType} is not supported"))
        } else {
          unsupported("LAST", info)
        }
      case bitAnd @ BitAndAgg(child) if bitwiseAggTypeSupported(bitAnd.dataType) =>
        val (childExpr, info) = exprToProto(child, inputs, binding)
        val dataType = serializeDataType(bitAnd.dataType)

        if (childExpr.isDefined && dataType.isDefined) {
          val bitAndBuilder = ExprOuterClass.BitAndAgg.newBuilder()
          bitAndBuilder.setChild(childExpr.get)
          bitAndBuilder.setDatatype(dataType.get)

          (
            Some(
              ExprOuterClass.AggExpr
                .newBuilder()
                .setBitAndAgg(bitAndBuilder)
                .build()),
            CometExplainInfo.none)
        } else if (dataType.isEmpty) {
          unsupported("BITAND", CometExplainInfo(s"datatype ${bitAnd.dataType} is not supported"))
        } else {
          unsupported("BITAND", info)
        }
      case bitOr @ BitOrAgg(child) if bitwiseAggTypeSupported(bitOr.dataType) =>
        val (childExpr, info) = exprToProto(child, inputs, binding)
        val dataType = serializeDataType(bitOr.dataType)

        if (childExpr.isDefined && dataType.isDefined) {
          val bitOrBuilder = ExprOuterClass.BitOrAgg.newBuilder()
          bitOrBuilder.setChild(childExpr.get)
          bitOrBuilder.setDatatype(dataType.get)

          (
            Some(
              ExprOuterClass.AggExpr
                .newBuilder()
                .setBitOrAgg(bitOrBuilder)
                .build()),
            CometExplainInfo.none)
        } else if (dataType.isEmpty) {
          unsupported("BITOR", CometExplainInfo(s"datatype ${bitOr.dataType} is not supported"))
        } else {
          unsupported("BITOR", info)
        }
      case bitXor @ BitXorAgg(child) if bitwiseAggTypeSupported(bitXor.dataType) =>
        val (childExpr, info) = exprToProto(child, inputs, binding)
        val dataType = serializeDataType(bitXor.dataType)

        if (childExpr.isDefined && dataType.isDefined) {
          val bitXorBuilder = ExprOuterClass.BitXorAgg.newBuilder()
          bitXorBuilder.setChild(childExpr.get)
          bitXorBuilder.setDatatype(dataType.get)

          (
            Some(
              ExprOuterClass.AggExpr
                .newBuilder()
                .setBitXorAgg(bitXorBuilder)
                .build()),
            CometExplainInfo.none)
        } else if (dataType.isEmpty) {
          unsupported("BITXOR", CometExplainInfo(s"datatype ${bitXor.dataType} is not supported"))
        } else {
          unsupported("BITXOR", info)
        }

      case fn =>
        emitWarning(s"unsupported Spark aggregate function: $fn")
        unsupported(fn.prettyName, CometExplainInfo.none)
    }
  }

  /**
   * Convert a Spark expression to protobuf.
   *
   * @param expr
   *   The input expression
   * @param inputs
   *   The input attributes
   * @param binding
   *   Whether to bind the expression to the input attributes
   * @return
   *   The protobuf representation of the expression, or None if the expression is not supported
   */
  def exprToProto(
      expr: Expression,
      input: Seq[Attribute],
      binding: Boolean = true): (Option[Expr], CometExplainInfo) = {
    def castToProto(
        timeZoneId: Option[String],
        dt: DataType,
        childExpr: Option[Expr]): (Option[Expr], CometExplainInfo) = {
      val dataType = serializeDataType(dt)

      if (childExpr.isDefined && dataType.isDefined) {
        val castBuilder = ExprOuterClass.Cast.newBuilder()
        castBuilder.setChild(childExpr.get)
        castBuilder.setDatatype(dataType.get)

        val timeZone = timeZoneId.getOrElse("UTC")
        castBuilder.setTimezone(timeZone)

        (
          Some(
            ExprOuterClass.Expr
              .newBuilder()
              .setCast(castBuilder)
              .build()),
          CometExplainInfo.none)
      } else {
        if (!dataType.isDefined) {
          unsupported("CAST", CometExplainInfo(s"Unsupported datatype ${dt}"))
        } else {
          unsupported("CAST", CometExplainInfo(s"Unsupported expression ${childExpr}"))
        }
      }
    }

    def exprToProtoInternal(
        expr: Expression,
        inputs: Seq[Attribute]): (Option[Expr], CometExplainInfo) = {
      SQLConf.get
      expr match {
        case a @ Alias(_, _) =>
          exprToProtoInternal(a.child, inputs)

        case cast @ Cast(_: Literal, dataType, _, _) =>
          // This can happen after promoting decimal precisions
          val value = cast.eval()
          exprToProtoInternal(Literal(value, dataType), inputs)

        case Cast(child, dt, timeZoneId, _) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          if (childExpr.isDefined) {
            castToProto(timeZoneId, dt, childExpr)
          } else {
            unsupported("CAST", info)
          }

        case add @ Add(left, right, _) if supportedDataType(left.dataType) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val addBuilder = ExprOuterClass.Add.newBuilder()
            addBuilder.setLeft(leftExpr.get)
            addBuilder.setRight(rightExpr.get)
            addBuilder.setFailOnError(getFailOnError(add))
            serializeDataType(add.dataType).foreach { t =>
              addBuilder.setReturnType(t)
            }

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setAdd(addBuilder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("ADD", Seq(leftInfo, rightInfo))
          }

        case sub @ Subtract(left, right, _) if supportedDataType(left.dataType) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.Subtract.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)
            builder.setFailOnError(getFailOnError(sub))
            serializeDataType(sub.dataType).foreach { t =>
              builder.setReturnType(t)
            }

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setSubtract(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("SUBTRACT", Seq(leftInfo, rightInfo))
          }

        case mul @ Multiply(left, right, _)
            if supportedDataType(left.dataType) && !decimalBeforeSpark34(left.dataType) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.Multiply.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)
            builder.setFailOnError(getFailOnError(mul))
            serializeDataType(mul.dataType).foreach { t =>
              builder.setReturnType(t)
            }

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setMultiply(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("MULTIPLY", Seq(leftInfo, rightInfo))
          }

        case div @ Divide(left, right, _)
            if supportedDataType(left.dataType) && !decimalBeforeSpark34(left.dataType) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          // Datafusion now throws an exception for dividing by zero
          // See https://github.com/apache/arrow-datafusion/pull/6792
          // For now, use NullIf to swap zeros with nulls.
          val (rightExpr, rightInfo) =
            exprToProtoInternal(nullIfWhenPrimitive(right), inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.Divide.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)
            builder.setFailOnError(getFailOnError(div))
            serializeDataType(div.dataType).foreach { t =>
              builder.setReturnType(t)
            }

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setDivide(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("DIVIDE", Seq(leftInfo, rightInfo))
          }

        case rem @ Remainder(left, right, _)
            if supportedDataType(left.dataType) && !decimalBeforeSpark34(left.dataType) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) =
            exprToProtoInternal(nullIfWhenPrimitive(right), inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.Remainder.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)
            builder.setFailOnError(getFailOnError(rem))
            serializeDataType(rem.dataType).foreach { t =>
              builder.setReturnType(t)
            }

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setRemainder(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("REMAINDER", Seq(leftInfo, rightInfo))
          }

        case EqualTo(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.Equal.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setEq(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("EQUALTO", Seq(leftInfo, rightInfo))
          }

        case Not(EqualTo(left, right)) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.NotEqual.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setNeq(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("NOTEQUALTO", Seq(leftInfo, rightInfo))
          }

        case EqualNullSafe(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.EqualNullSafe.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setEqNullSafe(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("EQUALNULLSAFE", Seq(leftInfo, rightInfo))
          }

        case Not(EqualNullSafe(left, right)) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.NotEqualNullSafe.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setNeqNullSafe(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("NOTEQUALNULLSAFE", Seq(leftInfo, rightInfo))
          }

        case GreaterThan(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.GreaterThan.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setGt(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("GREATERTHAN", Seq(leftInfo, rightInfo))
          }

        case GreaterThanOrEqual(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.GreaterThanEqual.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setGtEq(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("GREATERTHANOREQUAL", Seq(leftInfo, rightInfo))
          }

        case LessThan(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.LessThan.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setLt(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("LESSTHAN", Seq(leftInfo, rightInfo))
          }

        case LessThanOrEqual(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.LessThanEqual.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setLtEq(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("LESSTHANOREQUAL", Seq(leftInfo, rightInfo))
          }

        case Literal(value, dataType) if supportedDataType(dataType) =>
          val exprBuilder = ExprOuterClass.Literal.newBuilder()

          if (value == null) {
            exprBuilder.setIsNull(true)
          } else {
            exprBuilder.setIsNull(false)
            dataType match {
              case _: BooleanType => exprBuilder.setBoolVal(value.asInstanceOf[Boolean])
              case _: ByteType => exprBuilder.setByteVal(value.asInstanceOf[Byte])
              case _: ShortType => exprBuilder.setShortVal(value.asInstanceOf[Short])
              case _: IntegerType => exprBuilder.setIntVal(value.asInstanceOf[Int])
              case _: LongType => exprBuilder.setLongVal(value.asInstanceOf[Long])
              case _: FloatType => exprBuilder.setFloatVal(value.asInstanceOf[Float])
              case _: DoubleType => exprBuilder.setDoubleVal(value.asInstanceOf[Double])
              case _: StringType =>
                exprBuilder.setStringVal(value.asInstanceOf[UTF8String].toString)
              case _: TimestampType => exprBuilder.setLongVal(value.asInstanceOf[Long])
              case _: DecimalType =>
                // Pass decimal literal as bytes.
                val unscaled = value.asInstanceOf[Decimal].toBigDecimal.underlying.unscaledValue
                exprBuilder.setDecimalVal(
                  com.google.protobuf.ByteString.copyFrom(unscaled.toByteArray))
              case _: BinaryType =>
                val byteStr =
                  com.google.protobuf.ByteString.copyFrom(value.asInstanceOf[Array[Byte]])
                exprBuilder.setBytesVal(byteStr)
              case _: DateType => exprBuilder.setIntVal(value.asInstanceOf[Int])
              case dt =>
                logWarning(s"Unexpected date type '$dt' for literal value '$value'")
            }
          }

          val dt = serializeDataType(dataType)

          if (dt.isDefined) {
            exprBuilder.setDatatype(dt.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setLiteral(exprBuilder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("LITERAL", CometExplainInfo(s"Unsupported datatype $dataType"))
          }

        case Substring(str, Literal(pos, _), Literal(len, _)) =>
          val (strExpr, info) = exprToProtoInternal(str, inputs)

          if (strExpr.isDefined) {
            val builder = ExprOuterClass.Substring.newBuilder()
            builder.setChild(strExpr.get)
            builder.setStart(pos.asInstanceOf[Int])
            builder.setLen(len.asInstanceOf[Int])

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setSubstring(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("SUBSTRING", info)
          }

        case Like(left, right, _) =>
          // TODO escapeChar
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.Like.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setLike(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("LIKE", Seq(leftInfo, rightInfo))
          }

        // TODO waiting for arrow-rs update
//      case RLike(left, right) =>
//        val leftExpr = exprToProtoInternal(left, inputs)
//        val rightExpr = exprToProtoInternal(right, inputs)
//
//        if (leftExpr.isDefined && rightExpr.isDefined) {
//          val builder = ExprOuterClass.RLike.newBuilder()
//          builder.setLeft(leftExpr.get)
//          builder.setRight(rightExpr.get)
//
//          Some(
//            ExprOuterClass.Expr
//              .newBuilder()
//              .setRlike(builder)
//              .build())
//        } else {
//          None
//        }

        case StartsWith(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.StartsWith.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setStartsWith(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("STARTSWITH", Seq(leftInfo, rightInfo))
          }

        case EndsWith(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.EndsWith.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setEndsWith(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("ENDWITH", Seq(leftInfo, rightInfo))
          }

        case Contains(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.Contains.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setContains(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("CONTAINS", Seq(leftInfo, rightInfo))
          }

        case StringSpace(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)

          if (childExpr.isDefined) {
            val builder = ExprOuterClass.StringSpace.newBuilder()
            builder.setChild(childExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setStringSpace(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("STRINGSPACE", info)
          }

        case Hour(child, timeZoneId) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)

          if (childExpr.isDefined) {
            val builder = ExprOuterClass.Hour.newBuilder()
            builder.setChild(childExpr.get)

            val timeZone = timeZoneId.getOrElse("UTC")
            builder.setTimezone(timeZone)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setHour(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("HOUR", info)
          }

        case Minute(child, timeZoneId) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)

          if (childExpr.isDefined) {
            val builder = ExprOuterClass.Minute.newBuilder()
            builder.setChild(childExpr.get)

            val timeZone = timeZoneId.getOrElse("UTC")
            builder.setTimezone(timeZone)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setMinute(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("MINUTE", info)
          }

        case TruncDate(child, format) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          val (formatExpr, formatInfo) = exprToProtoInternal(format, inputs)

          if (childExpr.isDefined && formatExpr.isDefined) {
            val builder = ExprOuterClass.TruncDate.newBuilder()
            builder.setChild(childExpr.get)
            builder.setFormat(formatExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setTruncDate(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("TRUNCDATE", Seq(info, formatInfo))
          }

        case TruncTimestamp(format, child, timeZoneId) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          val (formatExpr, formatInfo) = exprToProtoInternal(format, inputs)

          if (childExpr.isDefined && formatExpr.isDefined) {
            val builder = ExprOuterClass.TruncTimestamp.newBuilder()
            builder.setChild(childExpr.get)
            builder.setFormat(formatExpr.get)

            val timeZone = timeZoneId.getOrElse("UTC")
            builder.setTimezone(timeZone)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setTruncTimestamp(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("TRUNCTIMESTAMP", Seq(info, formatInfo))
          }

        case Second(child, timeZoneId) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)

          if (childExpr.isDefined) {
            val builder = ExprOuterClass.Second.newBuilder()
            builder.setChild(childExpr.get)

            val timeZone = timeZoneId.getOrElse("UTC")
            builder.setTimezone(timeZone)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setSecond(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("SECOND", info)
          }

        case Year(child) =>
          val (periodType, _) = exprToProtoInternal(Literal("year"), inputs)
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProto("datepart", Seq(periodType, childExpr): _*)
            .map(e => {
              Expr
                .newBuilder()
                .setCast(
                  ExprOuterClass.Cast
                    .newBuilder()
                    .setChild(e)
                    .setDatatype(serializeDataType(IntegerType).get)
                    .build())
                .build()
            }) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("YEAR", info)
          }

        case IsNull(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)

          if (childExpr.isDefined) {
            val castBuilder = ExprOuterClass.IsNull.newBuilder()
            castBuilder.setChild(childExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setIsNull(castBuilder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("ISNULL", info)
          }

        case IsNotNull(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)

          if (childExpr.isDefined) {
            val castBuilder = ExprOuterClass.IsNotNull.newBuilder()
            castBuilder.setChild(childExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setIsNotNull(castBuilder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("ISNOTNULL", info)
          }

        case SortOrder(child, direction, nullOrdering, _) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)

          if (childExpr.isDefined) {
            val sortOrderBuilder = ExprOuterClass.SortOrder.newBuilder()
            sortOrderBuilder.setChild(childExpr.get)

            direction match {
              case Ascending => sortOrderBuilder.setDirectionValue(0)
              case Descending => sortOrderBuilder.setDirectionValue(1)
            }

            nullOrdering match {
              case NullsFirst => sortOrderBuilder.setNullOrderingValue(0)
              case NullsLast => sortOrderBuilder.setNullOrderingValue(1)
            }

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setSortOrder(sortOrderBuilder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("SORTORDER", info)
          }

        case And(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.And.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setAnd(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("AND", Seq(leftInfo, rightInfo))
          }

        case Or(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.Or.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setOr(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("OR", Seq(leftInfo, rightInfo))
          }

        case UnaryExpression(child) if expr.prettyName == "promote_precision" =>
          // `UnaryExpression` includes `PromotePrecision` for Spark 3.2 & 3.3
          // `PromotePrecision` is just a wrapper, don't need to serialize it.
          exprToProtoInternal(child, inputs)

        case CheckOverflow(child, dt, nullOnOverflow) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)

          if (childExpr.isDefined) {
            val builder = ExprOuterClass.CheckOverflow.newBuilder()
            builder.setChild(childExpr.get)
            builder.setFailOnError(!nullOnOverflow)

            // `dataType` must be decimal type
            val dataType = serializeDataType(dt)
            builder.setDatatype(dataType.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setCheckOverflow(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("CHECKOVERFLOW", info)
          }

        case attr: AttributeReference =>
          val dataType = serializeDataType(attr.dataType)

          if (dataType.isDefined) {
            if (binding) {
              val boundRef = BindReferences
                .bindReference(attr, inputs, allowFailures = false)
                .asInstanceOf[BoundReference]
              val boundExpr = ExprOuterClass.BoundReference
                .newBuilder()
                .setIndex(boundRef.ordinal)
                .setDatatype(dataType.get)
                .build()

              (
                Some(
                  ExprOuterClass.Expr
                    .newBuilder()
                    .setBound(boundExpr)
                    .build()),
                CometExplainInfo.none)
            } else {
              val unboundRef = ExprOuterClass.UnboundReference
                .newBuilder()
                .setDatatype(dataType.get)
                .build()

              (
                Some(
                  ExprOuterClass.Expr
                    .newBuilder()
                    .setUnbound(unboundRef)
                    .build()),
                CometExplainInfo.none)
            }
          } else {
            unsupported("ATTRREF", CometExplainInfo(s"unsupported datatype: ${attr.dataType}"))
          }

        case Abs(child, _) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          if (childExpr.isDefined) {
            val abs =
              ExprOuterClass.Abs
                .newBuilder()
                .setChild(childExpr.get)
                .build()
            (Some(Expr.newBuilder().setAbs(abs).build()), CometExplainInfo.none)
          } else {
            unsupported("ABS", info)
          }

        case Acos(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProto("acos", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("ACOS", info)
          }

        case Asin(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProto("asin", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("ASIN", info)
          }

        case Atan(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProto("atan", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("ATAN", info)
          }

        case Atan2(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)
          scalarExprToProto("atan2", leftExpr, rightExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("ATAN2", Seq(leftInfo, rightInfo))
          }

        case e @ Ceil(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          child.dataType match {
            case t: DecimalType if t.scale == 0 => // zero scale is no-op
              (childExpr, CometExplainInfo.none)
            case t: DecimalType if t.scale < 0 => // Spark disallows negative scale SPARK-30252
              unsupported("CEIL", Seq(info, CometExplainInfo("Decimal type has negative scale")))
            case _ =>
              scalarExprToProtoWithReturnType("ceil", e.dataType, childExpr) match {
                case Some(e) => (Some(e), CometExplainInfo.none)
                case None => unsupported("CEIL", info)
              }
          }

        case Cos(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProto("cos", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("COS", info)
          }

        case Exp(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProto("exp", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("EXP", info)
          }

        case e @ Floor(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          child.dataType match {
            case t: DecimalType if t.scale == 0 => // zero scale is no-op
              (childExpr, CometExplainInfo.none)
            case t: DecimalType if t.scale < 0 => // Spark disallows negative scale SPARK-30252
              unsupported("FLOOR", Seq(info, CometExplainInfo("Decimal type has negative scale")))
            case _ =>
              scalarExprToProtoWithReturnType("floor", e.dataType, childExpr) match {
                case Some(e) => (Some(e), CometExplainInfo.none)
                case None => unsupported("FLOOR", info)
              }
          }

        case Log(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProto("ln", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("LN", info)
          }

        case Log10(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProto("log10", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("LOG10", info)
          }

        case Log2(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProto("log2", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("LOG2", info)
          }

        case Pow(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)
          scalarExprToProto("pow", leftExpr, rightExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("ACOS", Seq(leftInfo, rightInfo))
          }

        // round function for Spark 3.2 does not allow negative round target scale. In addition,
        // it has different result precision/scale for decimals. Supporting only 3.3 and above.
        case r: Round if !isSpark32 =>
          // _scale s a constant, copied from Spark's RoundBase because it is a protected val
          val scaleV: Any = r.scale.eval(EmptyRow)
          val _scale: Int = scaleV.asInstanceOf[Int]

          lazy val (childExpr, info) = exprToProtoInternal(r.child, inputs)
          r.child.dataType match {
            case t: DecimalType if t.scale < 0 => // Spark disallows negative scale SPARK-30252
              unsupported("ROUND", Seq(info, CometExplainInfo("Decimal type has negative scale")))
            case _ if scaleV == null =>
              val (childScaleIsNull, infoScaleIsNull) =
                exprToProtoInternal(Literal(null), inputs)
              childScaleIsNull match {
                case Some(e) => (Some(e), CometExplainInfo.none)
                case None => unsupported("ROUND", Seq(info, infoScaleIsNull))
              }
            case _: ByteType | ShortType | IntegerType | LongType if _scale >= 0 =>
              (
                childExpr,
                CometExplainInfo.none
              ) // _scale(I.e. decimal place) >= 0 is a no-op for integer types in Spark
            case _: FloatType | DoubleType =>
              // We cannot properly match with the Spark behavior for floating-point numbers.
              // Spark uses BigDecimal for rounding float/double, and BigDecimal fist converts a
              // double to string internally in order to create its own internal representation.
              // The problem is BigDecimal uses java.lang.Double.toString() and it has complicated
              // rounding algorithm. E.g. -5.81855622136895E8 is actually
              // -581855622.13689494132995605468750. Note the 5th fractional digit is 4 instead of
              // 5. Java(Scala)'s toString() rounds it up to -581855622.136895. This makes a
              // difference when rounding at 5th digit, I.e. round(-5.81855622136895E8, 5) should be
              // -5.818556221369E8, instead of -5.8185562213689E8. There is also an example that
              // toString() does NOT round up. 6.1317116247283497E18 is 6131711624728349696. It can
              // be rounded up to 6.13171162472835E18 that still represents the same double number.
              // I.e. 6.13171162472835E18 == 6.1317116247283497E18. However, toString() does not.
              // That results in round(6.1317116247283497E18, -5) == 6.1317116247282995E18 instead
              // of 6.1317116247283999E18.
              unsupported(
                "ROUND",
                CometExplainInfo("Comet does not support Spark's BigDecimal rounding"))
            case _ =>
              // `scale` must be Int64 type in DataFusion
              val (scaleExpr, info) =
                exprToProtoInternal(Literal(_scale.toLong, LongType), inputs)
              scalarExprToProtoWithReturnType("round", r.dataType, childExpr, scaleExpr) match {
                case Some(e) => (Some(e), CometExplainInfo.none)
                case None => unsupported("ROUND", info)
              }
          }

        case Signum(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProto("signum", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("SIGNUM", info)
          }

        case Sin(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProto("sin", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("SIN", info)
          }

        case Sqrt(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProto("sqrt", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("SQRT", info)
          }

        case Tan(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProto("tan", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("TAN", info)
          }

        case Ascii(child) =>
          val (childExpr, info) = exprToProtoInternal(Cast(child, StringType), inputs)
          scalarExprToProto("ascii", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("ASCII", info)
          }

        case BitLength(child) =>
          val (childExpr, info) = exprToProtoInternal(Cast(child, StringType), inputs)
          scalarExprToProto("bit_length", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("BIT_LENGTH", info)
          }

        case If(predicate, trueValue, falseValue) =>
          val (predicateExpr, predicateInfo) = exprToProtoInternal(predicate, inputs)
          val (trueExpr, trueInfo) = exprToProtoInternal(trueValue, inputs)
          val (falseExpr, falseInfo) = exprToProtoInternal(falseValue, inputs)
          if (predicateExpr.isDefined && trueExpr.isDefined && falseExpr.isDefined) {
            val builder = ExprOuterClass.IfExpr.newBuilder()
            builder.setIfExpr(predicateExpr.get)
            builder.setTrueExpr(trueExpr.get)
            builder.setFalseExpr(falseExpr.get)
            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setIf(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("IF", Seq(predicateInfo, trueInfo, falseInfo))
          }

        case CaseWhen(branches, elseValue) =>
          val (whenSeq, whenInfo) =
            branches.map(elements => exprToProtoInternal(elements._1, inputs)).unzip
          val (thenSeq, thenInfo) =
            branches.map(elements => exprToProtoInternal(elements._2, inputs)).unzip
          assert(whenSeq.length == thenSeq.length)
          if (whenSeq.forall(_.isDefined) && thenSeq.forall(_.isDefined)) {
            val builder = ExprOuterClass.CaseWhen.newBuilder()
            builder.addAllWhen(whenSeq.map(_.get).asJava)
            builder.addAllThen(thenSeq.map(_.get).asJava)
            if (elseValue.isDefined) {
              val (elseValueExpr, elseValueInfo) =
                exprToProtoInternal(elseValue.get, inputs)
              if (elseValueExpr.isDefined) {
                builder.setElseExpr(elseValueExpr.get)
              } else {
                return unsupported("CASE", elseValueInfo)
              }
            }
            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setCaseWhen(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("CASE", whenInfo ++ thenInfo)
          }
        case ConcatWs(children) =>
          val exprs = children.map(e => exprToProtoInternal(Cast(e, StringType), inputs))
          scalarExprToProto("concat_ws", exprs.map(_._1): _*) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("CONCAT_WS", exprs.map(_._2))
          }

        case Chr(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProto("chr", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("CHR", info)
          }

        case InitCap(child) =>
          val (childExpr, info) = exprToProtoInternal(Cast(child, StringType), inputs)
          scalarExprToProto("initcap", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("INITCAP", info)
          }

        case Length(child) =>
          val (childExpr, info) = exprToProtoInternal(Cast(child, StringType), inputs)
          scalarExprToProto("length", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("LENGTH", info)
          }

        case Lower(child) =>
          val (childExpr, info) = exprToProtoInternal(Cast(child, StringType), inputs)
          scalarExprToProto("lower", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("LOWER", info)
          }

        case Md5(child) =>
          val (childExpr, info) = exprToProtoInternal(Cast(child, StringType), inputs)
          scalarExprToProto("md5", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("MD5", info)
          }

        case OctetLength(child) =>
          val (childExpr, info) = exprToProtoInternal(Cast(child, StringType), inputs)
          scalarExprToProto("octet_length", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("OCTET_LENGTH", info)
          }

        case Reverse(child) =>
          val (childExpr, info) = exprToProtoInternal(Cast(child, StringType), inputs)
          scalarExprToProto("reverse", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("REVERSE", info)
          }

        case StringInstr(str, substr) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(Cast(str, StringType), inputs)
          val (rightExpr, rightInfo) =
            exprToProtoInternal(Cast(substr, StringType), inputs)
          scalarExprToProto("strpos", leftExpr, rightExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("STRPOS", Seq(leftInfo, rightInfo))
          }

        case StringRepeat(str, times) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(Cast(str, StringType), inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(Cast(times, LongType), inputs)
          scalarExprToProto("repeat", leftExpr, rightExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("REPEAT", Seq(leftInfo, rightInfo))
          }

        case StringReplace(src, search, replace) =>
          val (srcExpr, srcInfo) = exprToProtoInternal(Cast(src, StringType), inputs)
          val (searchExpr, searchInfo) =
            exprToProtoInternal(Cast(search, StringType), inputs)
          val (replaceExpr, replaceInfo) =
            exprToProtoInternal(Cast(replace, StringType), inputs)
          scalarExprToProto("replace", srcExpr, searchExpr, replaceExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("REPLACE", Seq(srcInfo, searchInfo, replaceInfo))
          }

        case StringTranslate(src, matching, replace) =>
          val (srcExpr, srcInfo) = exprToProtoInternal(Cast(src, StringType), inputs)
          val (matchingExpr, matchingInfo) =
            exprToProtoInternal(Cast(matching, StringType), inputs)
          val (replaceExpr, replaceInfo) =
            exprToProtoInternal(Cast(replace, StringType), inputs)
          scalarExprToProto("translate", srcExpr, matchingExpr, replaceExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("TRANSLATE", Seq(srcInfo, matchingInfo, replaceInfo))
          }

        case StringTrim(srcStr, trimStr) =>
          trim(srcStr, trimStr, inputs, "trim")

        case StringTrimLeft(srcStr, trimStr) =>
          trim(srcStr, trimStr, inputs, "ltrim")

        case StringTrimRight(srcStr, trimStr) =>
          trim(srcStr, trimStr, inputs, "rtrim")

        case StringTrimBoth(srcStr, trimStr, _) =>
          trim(srcStr, trimStr, inputs, "btrim")

        case Upper(child) =>
          val (childExpr, info) = exprToProtoInternal(Cast(child, StringType), inputs)
          scalarExprToProto("upper", childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("UPPER", info)
          }

        case BitwiseAnd(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.BitwiseAnd.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setBitwiseAnd(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("BITWISE_AND", Seq(leftInfo, rightInfo))
          }

        case BitwiseNot(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)

          if (childExpr.isDefined) {
            val builder = ExprOuterClass.BitwiseNot.newBuilder()
            builder.setChild(childExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setBitwiseNot(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("BITWISE_NOT", info)
          }

        case BitwiseOr(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.BitwiseOr.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setBitwiseOr(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("BITWISE_OR", Seq(leftInfo, rightInfo))
          }

        case BitwiseXor(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = exprToProtoInternal(right, inputs)

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.BitwiseXor.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setBitwiseXor(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("BITWISE_XOR", Seq(leftInfo, rightInfo))
          }

        case ShiftRight(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = if (left.dataType == LongType) {
            // DataFusion bitwise shift right expression requires
            // same data type between left and right side
            exprToProtoInternal(Cast(right, LongType), inputs)
          } else {
            exprToProtoInternal(right, inputs)
          }

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.BitwiseShiftRight.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setBitwiseShiftRight(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("SHIFT_RIGHT", Seq(leftInfo, rightInfo))
          }

        case ShiftLeft(left, right) =>
          val (leftExpr, leftInfo) = exprToProtoInternal(left, inputs)
          val (rightExpr, rightInfo) = if (left.dataType == LongType) {
            // DataFusion bitwise shift left expression requires
            // same data type between left and right side
            exprToProtoInternal(Cast(right, LongType), inputs)
          } else {
            exprToProtoInternal(right, inputs)
          }

          if (leftExpr.isDefined && rightExpr.isDefined) {
            val builder = ExprOuterClass.BitwiseShiftLeft.newBuilder()
            builder.setLeft(leftExpr.get)
            builder.setRight(rightExpr.get)

            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setBitwiseShiftLeft(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("SHIFT_LEFT", Seq(leftInfo, rightInfo))
          }

        case In(value, list) =>
          in(value, list, inputs, false, "IN")

        case InSet(value, hset) =>
          val valueDataType = value.dataType
          val list = hset.map { setVal =>
            Literal(setVal, valueDataType)
          }.toSeq
          // Change `InSet` to `In` expression
          // We do Spark `InSet` optimization in native (DataFusion) side.
          in(value, list, inputs, false, "INSET")

        case Not(In(value, list)) =>
          in(value, list, inputs, true, "NOT_IN")

        case Not(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          if (childExpr.isDefined) {
            val builder = ExprOuterClass.Not.newBuilder()
            builder.setChild(childExpr.get)
            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setNot(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("NOT", info)
          }

        case UnaryMinus(child, _) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          if (childExpr.isDefined) {
            val builder = ExprOuterClass.Negative.newBuilder()
            builder.setChild(childExpr.get)
            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setNegative(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("UNARY_MINUS", info)
          }

        case a @ Coalesce(_) =>
          val (exprChildren, info) = a.children.map(exprToProtoInternal(_, inputs)).unzip
          val childExpr = scalarExprToProto("coalesce", exprChildren: _*)
          // TODO: Remove this once we have new DataFusion release which includes
          // the fix: https://github.com/apache/arrow-datafusion/pull/9459
          if (childExpr.isDefined) {
            castToProto(None, a.dataType, childExpr)
          } else {
            unsupported("COALESCE", info)
          }

        // With Spark 3.4, CharVarcharCodegenUtils.readSidePadding gets called to pad spaces for
        // char types. Use rpad to achieve the behavior.
        // See https://github.com/apache/spark/pull/38151
        case StaticInvoke(
              _: Class[CharVarcharCodegenUtils],
              _: StringType,
              "readSidePadding",
              arguments,
              _,
              true,
              false,
              true) if arguments.size == 2 =>
          val (argsExpr, argsInfo) = Seq(
            exprToProtoInternal(Cast(arguments(0), StringType), inputs),
            exprToProtoInternal(arguments(1), inputs)).unzip

          if (argsExpr.forall(_.isDefined)) {
            val builder = ExprOuterClass.ScalarFunc.newBuilder()
            builder.setFunc("rpad")
            argsExpr.foreach(arg => builder.addArgs(arg.get))

            (
              Some(ExprOuterClass.Expr.newBuilder().setScalarFunc(builder).build()),
              CometExplainInfo.none)
          } else {
            unsupported("STATICINVOKE_RPAD", argsInfo)
          }

        case KnownFloatingPointNormalized(NormalizeNaNAndZero(expr)) =>
          val name = "FP_NORMALIZED"
          val dataType = serializeDataType(expr.dataType)
          if (dataType.isEmpty) {
            return unsupported(name, CometExplainInfo(s"Unsupported datatype ${expr.dataType}"))
          }
          val (ex, _) = exprToProtoInternal(expr, inputs)
          (
            ex.map { child =>
              val builder = ExprOuterClass.NormalizeNaNAndZero
                .newBuilder()
                .setChild(child)
                .setDatatype(dataType.get)
              ExprOuterClass.Expr.newBuilder().setNormalizeNanAndZero(builder).build()
            },
            CometExplainInfo.none)

        case s @ execution.ScalarSubquery(_, _) =>
          val dataType = serializeDataType(s.dataType)
          if (dataType.isEmpty) {
            return unsupported(
              "SCALAR_SUBQUERY",
              CometExplainInfo(s"Unsupported datatype ${s.dataType}"))
          }

          val builder = ExprOuterClass.Subquery
            .newBuilder()
            .setId(s.exprId.id)
            .setDatatype(dataType.get)
          (
            Some(ExprOuterClass.Expr.newBuilder().setSubquery(builder).build()),
            CometExplainInfo.none)

        case UnscaledValue(child) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProtoWithReturnType("unscaled_value", LongType, childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("UNSCALED_VALUE", info)
          }

        case MakeDecimal(child, precision, scale, true) =>
          val (childExpr, info) = exprToProtoInternal(child, inputs)
          scalarExprToProtoWithReturnType(
            "make_decimal",
            DecimalType(precision, scale),
            childExpr) match {
            case Some(e) => (Some(e), CometExplainInfo.none)
            case None => unsupported("MAKE_DECIMAL", info)
          }

        case b @ BinaryExpression(_, _) if isBloomFilterMightContain(b) =>
          val bloomFilter = b.left
          val value = b.right
          val (bloomFilterExpr, bloomInfo) = exprToProtoInternal(bloomFilter, inputs)
          val (valueExpr, valueInfo) = exprToProtoInternal(value, inputs)
          if (bloomFilterExpr.isDefined && valueExpr.isDefined) {
            val builder = ExprOuterClass.BloomFilterMightContain.newBuilder()
            builder.setBloomFilter(bloomFilterExpr.get)
            builder.setValue(valueExpr.get)
            (
              Some(
                ExprOuterClass.Expr
                  .newBuilder()
                  .setBloomFilterMightContain(builder)
                  .build()),
              CometExplainInfo.none)
          } else {
            unsupported("BLOOMFILTER", Seq(bloomInfo, valueInfo))
          }

        case _ =>
          unsupported(expr.prettyName, CometExplainInfo(s"${expr.prettyName} is not supported"))
      }
    }

    def trim(
        srcStr: Expression,
        trimStr: Option[Expression],
        inputs: Seq[Attribute],
        trimType: String): (Option[Expr], CometExplainInfo) = {
      val (srcExpr, srcInfo) = exprToProtoInternal(Cast(srcStr, StringType), inputs)
      if (trimStr.isDefined) {
        val (trimExpr, trimInfo) =
          exprToProtoInternal(Cast(trimStr.get, StringType), inputs)
        scalarExprToProto(trimType, srcExpr, trimExpr) match {
          case Some(e) => (Some(e), CometExplainInfo.none)
          case None => unsupported(trimType.toUpperCase(), Seq(srcInfo, trimInfo))
        }
      } else {
        scalarExprToProto(trimType, srcExpr) match {
          case Some(e) => (Some(e), CometExplainInfo.none)
          case None => unsupported(trimType.toUpperCase(), srcInfo)
        }
      }
    }

    def in(
        value: Expression,
        list: Seq[Expression],
        inputs: Seq[Attribute],
        negate: Boolean,
        displayName: String): (Option[Expr], CometExplainInfo) = {
      val (valueExpr, valueInfo) = exprToProtoInternal(value, inputs)
      val (listExprs, listInfos) = list.map(exprToProtoInternal(_, inputs)).unzip
      if (valueExpr.isDefined && listExprs.forall(_.isDefined)) {
        val builder = ExprOuterClass.In.newBuilder()
        builder.setInValue(valueExpr.get)
        builder.addAllLists(listExprs.map(_.get).asJava)
        builder.setNegated(negate)
        (
          Some(
            ExprOuterClass.Expr
              .newBuilder()
              .setIn(builder)
              .build()),
          CometExplainInfo.none)
      } else {
        unsupported(displayName, listExprs ++ Seq(valueInfo))
      }
    }

    val conf = SQLConf.get
    val newExpr =
      DecimalPrecision.promote(conf.decimalOperationsAllowPrecisionLoss, expr, !conf.ansiEnabled)
    exprToProtoInternal(newExpr, input)
  }

  def scalarExprToProtoWithReturnType(
      funcName: String,
      returnType: DataType,
      args: Option[Expr]*): Option[Expr] = {
    val builder = ExprOuterClass.ScalarFunc.newBuilder()
    builder.setFunc(funcName)
    serializeDataType(returnType).flatMap { t =>
      builder.setReturnType(t)
      scalarExprToProto0(builder, args: _*)
    }
  }

  def scalarExprToProto(funcName: String, args: Option[Expr]*): Option[Expr] = {
    val builder = ExprOuterClass.ScalarFunc.newBuilder()
    builder.setFunc(funcName)
    scalarExprToProto0(builder, args: _*)
  }

  private def scalarExprToProto0(
      builder: ScalarFunc.Builder,
      args: Option[Expr]*): Option[Expr] = {
    args.foreach {
      case Some(a) => builder.addArgs(a)
      case _ => return None
    }
    Some(ExprOuterClass.Expr.newBuilder().setScalarFunc(builder).build())
  }

  def isPrimitive(expression: Expression): Boolean = expression.dataType match {
    case _: ByteType | _: ShortType | _: IntegerType | _: LongType | _: FloatType |
        _: DoubleType | _: TimestampType | _: DateType | _: BooleanType | _: DecimalType =>
      true
    case _ => false
  }

  def nullIfWhenPrimitive(expression: Expression): Expression = if (isPrimitive(expression)) {
    new NullIf(expression, Literal.default(expression.dataType)).child
  } else {
    expression
  }

  /**
   * Convert a Spark plan operator to a protobuf Comet operator.
   *
   * @param op
   *   Spark plan operator
   * @param childOp
   *   previously converted protobuf Comet operators, which will be consumed by the Spark plan
   *   operator as its children
   * @return
   *   The converted Comet native operator for the input `op`, or `None` if the `op` cannot be
   *   converted to a native operator.
   */
  def operator2Proto(op: SparkPlan, childOp: Operator*): (Option[Operator], CometExplainInfo) = {
    val result = OperatorOuterClass.Operator.newBuilder()
    childOp.foreach(result.addChildren)

    op match {
      case ProjectExec(projectList, child) if isCometOperatorEnabled(op.conf, "project") =>
        val (exprs, exprsInfo) = projectList.map(exprToProto(_, child.output)).unzip

        if (exprs.forall(_.isDefined) && childOp.nonEmpty) {
          val projectBuilder = OperatorOuterClass.Projection
            .newBuilder()
            .addAllProjectList(exprs.map(_.get).asJava)
          (Some(result.setProjection(projectBuilder).build()), null)
        } else {
          unsupported("CometProject", exprsInfo)
        }

      case FilterExec(condition, child) if isCometOperatorEnabled(op.conf, "filter") =>
        val (cond, info) = exprToProto(condition, child.output)

        if (cond.isDefined && childOp.nonEmpty) {
          val filterBuilder = OperatorOuterClass.Filter.newBuilder().setPredicate(cond.get)
          (Some(result.setFilter(filterBuilder).build()), CometExplainInfo.none)
        } else {
          unsupported("CometFilter", info)
        }

      case SortExec(sortOrder, _, child, _) if isCometOperatorEnabled(op.conf, "sort") =>
        val (sortOrders, sortOrdersInfo) = sortOrder.map(exprToProto(_, child.output)).unzip

        if (sortOrders.forall(_.isDefined) && childOp.nonEmpty) {
          val sortBuilder = OperatorOuterClass.Sort
            .newBuilder()
            .addAllSortOrders(sortOrders.map(_.get).asJava)
          (Some(result.setSort(sortBuilder).build()), CometExplainInfo.none)
        } else {
          unsupported("CometSort", sortOrdersInfo)
        }

      case LocalLimitExec(limit, _) if isCometOperatorEnabled(op.conf, "local_limit") =>
        if (childOp.nonEmpty) {
          // LocalLimit doesn't use offset, but it shares same operator serde class.
          // Just set it to zero.
          val limitBuilder = OperatorOuterClass.Limit
            .newBuilder()
            .setLimit(limit)
            .setOffset(0)
          (Some(result.setLimit(limitBuilder).build()), CometExplainInfo.none)
        } else {
          unsupported("CometLocalLimit", CometExplainInfo("No child operator"))
        }

      case globalLimitExec: GlobalLimitExec if isCometOperatorEnabled(op.conf, "global_limit") =>
        if (childOp.nonEmpty) {
          val limitBuilder = OperatorOuterClass.Limit.newBuilder()

          // Spark 3.2 doesn't support offset for GlobalLimit, but newer Spark versions
          // support it. Before we upgrade to Spark 3.3, just set it zero.
          // TODO: Spark 3.3 might have negative limit (-1) for Offset usage.
          // When we upgrade to Spark 3.3., we need to address it here.
          assert(globalLimitExec.limit >= 0, "limit should be greater or equal to zero")
          limitBuilder.setLimit(globalLimitExec.limit)
          limitBuilder.setOffset(0)

          (Some(result.setLimit(limitBuilder).build()), CometExplainInfo.none)
        } else {
          unsupported("CometGlobalLimit", CometExplainInfo("No child operator"))
        }

      case ExpandExec(projections, _, child) if isCometOperatorEnabled(op.conf, "expand") =>
        val (projExprs, projInfos) =
          projections.flatMap(_.map(exprToProto(_, child.output))).unzip

        if (projExprs.forall(_.isDefined) && childOp.nonEmpty) {
          val expandBuilder = OperatorOuterClass.Expand
            .newBuilder()
            .addAllProjectList(projExprs.map(_.get).asJava)
            .setNumExprPerProject(projections.head.size)
          (Some(result.setExpand(expandBuilder).build()), CometExplainInfo.none)
        } else {
          unsupported("CometExpand", projInfos)
        }

      case HashAggregateExec(
            _,
            _,
            _,
            groupingExpressions,
            aggregateExpressions,
            aggregateAttributes,
            _,
            resultExpressions,
            child) if isCometOperatorEnabled(op.conf, "aggregate") =>
        if (groupingExpressions.isEmpty && aggregateExpressions.isEmpty) {
          return unsupported("CometHashAggregate", CometExplainInfo("No group by or aggregation"))
        }

        val (groupingExprs, groupingExprsInfos) =
          groupingExpressions.map(exprToProto(_, child.output)).unzip

        // In some of the cases, the aggregateExpressions could be empty.
        // For example, if the aggregate functions only have group by or if the aggregate
        // functions only have distinct aggregate functions:
        //
        // SELECT COUNT(distinct col2), col1 FROM test group by col1
        //  +- HashAggregate (keys =[col1# 6], functions =[count (distinct col2#7)] )
        //    +- Exchange hashpartitioning (col1#6, 10), ENSURE_REQUIREMENTS, [plan_id = 36]
        //      +- HashAggregate (keys =[col1#6], functions =[partial_count (distinct col2#7)] )
        //        +- HashAggregate (keys =[col1#6, col2#7], functions =[] )
        //          +- Exchange hashpartitioning (col1#6, col2#7, 10), ENSURE_REQUIREMENTS, ...
        //            +- HashAggregate (keys =[col1#6, col2#7], functions =[] )
        //              +- FileScan parquet spark_catalog.default.test[col1#6, col2#7] ......
        // If the aggregateExpressions is empty, we only want to build groupingExpressions,
        // and skip processing of aggregateExpressions.
        if (aggregateExpressions.isEmpty) {
          val hashAggBuilder = OperatorOuterClass.HashAggregate.newBuilder()
          hashAggBuilder.addAllGroupingExprs(groupingExprs.map(_.get).asJava)
          val attributes = groupingExpressions.map(_.toAttribute) ++ aggregateAttributes
          val (resultExprs, _) = resultExpressions.map(exprToProto(_, attributes)).unzip
          if (resultExprs.exists(_.isEmpty)) {
            val msg = s"Unsupported result expressions found in: ${resultExpressions}"
            emitWarning(msg)
            return unsupported("CometHashAggregate", CometExplainInfo(msg))
          }
          hashAggBuilder.addAllResultExprs(resultExprs.map(_.get).asJava)
          (Some(result.setHashAgg(hashAggBuilder).build()), CometExplainInfo.none)
        } else {
          val modes = aggregateExpressions.map(_.mode).distinct

          if (modes.size != 1) {
            // This shouldn't happen as all aggregation expressions should share the same mode.
            // Fallback to Spark nevertheless here.
            return unsupported(
              "CometHashAggregate",
              CometExplainInfo("All aggregate expressions do not have the same mode"))
          }

          val mode = modes.head match {
            case Partial => CometAggregateMode.Partial
            case Final => CometAggregateMode.Final
            case _ =>
              return unsupported(
                "CometHashAggregate",
                CometExplainInfo(s"Unsupported aggregation mode ${modes.head}"))
          }

          val output = mode match {
            case CometAggregateMode.Partial => child.output
            case CometAggregateMode.Final =>
              // Assuming `Final` always follows `Partial` aggregation, this find the first
              // `Partial` aggregation and get the input attributes from it.
              // During finding partial aggregation, we must ensure all traversed op are
              // native operators. If not, we should fallback to Spark.
              var seenNonNativeOp = false
              var partialAggInput: Option[Seq[Attribute]] = None
              child.transformDown {
                case op if !op.isInstanceOf[CometPlan] =>
                  seenNonNativeOp = true
                  op
                case op @ CometHashAggregateExec(_, _, _, _, input, Some(Partial), _, _) =>
                  if (!seenNonNativeOp && partialAggInput.isEmpty) {
                    partialAggInput = Some(input)
                  }
                  op
              }

              if (partialAggInput.isDefined) {
                partialAggInput.get
              } else {
                return unsupported(
                  "CometHashAggregate",
                  CometExplainInfo("No input for partial aggregate"))
              }
            case _ =>
              return unsupported(
                "CometHashAggregate",
                CometExplainInfo(s"Unsupported mode $mode"))
          }
          val binding = if (mode == CometAggregateMode.Final) {
// In final mode, the aggregate expressions are bound to the output of the
// child and partial aggregate expressions buffer attributes produced by partial
// aggregation. This is done in Spark `HashAggregateExec` internally. In Comet,
// we don't have to do this because we don't use the merging expression.
            false
          } else {
            true
          }

          val (aggExprs, aggExprsInfos) =
            aggregateExpressions.map(aggExprToProto(_, output, binding)).unzip
          if (childOp.nonEmpty && groupingExprs.forall(_.isDefined) &&
            aggExprs.forall(_.isDefined)) {
            val hashAggBuilder = OperatorOuterClass.HashAggregate.newBuilder()
            hashAggBuilder.addAllGroupingExprs(groupingExprs.map(_.get).asJava)
            hashAggBuilder.addAllAggExprs(aggExprs.map(_.get).asJava)
            if (mode == CometAggregateMode.Final) {
              val attributes = groupingExpressions.map(_.toAttribute) ++ aggregateAttributes
              val (resultExprs, _) = resultExpressions.map(exprToProto(_, attributes)).unzip
              if (resultExprs.exists(_.isEmpty)) {
                val msg = s"Unsupported result expressions found in: ${resultExpressions}"
                emitWarning(msg)
                return unsupported("CometHashAggregate", CometExplainInfo(msg))
              }
              hashAggBuilder.addAllResultExprs(resultExprs.map(_.get).asJava)
            }
            hashAggBuilder.setModeValue(mode.getNumber)
            (Some(result.setHashAgg(hashAggBuilder).build()), CometExplainInfo.none)
          } else {
            unsupported("CometHashAggregate", aggExprsInfos ++ groupingExprsInfos)
          }
        }

      case join: HashJoin =>
        // `HashJoin` has only two implementations in Spark, but we check the type of the join to
        // make sure we are handling the correct join type.
        if (!(isCometOperatorEnabled(op.conf, "hash_join") &&
            join.isInstanceOf[ShuffledHashJoinExec]) &&
          !(isCometOperatorEnabled(op.conf, "broadcast_hash_join") &&
            join.isInstanceOf[BroadcastHashJoinExec])) {
          return unsupported("HashJoin", s"Invalid hash join type ${join.nodeName}")
        }

        if (join.buildSide == BuildRight) {
          // DataFusion HashJoin assumes build side is always left.
          // TODO: support BuildRight
          return unsupported("HashJoin", "BuildRight is not supported")
        }

        val condition = join.condition.map { cond =>
          val (condProto, condInfo) = exprToProto(cond, join.left.output ++ join.right.output)
          if (condProto.isEmpty) {
            return unsupported("HashJoin", condInfo)
          }
          condProto.get
        }

        val joinType = join.joinType match {
          case Inner => JoinType.Inner
          case LeftOuter => JoinType.LeftOuter
          case RightOuter => JoinType.RightOuter
          case FullOuter => JoinType.FullOuter
          case LeftSemi => JoinType.LeftSemi
          case LeftAnti => JoinType.LeftAnti
          case _ =>
            return unsupported(
              "HashJoin",
              s"Unsupported join type ${join.joinType}"
            ) // Spark doesn't support other join types
        }

        val (leftKeys, leftInfos) = join.leftKeys.map(exprToProto(_, join.left.output)).unzip
        val (rightKeys, rightInfos) = join.rightKeys.map(exprToProto(_, join.right.output)).unzip

        if (leftKeys.forall(_.isDefined) &&
          rightKeys.forall(_.isDefined) &&
          childOp.nonEmpty) {
          val joinBuilder = OperatorOuterClass.HashJoin
            .newBuilder()
            .setJoinType(joinType)
            .addAllLeftJoinKeys(leftKeys.map(_.get).asJava)
            .addAllRightJoinKeys(rightKeys.map(_.get).asJava)
          condition.foreach(joinBuilder.setCondition)
          (Some(result.setHashJoin(joinBuilder).build()), CometExplainInfo.none)
        } else {
          unsupported("HashJoin", leftInfos ++ rightInfos)
        }

      case join: SortMergeJoinExec if isCometOperatorEnabled(op.conf, "sort_merge_join") =>
        // `requiredOrders` and `getKeyOrdering` are copied from Spark's SortMergeJoinExec.
        def requiredOrders(keys: Seq[Expression]): Seq[SortOrder] = {
          keys.map(SortOrder(_, Ascending))
        }

        def getKeyOrdering(
            keys: Seq[Expression],
            childOutputOrdering: Seq[SortOrder]): Seq[SortOrder] = {
          val requiredOrdering = requiredOrders(keys)
          if (SortOrder.orderingSatisfies(childOutputOrdering, requiredOrdering)) {
            keys.zip(childOutputOrdering).map { case (key, childOrder) =>
              val sameOrderExpressionsSet = ExpressionSet(childOrder.children) - key
              SortOrder(key, Ascending, sameOrderExpressionsSet.toSeq)
            }
          } else {
            requiredOrdering
          }
        }

        // TODO: Support SortMergeJoin with join condition after new DataFusion release
        if (join.condition.isDefined) {
          return unsupported(
            op.nodeName,
            CometExplainInfo("Sort merge join with a join condition is not supported"))
        }

        val joinType = join.joinType match {
          case Inner => JoinType.Inner
          case LeftOuter => JoinType.LeftOuter
          case RightOuter => JoinType.RightOuter
          case FullOuter => JoinType.FullOuter
          case LeftSemi => JoinType.LeftSemi
          case LeftAnti => JoinType.LeftAnti
          case _ =>
            return unsupported(
              op.nodeName,
              CometExplainInfo(s"Unsupported join type ${join.joinType}")
            ) // Spark doesn't support other join types
        }

        val (leftKeys, leftInfo) = join.leftKeys.map(exprToProto(_, join.left.output)).unzip
        val (rightKeys, rightInfo) = join.rightKeys.map(exprToProto(_, join.right.output)).unzip

        val (sortOptions, sortOptionsInfo) =
          getKeyOrdering(join.leftKeys, join.left.outputOrdering)
            .map(exprToProto(_, join.left.output))
            .unzip

        if (sortOptions.forall(_.isDefined) &&
          leftKeys.forall(_.isDefined) &&
          rightKeys.forall(_.isDefined) &&
          childOp.nonEmpty) {
          val joinBuilder = OperatorOuterClass.SortMergeJoin
            .newBuilder()
            .setJoinType(joinType)
            .addAllSortOptions(sortOptions.map(_.get).asJava)
            .addAllLeftJoinKeys(leftKeys.map(_.get).asJava)
            .addAllRightJoinKeys(rightKeys.map(_.get).asJava)
          (Some(result.setSortMergeJoin(joinBuilder).build()), CometExplainInfo.none)
        } else {

          unsupported(op.nodeName, leftInfo ++ rightInfo ++ sortOptionsInfo)
        }

      case op if isCometSink(op) =>
        // These operators are source of Comet native execution chain
        val scanBuilder = OperatorOuterClass.Scan.newBuilder()

        val scanTypes = op.output.flatten { attr =>
          serializeDataType(attr.dataType)
        }

        if (scanTypes.length == op.output.length) {
          scanBuilder.addAllFields(scanTypes.asJava)

          // Sink operators don't have children
          result.clearChildren()

          (Some(result.setScan(scanBuilder).build()), CometExplainInfo.none)
        } else {
          // There are unsupported scan type
          val msg =
            s"unsupported Comet operator: ${op.nodeName}, due to unsupported data types above"
          emitWarning(msg)
          unsupported(op.nodeName, msg)
        }

      case op =>
        // Emit warning if:
        //  1. it is not Spark shuffle operator, which is handled separately
        //  2. it is not a Comet operator
        if (!op.nodeName.contains("Comet") && !op.isInstanceOf[ShuffleExchangeExec]) {
          emitWarning(s"unsupported Spark operator: ${op.nodeName}")
        }
        unsupported(op.nodeName, CometExplainInfo.none)
    }
  }

  /**
   * Whether the input Spark operator `op` can be considered as a Comet sink, i.e., the start of
   * native execution. If it is true, we'll wrap `op` with `CometScanWrapper` or
   * `CometSinkPlaceHolder` later in `CometSparkSessionExtensions` after `operator2proto` is
   * called.
   */
  private def isCometSink(op: SparkPlan): Boolean = {
    op match {
      case s if isCometScan(s) => true
      case _: CometSinkPlaceHolder => true
      case _: CoalesceExec => true
      case _: UnionExec => true
      case _: ShuffleExchangeExec => true
      case ShuffleQueryStageExec(_, _: CometShuffleExchangeExec, _) => true
      case ShuffleQueryStageExec(_, ReusedExchangeExec(_, _: CometShuffleExchangeExec), _) => true
      case _: TakeOrderedAndProjectExec => true
      case BroadcastQueryStageExec(_, _: CometBroadcastExchangeExec, _) => true
      case _: BroadcastExchangeExec => true
      case _ => false
    }
  }

  /**
   * Checks whether `dt` is a decimal type AND whether Spark version is before 3.4
   */
  private def decimalBeforeSpark34(dt: DataType): Boolean = {
    !isSpark34Plus && (dt match {
      case _: DecimalType => true
      case _ => false
    })
  }

  /**
   * Check if the datatypes of shuffle input are supported. This is used for Columnar shuffle
   * which supports struct/array.
   */
  def supportPartitioningTypes(
      inputs: Seq[Attribute],
      partitioning: Partitioning): (Boolean, String) = {
    def supportedDataType(dt: DataType): Boolean = dt match {
      case _: ByteType | _: ShortType | _: IntegerType | _: LongType | _: FloatType |
          _: DoubleType | _: StringType | _: BinaryType | _: TimestampType | _: DecimalType |
          _: DateType | _: BooleanType =>
        true
      case StructType(fields) =>
        fields.forall(f => supportedDataType(f.dataType))
      case ArrayType(ArrayType(_, _), _) => false // TODO: nested array is not supported
      case ArrayType(MapType(_, _, _), _) => false // TODO: map array element is not supported
      case ArrayType(elementType, _) =>
        supportedDataType(elementType)
      case MapType(MapType(_, _, _), _, _) => false // TODO: nested map is not supported
      case MapType(_, MapType(_, _, _), _) => false
      case MapType(StructType(_), _, _) => false // TODO: struct map key/value is not supported
      case MapType(_, StructType(_), _) => false
      case MapType(ArrayType(_, _), _, _) => false // TODO: array map key/value is not supported
      case MapType(_, ArrayType(_, _), _) => false
      case MapType(keyType, valueType, _) =>
        supportedDataType(keyType) && supportedDataType(valueType)
      case _ =>
        false
    }

    // Check if the datatypes of shuffle input are supported.
    val supported = inputs.forall(attr => supportedDataType(attr.dataType))
    if (!supported) {
      val msg = s"unsupported Spark partitioning: ${inputs.map(_.dataType)}"
      emitWarning(msg)
      (false, msg)
    } else {
      partitioning match {
        case HashPartitioning(expressions, _) =>
          (expressions.map(QueryPlanSerde.exprToProto(_, inputs)).forall(_._1.isDefined), null)
        case SinglePartition => (true, null)
        case _: RoundRobinPartitioning => (true, null)
        case RangePartitioning(ordering, _) =>
          (ordering.map(QueryPlanSerde.exprToProto(_, inputs)).forall(_._1.isDefined), null)
        case other =>
          val msg = s"unsupported Spark partitioning: ${other.getClass.getName}"
          emitWarning(msg)
          (false, msg)
      }
    }
  }

  /**
   * Whether the given Spark partitioning is supported by Comet.
   */
  def supportPartitioning(
      inputs: Seq[Attribute],
      partitioning: Partitioning): (Boolean, String) = {
    def supportedDataType(dt: DataType): Boolean = dt match {
      case _: ByteType | _: ShortType | _: IntegerType | _: LongType | _: FloatType |
          _: DoubleType | _: StringType | _: BinaryType | _: TimestampType | _: DecimalType |
          _: DateType | _: BooleanType =>
        true
      case _ =>
        // Native shuffle doesn't support struct/array yet
        false
    }

    // Check if the datatypes of shuffle input are supported.
    val supported = inputs.forall(attr => supportedDataType(attr.dataType))

    if (!supported) {
      val msg = s"unsupported Spark partitioning: ${inputs.map(_.dataType)}"
      emitWarning(msg)
      (false, msg)
    } else {
      partitioning match {
        case HashPartitioning(expressions, _) =>
          (expressions.map(QueryPlanSerde.exprToProto(_, inputs)).forall(_._1.isDefined), null)
        case SinglePartition => (true, null)
        case other =>
          val msg = s"unsupported Spark partitioning: ${other.getClass.getName}"
          emitWarning(msg)
          (false, msg)
      }
    }
  }

}

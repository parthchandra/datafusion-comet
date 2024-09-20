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

package org.apache.spark.sql.comet

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.{SparkConf, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.UnsafeRow
import org.apache.spark.sql.comet.util.Utils.getUnsafeRowBatchSize
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.unsafe.memory.{MemoryAllocator, MemoryBlock}

import org.apache.comet.Native
import org.apache.comet.vector.{CometVector, NativeUtil}

object CometUnsafeRowIterators extends Logging {

  private[sql] class ColumnBatchToSparkRowIter(
      conf: SparkConf,
      batch: ColumnarBatch,
      context: TaskContext)
      extends Iterator[InternalRow]
      with AutoCloseable {

    val nativeUtil = new NativeUtil()

    protected var closed: Boolean = false

    Option(context).foreach {
      _.addTaskCompletionListener[Unit] { _ =>
        close(true)
      }
    }

    private val blockSize = {
      val vectors = mutable.Buffer[CometVector]()
      for (i <- 0 until batch.numCols()) {
        vectors += batch.column(i).asInstanceOf[CometVector]
      }
      getUnsafeRowBatchSize(vectors.toArray)
    }
//    private val allocator: CometShuffleMemoryAllocator =
//      CometShuffleMemoryAllocator.getInstance(conf, context.taskMemoryManager(), blockSize)
    private val allocator: MemoryAllocator = MemoryAllocator.UNSAFE
    private val block = allocator.allocate(blockSize)

    private val unsafeRows = toUnsafeRows

    private def toUnsafeRows: mutable.ArrayBuffer[InternalRow] = {
      val numRows = batch.numRows()
      val numCols = batch.numCols()
      val rows = ArrayBuffer[InternalRow]()
      val (arrayAddrs, schemaAddrs) = nativeUtil.exportColumnarBatch(batch)
      val converted = getUnsafeRowsNative(block, arrayAddrs, schemaAddrs)
      val rowWidth = UnsafeRow.calculateBitSetWidthInBytes(numCols) + 8 * numCols
      for (rowNum <- 0 until numRows) {
        // TODO: Make UnsafeRow from the block for variable length types
        // We need the row start offsets for each row.
        val row = new UnsafeRow(batch.numCols())
        row.pointTo(block.getBaseObject, block.getBaseOffset + rowNum * rowWidth, rowWidth)
        rows += row
      }
      rows
    }

    //    private val rowIter: ju.Iterator[InternalRow] = colBatch.rowIterator()
    private val unsafeRowIter: Iterator[InternalRow] = unsafeRows.iterator

    override def hasNext: Boolean = unsafeRowIter.hasNext || {
      close(false)
      false
    }

    override def next(): InternalRow = unsafeRowIter.next()

    override def close(): Unit = {
      close(false)
    }

    protected def close(freeBlock: Boolean): Unit = {
      if (!closed) {
        closed = true
      }

    }

    def getUnsafeRowsNative(
        block: MemoryBlock,
        arrayAddrs: Array[Long],
        schemaAddrs: Array[Long]): Long = {
      val native = new Native()
      native.getUnsafeRowsNative(
        block.getBaseObject,
        block.getBaseOffset,
        block.size,
        arrayAddrs,
        schemaAddrs)
    }

  }

  def columnarBatchToSparkRowIter(
      conf: SparkConf,
      batch: ColumnarBatch,
      context: TaskContext): Iterator[InternalRow] = {

    new ColumnBatchToSparkRowIter(conf, batch, context)
  }
}

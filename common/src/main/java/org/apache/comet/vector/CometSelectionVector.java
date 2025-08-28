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

package org.apache.comet.vector;

import java.util.Arrays;
import java.util.List;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.ValueVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.dictionary.DictionaryProvider;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarArray;
import org.apache.spark.sql.vectorized.ColumnarMap;
import org.apache.spark.unsafe.Platform;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * A zero-copy selection vector that provides a view into another CometVector using a list of
 * indices. This implementation allows selecting specific elements from a source vector without
 * copying the underlying data.
 *
 * <p>For example, if the original vector has values [v0, v1, v2, v3, v4, v5, v6, v7] and the
 * selection indices are [0, 1, 3, 4, 5, 7], then this selection vector will logically represent
 * [v0, v1, v3, v4, v5, v7] without actually copying the data.
 */
public class CometSelectionVector extends CometVector {
  /** The original vector being selected from */
  private final CometVector originalVector;

  /** The indices that are selected from the original vector, stored as a CometVector */
  private final CometVector selectedIndicesVector;

  /** Number of selected elements */
  private final int numValues;

  /** Number of null values in the selection */
  private int numNulls;

  /** Whether nulls have been computed */
  private boolean nullsComputed = false;

  /**
   * Creates a new selection vector from the given vector and indices.
   *
   * @param originalVector The original vector to select from
   * @param selectedIndices The indices to select from the original vector
   * @throws IllegalArgumentException if any index is out of bounds
   */
  public CometSelectionVector(CometVector originalVector, int[] selectedIndices) {
    super(originalVector.dataType(), originalVector.useDecimal128);
    this.originalVector = originalVector;
    this.numValues = selectedIndices.length;

    // Create a CometVector from the selected indices array
    this.selectedIndicesVector =
        createIndicesVector(selectedIndices, originalVector.getValueVector().getAllocator());

    // Validate indices are within bounds
    int originalLength = originalVector.numValues();
    for (int i = 0; i < selectedIndices.length; i++) {
      int idx = selectedIndices[i];
      if (idx < 0 || idx >= originalLength) {
        throw new IllegalArgumentException(
            String.format(
                "Index %d is out of bounds for vector of length %d", idx, originalLength));
      }
    }
  }

  /**
   * Creates a CometVector containing the selection indices.
   *
   * @param indices The array of indices
   * @param allocator The buffer allocator to use
   * @return A CometVector containing the indices
   */
  private static CometVector createIndicesVector(int[] indices, BufferAllocator allocator) {
    IntVector indicesVector = new IntVector("selection_indices", allocator);
    indicesVector.allocateNew(indices.length);
    for (int i = 0; i < indices.length; i++) {
      indicesVector.set(i, indices[i]);
    }
    indicesVector.setValueCount(indices.length);

    return new CometPlainVector(indicesVector, false);
  }

  /**
   * Returns the original index for the given selection vector index.
   *
   * @param selectionIndex Index in the selection vector
   * @return The corresponding index in the original vector
   * @throws IndexOutOfBoundsException if selectionIndex is out of bounds
   */
  public int getOriginalIndex(int selectionIndex) {
    if (selectionIndex < 0 || selectionIndex >= numValues) {
      throw new IndexOutOfBoundsException(
          String.format(
              "Selection index %d is out of bounds for selection vector of length %d",
              selectionIndex, numValues));
    }
    return selectedIndicesVector.getInt(selectionIndex);
  }

  /**
   * Returns a reference to the original vector.
   *
   * @return The original CometVector
   */
  public CometVector getOriginalVector() {
    return originalVector;
  }

  /**
   * Returns the selection indices as a CometVector.
   *
   * @return A CometVector containing the selection indices
   */
  public CometVector getSelectedIndicesVector() {
    return selectedIndicesVector;
  }

  /**
   * Returns a copy of the selected indices.
   *
   * @return Array of selected indices
   */
  public int[] getSelectedIndices() {
    int[] indices = new int[numValues];
    for (int i = 0; i < numValues; i++) {
      indices[i] = selectedIndicesVector.getInt(i);
    }
    return indices;
  }

  /**
   * Creates a nested selection by applying additional indices to this selection vector. This allows
   * chaining selections without materialization.
   *
   * @param indices The indices to select from this selection vector
   * @return A new CometSelectionVector representing the nested selection
   * @throws IllegalArgumentException if any index is out of bounds
   */
  public CometSelectionVector take(int[] indices) {
    int[] newIndices = new int[indices.length];
    for (int i = 0; i < indices.length; i++) {
      int idx = indices[i];
      if (idx < 0 || idx >= numValues) {
        throw new IllegalArgumentException(
            String.format(
                "Index %d is out of bounds for selection vector of length %d", idx, numValues));
      }
      newIndices[i] = selectedIndicesVector.getInt(idx);
    }
    return new CometSelectionVector(originalVector, newIndices);
  }

  // CometVector abstract method implementations

  @Override
  public void setNumNulls(int numNulls) {
    this.numNulls = numNulls;
    this.nullsComputed = true;
  }

  @Override
  public void setNumValues(int numValues) {
    throw new UnsupportedOperationException("Cannot modify number of values in a selection vector");
  }

  @Override
  public int numValues() {
    return numValues;
  }

  @Override
  public ValueVector getValueVector() {
    return originalVector.getValueVector();
  }

  @Override
  public CometVector slice(int offset, int length) {
    if (offset < 0 || length < 0 || offset + length > numValues) {
      throw new IllegalArgumentException("Invalid slice parameters");
    }
    int[] slicedIndices = new int[length];
    for (int i = 0; i < length; i++) {
      slicedIndices[i] = selectedIndicesVector.getInt(offset + i);
    }
    return new CometSelectionVector(originalVector, slicedIndices);
  }

  // ColumnVector method implementations - delegate to original vector with index mapping

  @Override
  public boolean hasNull() {
    if (!nullsComputed) {
      computeNulls();
    }
    return numNulls > 0;
  }

  @Override
  public int numNulls() {
    if (!nullsComputed) {
      computeNulls();
    }
    return numNulls;
  }

  @Override
  public boolean isNullAt(int rowId) {
    return originalVector.isNullAt(getOriginalIndex(rowId));
  }

  @Override
  public boolean getBoolean(int rowId) {
    return originalVector.getBoolean(getOriginalIndex(rowId));
  }

  @Override
  public byte getByte(int rowId) {
    return originalVector.getByte(getOriginalIndex(rowId));
  }

  @Override
  public short getShort(int rowId) {
    return originalVector.getShort(getOriginalIndex(rowId));
  }

  @Override
  public int getInt(int rowId) {
    return originalVector.getInt(getOriginalIndex(rowId));
  }

  @Override
  public long getLong(int rowId) {
    return originalVector.getLong(getOriginalIndex(rowId));
  }

  @Override
  public long getLongDecimal(int rowId) {
    return originalVector.getLongDecimal(getOriginalIndex(rowId));
  }

  @Override
  public float getFloat(int rowId) {
    return originalVector.getFloat(getOriginalIndex(rowId));
  }

  @Override
  public double getDouble(int rowId) {
    return originalVector.getDouble(getOriginalIndex(rowId));
  }

  @Override
  public UTF8String getUTF8String(int rowId) {
    return originalVector.getUTF8String(getOriginalIndex(rowId));
  }

  @Override
  public byte[] getBinary(int rowId) {
    return originalVector.getBinary(getOriginalIndex(rowId));
  }

  @Override
  public ColumnarArray getArray(int rowId) {
    return originalVector.getArray(getOriginalIndex(rowId));
  }

  @Override
  public ColumnarMap getMap(int rowId) {
    return originalVector.getMap(getOriginalIndex(rowId));
  }

  @Override
  public ColumnVector getChild(int ordinal) {
    return originalVector.getChild(ordinal);
  }

  @Override
  public DictionaryProvider getDictionaryProvider() {
    return originalVector.getDictionaryProvider();
  }

  @Override
  public void close() {
    // Selection vectors don't own the original vector, so we don't close it
    // The original vector should be closed by its owner
  }

  /**
   * Exports this selection vector to native side via Arrow FFI. The selection vector is exported as
   * a struct with two fields: the original data vector and the selection indices array. This allows
   * the native side to apply the selection logic efficiently.
   *
   * @param arrayAddr Memory address of the ArrowArray struct to write to
   * @param schemaAddr Memory address of the ArrowSchema struct to write to
   * @throws Exception if export fails
   */
  public void exportToNative(long arrayAddr, long schemaAddr) throws Exception {
    BufferAllocator allocator = originalVector.getValueVector().getAllocator();
    ArrowArray array = null;
    ArrowSchema schema = null;
    StructVector structVector = null;

    try {
      // Use the existing selectedIndicesVector instead of creating a new one
      ValueVector indicesValueVector = selectedIndicesVector.getValueVector();

      // Create struct vector with original data and selection indices
      List<Field> fields =
          Arrays.asList(
              new Field(
                  "original_data", originalVector.getValueVector().getField().getFieldType(), null),
              new Field("selection_indices", indicesValueVector.getField().getFieldType(), null));

      // Create field for the struct vector
      Field structField =
          new Field(
              "selection_vector", new FieldType(false, ArrowType.Struct.INSTANCE, null), fields);

      // Create struct vector using the proper field
      structVector = new StructVector(structField, allocator, null);
      structVector.initializeChildrenFromFields(fields);
      structVector.setValueCount(numValues); // Set value count to match selection size

      // Get child vectors and transfer data
      ValueVector originalDataChild = structVector.getChild("original_data");
      ValueVector selectionIndicesChild = structVector.getChild("selection_indices");

      // Transfer the original vector data
      originalVector.getValueVector().makeTransferPair(originalDataChild).transfer();
      // Transfer the selection indices
      indicesValueVector.makeTransferPair(selectionIndicesChild).transfer();

      // Export via Arrow FFI
      array = ArrowArray.allocateNew(allocator);
      schema = ArrowSchema.allocateNew(allocator);

      Data.exportVector(allocator, structVector, null, array, schema);

      // Copy to target memory addresses using platform-specific memory operations
      Platform.copyMemory(null, array.memoryAddress(), null, arrayAddr, getArrowArraySize());
      Platform.copyMemory(null, schema.memoryAddress(), null, schemaAddr, getArrowSchemaSize());

    } finally {
      // Clean up resources
      if (array != null) {
        array.close();
      }
      if (schema != null) {
        schema.close();
      }
      if (structVector != null) {
        structVector.close();
      }
    }
  }

  /**
   * Returns the size of ArrowArray struct for memory copy operations. This is platform-dependent
   * but typically 144 bytes on 64-bit systems.
   */
  private static long getArrowArraySize() {
    // ArrowArray struct size - this should match Arrow C data interface
    return 144L; // Typical size for 64-bit systems
  }

  /**
   * Returns the size of ArrowSchema struct for memory copy operations. This is platform-dependent
   * but typically 144 bytes on 64-bit systems.
   */
  private static long getArrowSchemaSize() {
    // ArrowSchema struct size - this should match Arrow C data interface
    return 144L; // Typical size for 64-bit systems
  }

  /** Computes the number of nulls in this selection vector by checking each selected element. */
  private void computeNulls() {
    int nullCount = 0;
    for (int i = 0; i < numValues; i++) {
      int selectedIndex = selectedIndicesVector.getInt(i);
      if (originalVector.isNullAt(selectedIndex)) {
        nullCount++;
      }
    }
    this.numNulls = nullCount;
    this.nullsComputed = true;
  }
}

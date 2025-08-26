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
import org.apache.spark.unsafe.types.UTF8String;

/**
 * A zero-copy selection vector that extends CometStructVector. This implementation creates an
 * underlying Arrow StructVector containing both the original data vector and selection indices,
 * providing efficient access to selected elements without copying the underlying data.
 *
 * <p>The struct contains two fields: - "original_data": The original CometVector data -
 * "selection_indices": An IntVector containing the selection indices
 *
 * <p>For example, if the original vector has values [v0, v1, v2, v3, v4, v5, v6, v7] and the
 * selection indices are [0, 1, 3, 4, 5, 7], then this selection vector will logically represent
 * [v0, v1, v3, v4, v5, v7] without actually copying the data.
 */
public class CometSelectionVector extends CometStructVector {
  /** Number of selected elements */
  private final int numValues;

  /**
   * Creates a new selection vector from the given vector and indices.
   *
   * @param values The original vector to select from
   * @param indices The indices to select from the original vector
   * @throws IllegalArgumentException if any index is out of bounds
   */
  public CometSelectionVector(CometVector values, int[] indices) {
    super(
        createStructVector(values, indices), values.useDecimal128, values.getDictionaryProvider());

    this.numValues = indices.length;

    // Validate indices are within bounds
    int originalLength = values.numValues();
    for (int i = 0; i < indices.length; i++) {
      int idx = indices[i];
      if (idx < 0 || idx >= originalLength) {
        throw new IllegalArgumentException(
            String.format(
                "Index %d is out of bounds for vector of length %d", idx, originalLength));
      }
    }
  }

  /**
   * Creates the underlying StructVector containing original data and selection indices.
   *
   * @param originalVector The original vector to select from
   * @param selectionIndices The indices to select from the original vector
   * @return A StructVector containing both original data and selection indices
   */
  private static StructVector createStructVector(
      CometVector originalVector, int[] selectionIndices) {
    BufferAllocator allocator = originalVector.getValueVector().getAllocator();
    ValueVector originalValueVector = originalVector.getValueVector();

    // Create selection indices vector
    IntVector indicesVector = new IntVector("selection_indices", allocator);
    indicesVector.allocateNew(selectionIndices.length);
    for (int i = 0; i < selectionIndices.length; i++) {
      indicesVector.set(i, selectionIndices[i]);
    }
    indicesVector.setValueCount(selectionIndices.length);

    // Create field definitions for the struct
    List<Field> fields =
        Arrays.asList(
            new Field("sv_values", originalValueVector.getField().getFieldType(), null),
            new Field("sv_indices", indicesVector.getField().getFieldType(), null));

    // Create struct field
    Field structField =
        new Field(
            "comet_selection_vector",
            new FieldType(false, ArrowType.Struct.INSTANCE, null),
            fields);

    // Create struct vector
    StructVector structVector = new StructVector(structField, allocator, null);
    structVector.initializeChildrenFromFields(fields);
    structVector.setValueCount(selectionIndices.length);

    // Transfer data to struct children
    ValueVector originalDataChild = structVector.getChild("sv_values");
    ValueVector selectionIndicesChild = structVector.getChild("sv_indices");

    originalValueVector.makeTransferPair(originalDataChild).transfer();
    indicesVector.makeTransferPair(selectionIndicesChild).transfer();

    // Clean up temporary indices vector
    indicesVector.close();

    return structVector;
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
    // Get the index from the indices vector stored in the struct
    CometVector indicesVector = getIndicesVector();
    return indicesVector.getInt(selectionIndex);
  }

  /**
   * Returns a reference to the values vector from the struct.
   *
   * @return The CometVector containing the values
   */
  public CometVector getValues() {
    // Get the values from the struct vector's sv_values child
    return (CometVector) getChild(0); // sv_values is the first child
  }

  /**
   * Returns the indices vector from the struct.
   *
   * @return The CometVector containing the indices
   */
  private CometVector getIndicesVector() {
    // Get the indices from the struct vector's sv_indices child
    return (CometVector) getChild(1); // sv_indices is the second child
  }

  /**
   * Returns a copy of the selected indices.
   *
   * @return Array of selected indices
   */
  public int[] getSelectedIndices() {
    CometVector indicesVector = getIndicesVector();
    int[] result = new int[numValues];
    for (int i = 0; i < numValues; i++) {
      result[i] = indicesVector.getInt(i);
    }
    return result;
  }

  /**
   * Creates a nested selection by applying additional indices to this selection vector. Returns the
   * same type as this vector.
   *
   * @param indices The indices to select from this selection vector
   * @return A new CometSelectionVector representing the nested selection
   * @throws IllegalArgumentException if any index is out of bounds
   */
  public CometSelectionVector take(int[] indices) {
    int[] newIndices = new int[indices.length];
    CometVector indicesVector = getIndicesVector();
    for (int i = 0; i < indices.length; i++) {
      int idx = indices[i];
      if (idx < 0 || idx >= numValues) {
        throw new IllegalArgumentException(
            String.format(
                "Index %d is out of bounds for selection vector of length %d", idx, numValues));
      }
      newIndices[i] = indicesVector.getInt(idx);
    }
    return new CometSelectionVector(getValues(), newIndices);
  }

  @Override
  public int numValues() {
    return numValues;
  }

  // ColumnVector method implementations - delegate to original vector with index mapping

  @Override
  public boolean isNullAt(int rowId) {
    return getValues().isNullAt(getOriginalIndex(rowId));
  }

  @Override
  public boolean getBoolean(int rowId) {
    return getValues().getBoolean(getOriginalIndex(rowId));
  }

  @Override
  public byte getByte(int rowId) {
    return getValues().getByte(getOriginalIndex(rowId));
  }

  @Override
  public short getShort(int rowId) {
    return getValues().getShort(getOriginalIndex(rowId));
  }

  @Override
  public int getInt(int rowId) {
    return getValues().getInt(getOriginalIndex(rowId));
  }

  @Override
  public long getLong(int rowId) {
    return getValues().getLong(getOriginalIndex(rowId));
  }

  @Override
  public long getLongDecimal(int rowId) {
    return getValues().getLongDecimal(getOriginalIndex(rowId));
  }

  @Override
  public float getFloat(int rowId) {
    return getValues().getFloat(getOriginalIndex(rowId));
  }

  @Override
  public double getDouble(int rowId) {
    return getValues().getDouble(getOriginalIndex(rowId));
  }

  @Override
  public UTF8String getUTF8String(int rowId) {
    return getValues().getUTF8String(getOriginalIndex(rowId));
  }

  @Override
  public byte[] getBinary(int rowId) {
    return getValues().getBinary(getOriginalIndex(rowId));
  }

  @Override
  public ColumnarArray getArray(int rowId) {
    return getValues().getArray(getOriginalIndex(rowId));
  }

  @Override
  public ColumnarMap getMap(int rowId) {
    return getValues().getMap(getOriginalIndex(rowId));
  }

  @Override
  public ColumnVector getChild(int ordinal) {
    // Return the child from the original vector with selection applied
    return getValues().getChild(ordinal);
  }

  @Override
  public DictionaryProvider getDictionaryProvider() {
    return getValues().getDictionaryProvider();
  }

  @Override
  public CometVector slice(int offset, int length) {
    if (offset < 0 || length < 0 || offset + length > numValues) {
      throw new IllegalArgumentException("Invalid slice parameters");
    }
    // Get the current indices and slice them
    int[] currentIndices = getSelectedIndices();
    int[] slicedIndices = new int[length];
    System.arraycopy(currentIndices, offset, slicedIndices, 0, length);
    return new CometSelectionVector(getValues(), slicedIndices);
  }

  @Override
  public void close() {
    // Close the underlying struct vector
    super.close();
    // Note: We don't close the original vector as it may be owned by someone else
  }
}

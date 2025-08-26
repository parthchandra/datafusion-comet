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

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;

import static org.junit.Assert.*;

public class CometSelectionVectorTest {

  private RootAllocator allocator;

  @Before
  public void setUp() {
    allocator = new RootAllocator(1024 * 1024);
  }

  @After
  public void tearDown() {
    if (allocator != null) {
      allocator.close();
    }
  }

  @Test
  public void testIntVectorSelection() {
    // Create a vector with values [10, 20, 30, 40, 50, 60, 70, 80]
    IntVector intVector = new IntVector("test", allocator);
    intVector.allocateNew(8);
    for (int i = 0; i < 8; i++) {
      intVector.set(i, (i + 1) * 10);
    }
    intVector.setValueCount(8);

    CometPlainVector originalVector = new CometPlainVector(intVector, false);

    // Select indices [0, 1, 3, 4, 5, 7] which should give us [10, 20, 40, 50, 60, 80]
    int[] indices = {0, 1, 3, 4, 5, 7};
    CometSelectionVector selectionVector = new CometSelectionVector(originalVector, indices);

    // Test basic properties
    assertEquals(6, selectionVector.numValues());
    assertFalse(selectionVector.hasNull());
    assertEquals(0, selectionVector.numNulls());

    // Test value access
    assertEquals(10, selectionVector.getInt(0));
    assertEquals(20, selectionVector.getInt(1));
    assertEquals(40, selectionVector.getInt(2));
    assertEquals(50, selectionVector.getInt(3));
    assertEquals(60, selectionVector.getInt(4));
    assertEquals(80, selectionVector.getInt(5));

    // Test index mapping
    assertEquals(0, selectionVector.getOriginalIndex(0));
    assertEquals(1, selectionVector.getOriginalIndex(1));
    assertEquals(3, selectionVector.getOriginalIndex(2));
    assertEquals(4, selectionVector.getOriginalIndex(3));
    assertEquals(5, selectionVector.getOriginalIndex(4));
    assertEquals(7, selectionVector.getOriginalIndex(5));

    // Test accessing the original vector reference
    assertEquals(originalVector, selectionVector.getOriginalVector());

    // Test getting selected indices
    assertArrayEquals(indices, selectionVector.getSelectedIndices());

    originalVector.close();
  }

  @Test
  public void testStringVectorSelection() {
    // Create a vector with string values ["a", "b", "c", "d", "e", "f", "g", "h"]
    VarCharVector varCharVector = new VarCharVector("test", allocator);
    varCharVector.allocateNew(64, 8);
    String[] values = {"a", "b", "c", "d", "e", "f", "g", "h"};
    for (int i = 0; i < values.length; i++) {
      varCharVector.setSafe(i, values[i].getBytes());
    }
    varCharVector.setValueCount(8);

    CometPlainVector originalVector = new CometPlainVector(varCharVector, false);

    // Select indices [0, 2, 4, 6] which should give us ["a", "c", "e", "g"]
    int[] indices = {0, 2, 4, 6};
    CometSelectionVector selectionVector = new CometSelectionVector(originalVector, indices);

    assertEquals(4, selectionVector.numValues());

    // Test string value access
    assertEquals("a", selectionVector.getUTF8String(0).toString());
    assertEquals("c", selectionVector.getUTF8String(1).toString());
    assertEquals("e", selectionVector.getUTF8String(2).toString());
    assertEquals("g", selectionVector.getUTF8String(3).toString());

    originalVector.close();
  }

  @Test
  public void testNestedSelection() {
    // Create a vector with values [10, 20, 30, 40, 50, 60, 70, 80]
    IntVector intVector = new IntVector("test", allocator);
    intVector.allocateNew(8);
    for (int i = 0; i < 8; i++) {
      intVector.set(i, (i + 1) * 10);
    }
    intVector.setValueCount(8);

    CometPlainVector originalVector = new CometPlainVector(intVector, false);

    // First selection: [0, 1, 3, 4, 5, 7] -> [10, 20, 40, 50, 60, 80]
    int[] firstIndices = {0, 1, 3, 4, 5, 7};
    CometSelectionVector firstSelection = new CometSelectionVector(originalVector, firstIndices);

    // Second selection from the first: [1, 2, 4] -> should get elements at positions 1, 3, 5
    // from original which are [20, 40, 60]
    int[] secondIndices = {1, 2, 4};
    CometSelectionVector secondSelection = firstSelection.take(secondIndices);

    assertEquals(3, secondSelection.numValues());
    assertEquals(20, secondSelection.getInt(0));
    assertEquals(40, secondSelection.getInt(1));
    assertEquals(60, secondSelection.getInt(2));

    // Verify the original indices are correct
    assertEquals(1, secondSelection.getOriginalIndex(0)); // maps to original[1] = 20
    assertEquals(3, secondSelection.getOriginalIndex(1)); // maps to original[3] = 40
    assertEquals(5, secondSelection.getOriginalIndex(2)); // maps to original[5] = 60

    originalVector.close();
  }

  @Test
  public void testTakeMethodOnCometVector() {
    // Create a vector with values [10, 20, 30, 40, 50, 60, 70, 80]
    IntVector intVector = new IntVector("test", allocator);
    intVector.allocateNew(8);
    for (int i = 0; i < 8; i++) {
      intVector.set(i, (i + 1) * 10);
    }
    intVector.setValueCount(8);

    CometPlainVector originalVector = new CometPlainVector(intVector, false);

    // Use the take method from CometVector
    int[] indices = {0, 1, 3, 4, 5, 7};
    CometSelectionVector selectionVector = originalVector.take(indices);

    assertEquals(6, selectionVector.numValues());
    assertEquals(10, selectionVector.getInt(0));
    assertEquals(20, selectionVector.getInt(1));
    assertEquals(40, selectionVector.getInt(2));
    assertEquals(50, selectionVector.getInt(3));
    assertEquals(60, selectionVector.getInt(4));
    assertEquals(80, selectionVector.getInt(5));

    originalVector.close();
  }

  @Test
  public void testSliceOperation() {
    // Create a vector with values [10, 20, 30, 40, 50, 60, 70, 80]
    IntVector intVector = new IntVector("test", allocator);
    intVector.allocateNew(8);
    for (int i = 0; i < 8; i++) {
      intVector.set(i, (i + 1) * 10);
    }
    intVector.setValueCount(8);

    CometPlainVector originalVector = new CometPlainVector(intVector, false);

    // Create selection [0, 1, 3, 4, 5, 7] -> [10, 20, 40, 50, 60, 80]
    int[] indices = {0, 1, 3, 4, 5, 7};
    CometSelectionVector selectionVector = new CometSelectionVector(originalVector, indices);

    // Slice from offset 1 with length 3 -> should get [20, 40, 50]
    CometVector sliced = selectionVector.slice(1, 3);
    assertEquals(3, sliced.numValues());
    assertEquals(20, sliced.getInt(0));
    assertEquals(40, sliced.getInt(1));
    assertEquals(50, sliced.getInt(2));

    originalVector.close();
    sliced.close();
  }

  @Test
  public void testEmptySelection() {
    IntVector intVector = new IntVector("test", allocator);
    intVector.allocateNew(5);
    for (int i = 0; i < 5; i++) {
      intVector.set(i, i + 1);
    }
    intVector.setValueCount(5);

    CometPlainVector originalVector = new CometPlainVector(intVector, false);

    // Empty selection
    int[] indices = {};
    CometSelectionVector selectionVector = new CometSelectionVector(originalVector, indices);

    assertEquals(0, selectionVector.numValues());
    assertEquals(0, selectionVector.numNulls());
    assertFalse(selectionVector.hasNull());

    originalVector.close();
  }

  @Test
  public void testWithNulls() {
    IntVector intVector = new IntVector("test", allocator);
    intVector.allocateNew(5);
    intVector.set(0, 10);
    intVector.setNull(1);
    intVector.set(2, 30);
    intVector.setNull(3);
    intVector.set(4, 50);
    intVector.setValueCount(5);

    CometPlainVector originalVector = new CometPlainVector(intVector, false);

    // Select indices that include both nulls and non-nulls
    int[] indices = {0, 1, 2, 3}; // [10, null, 30, null]
    CometSelectionVector selectionVector = new CometSelectionVector(originalVector, indices);

    assertEquals(4, selectionVector.numValues());
    assertTrue(selectionVector.hasNull());
    assertEquals(2, selectionVector.numNulls());

    // Test null checks
    assertFalse(selectionVector.isNullAt(0)); // 10
    assertTrue(selectionVector.isNullAt(1)); // null
    assertFalse(selectionVector.isNullAt(2)); // 30
    assertTrue(selectionVector.isNullAt(3)); // null

    // Test value access
    assertEquals(10, selectionVector.getInt(0));
    assertEquals(30, selectionVector.getInt(2));

    originalVector.close();
  }

  @Test(expected = IllegalArgumentException.class)
  public void testOutOfBoundsIndex() {
    IntVector intVector = new IntVector("test", allocator);
    intVector.allocateNew(5);
    for (int i = 0; i < 5; i++) {
      intVector.set(i, i + 1);
    }
    intVector.setValueCount(5);

    CometPlainVector originalVector = new CometPlainVector(intVector, false);

    // Index 5 is out of bounds for a vector of length 5
    int[] indices = {0, 1, 5};
    new CometSelectionVector(originalVector, indices);

    originalVector.close();
  }

  @Test(expected = IndexOutOfBoundsException.class)
  public void testInvalidSelectionIndex() {
    IntVector intVector = new IntVector("test", allocator);
    intVector.allocateNew(5);
    for (int i = 0; i < 5; i++) {
      intVector.set(i, i + 1);
    }
    intVector.setValueCount(5);

    CometPlainVector originalVector = new CometPlainVector(intVector, false);

    int[] indices = {0, 1, 2};
    CometSelectionVector selectionVector = new CometSelectionVector(originalVector, indices);

    // Accessing index 3 in a selection vector of length 3 should fail
    selectionVector.getOriginalIndex(3);

    originalVector.close();
  }

  @Test(expected = IllegalArgumentException.class)
  public void testNestedSelectionOutOfBounds() {
    IntVector intVector = new IntVector("test", allocator);
    intVector.allocateNew(5);
    for (int i = 0; i < 5; i++) {
      intVector.set(i, i + 1);
    }
    intVector.setValueCount(5);

    CometPlainVector originalVector = new CometPlainVector(intVector, false);

    int[] firstIndices = {0, 1, 2};
    CometSelectionVector firstSelection = new CometSelectionVector(originalVector, firstIndices);

    // Index 3 is out of bounds for the first selection vector of length 3
    int[] secondIndices = {0, 1, 3};
    firstSelection.take(secondIndices);

    originalVector.close();
  }
}

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

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VarCharVector;

/**
 * Example demonstrating how to use CometSelectionVectorV2 which extends CometStructVector. This
 * shows how the new implementation creates an underlying StructVector containing both original data
 * and selection indices.
 */
public class CometSelectionVectorV2Example {

  public static void main(String[] args) throws Exception {
    // Set up allocator
    try (BufferAllocator allocator = new RootAllocator()) {

      // Create sample data
      VarCharVector dataVector = new VarCharVector("sample_data", allocator);
      dataVector.allocateNew(1024, 8);
      String[] sampleData = {"Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry"};

      for (int i = 0; i < sampleData.length; i++) {
        dataVector.setSafe(i, sampleData[i].getBytes());
      }
      dataVector.setValueCount(sampleData.length);

      // Wrap in CometVector
      CometVector cometVector = new CometPlainVector(dataVector, false);

      // Create selection vector V2 - select every other element starting from index 1
      int[] selectionIndices = {1, 3, 5, 7}; // Bob, David, Frank, Henry
      CometSelectionVectorV2 selectionVectorV2 =
          new CometSelectionVectorV2(cometVector, selectionIndices);

      System.out.println(
          "Created CometSelectionVectorV2 with " + selectionVectorV2.numValues() + " elements:");
      for (int i = 0; i < selectionVectorV2.numValues(); i++) {
        System.out.println("  [" + i + "] = " + selectionVectorV2.getUTF8String(i).toString());
      }

      System.out.println("\nUnderylng struct vector details:");
      System.out.println("  Data type: " + selectionVectorV2.dataType());
      System.out.println(
          "  Is struct vector: " + selectionVectorV2.getValueVector().getClass().getSimpleName());
      System.out.println("  Number of children: " + selectionVectorV2.children.size());

      // Access struct children
      System.out.println("\nStruct children:");
      for (int i = 0; i < selectionVectorV2.children.size(); i++) {
        System.out.println(
            "  Child " + i + ": " + selectionVectorV2.getChild(i).getClass().getSimpleName());
      }

      // Test take operation (returns CometSelectionVector for compatibility)
      int[] takeIndices = {0, 2}; // Take first and third elements from selection
      CometSelectionVector taken = selectionVectorV2.take(takeIndices);
      System.out.println("\nAfter take([0, 2]):");
      for (int i = 0; i < taken.numValues(); i++) {
        System.out.println("  [" + i + "] = " + taken.getUTF8String(i).toString());
      }

      // Test takeV2 operation (returns CometSelectionVectorV2)
      CometSelectionVectorV2 takenV2 = selectionVectorV2.takeV2(takeIndices);
      System.out.println("\nAfter takeV2([0, 2]):");
      for (int i = 0; i < takenV2.numValues(); i++) {
        System.out.println("  [" + i + "] = " + takenV2.getUTF8String(i).toString());
      }

      // Test slice operation
      CometVector sliced = selectionVectorV2.slice(1, 2);
      System.out.println("\nAfter slice(1, 2):");
      for (int i = 0; i < sliced.numValues(); i++) {
        System.out.println("  [" + i + "] = " + sliced.getUTF8String(i).toString());
      }

      System.out.println("\n✓ CometSelectionVectorV2 example completed successfully");

      // Clean up
      selectionVectorV2.close();
      takenV2.close();
      taken.close();
      sliced.close();
      dataVector.close();
    }
  }
}

/*
Expected output:
Created CometSelectionVectorV2 with 4 elements:
  [0] = Bob
  [1] = David
  [2] = Frank
  [3] = Henry

Underylng struct vector details:
  Data type: struct<original_data:utf8,selection_indices:int32>
  Is struct vector: StructVector
  Number of children: 2

Struct children:
  Child 0: CometPlainVector
  Child 1: CometPlainVector

After take([0, 2]):
  [0] = Bob
  [1] = Frank

After takeV2([0, 2]):
  [0] = Bob
  [1] = Frank

After slice(1, 2):
  [0] = David
  [1] = Frank

✓ CometSelectionVectorV2 example completed successfully
*/

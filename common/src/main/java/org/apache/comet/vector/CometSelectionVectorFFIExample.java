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

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VarCharVector;

import org.apache.comet.parquet.Native;

/**
 * Example demonstrating how to use CometSelectionVector FFI export functionality. This shows how to
 * export a selection vector to the native side for processing.
 */
public class CometSelectionVectorFFIExample {

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

      // Create selection vector - select every other element starting from index 1
      int[] selectionIndices = {1, 3, 5, 7}; // Bob, David, Frank, Henry
      CometSelectionVector selectionVector =
          new CometSelectionVector(cometVector, selectionIndices);

      System.out.println(
          "Created selection vector with " + selectionVector.numValues() + " elements:");
      for (int i = 0; i < selectionVector.numValues(); i++) {
        System.out.println("  [" + i + "] = " + selectionVector.getUTF8String(i).toString());
      }

      // Export to native side via Arrow FFI
      ArrowArray array = ArrowArray.allocateNew(allocator);
      ArrowSchema schema = ArrowSchema.allocateNew(allocator);

      try {
        System.out.println("\nExporting selection vector to native side...");

        // Method 1: Direct export via CometSelectionVector
        selectionVector.exportToNative(array.memoryAddress(), schema.memoryAddress());
        System.out.println("✓ Direct export successful");

        // Method 2: Export via Native utility method (alternative approach)
        ArrowArray array2 = ArrowArray.allocateNew(allocator);
        ArrowSchema schema2 = ArrowSchema.allocateNew(allocator);

        try {
          Native.exportSelectionVector(
              selectionVector, array2.memoryAddress(), schema2.memoryAddress());
          System.out.println("✓ Native utility export successful");
        } finally {
          array2.close();
          schema2.close();
        }

        System.out.println("\nFFI Export Details:");
        System.out.println("  Array memory address: 0x" + Long.toHexString(array.memoryAddress()));
        System.out.println(
            "  Schema memory address: 0x" + Long.toHexString(schema.memoryAddress()));

        // At this point, the native side can import the selection vector using:
        // CometSelectionVector::from_ffi(array_ptr, schema_ptr)
        // and then apply the selection using:
        // selection_vector.apply_selection()

        System.out.println("\n✓ Selection vector successfully exported via Arrow FFI");
        System.out.println("  Native side can now import and process the selection vector");

      } finally {
        array.close();
        schema.close();
      }

      // Clean up
      selectionVector.close();
      dataVector.close();
    }
  }
}

/*
Expected output:
Created selection vector with 4 elements:
  [0] = Bob
  [1] = David
  [2] = Frank
  [3] = Henry

Exporting selection vector to native side...
✓ Direct export successful
✓ Native utility export successful

FFI Export Details:
  Array memory address: 0x7f8b2c001000
  Schema memory address: 0x7f8b2c001200

✓ Selection vector successfully exported via Arrow FFI
  Native side can now import and process the selection vector
*/

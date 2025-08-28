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

package org.apache.comet.parquet;

/**
 * Example demonstrating how to use the Java array export functionality to pass arrays to native
 * Rust code via JNI. This shows proper memory management patterns and safe usage of the export/free
 * APIs.
 */
public class JavaArrayExportExample {

  /** Demonstrates basic array export usage with proper resource management. */
  public static void basicArrayExportExample() {
    System.out.println("=== Basic Array Export Example ===");

    // Sample data arrays
    int[] intData = {1, 2, 3, 4, 5};
    long[] longData = {1000L, 2000L, 3000L};
    double[] doubleData = {1.5, 2.7, 3.14159};
    String[] stringData = {"Hello", "from", "Java", "to", "Rust"};

    // Export arrays to native Rust memory
    long intPtr = Native.exportIntArray(intData);
    long longPtr = Native.exportLongArray(longData);
    long doublePtr = Native.exportDoubleArray(doubleData);
    long stringPtr = Native.exportStringArray(stringData);

    try {
      System.out.println("Exported arrays to native memory:");
      System.out.println("  Int array pointer: 0x" + Long.toHexString(intPtr));
      System.out.println("  Long array pointer: 0x" + Long.toHexString(longPtr));
      System.out.println("  Double array pointer: 0x" + Long.toHexString(doublePtr));
      System.out.println("  String array pointer: 0x" + Long.toHexString(stringPtr));

      // At this point, the native Rust code can access the arrays via these pointers
      // For example, native processing functions could be called here:
      // processIntArray(intPtr, intData.length);
      // processDoubleArray(doublePtr, doubleData.length);

    } finally {
      // IMPORTANT: Always free exported arrays to prevent memory leaks
      Native.freeExportedIntArray(intPtr);
      Native.freeExportedLongArray(longPtr);
      Native.freeExportedDoubleArray(doublePtr);
      Native.freeExportedStringArray(stringPtr);

      System.out.println("✓ All exported arrays have been freed");
    }
  }

  /** Demonstrates safe array export with automatic resource management. */
  public static void safeArrayExportExample() {
    System.out.println("\n=== Safe Array Export with Resource Management ===");

    // Example of a utility class for safe array export
    try (ArrayExportManager manager = new ArrayExportManager()) {
      int[] data = {10, 20, 30, 40, 50};
      long ptr = manager.exportIntArray(data);

      System.out.println("Safely exported int array to: 0x" + Long.toHexString(ptr));

      // Process the array in native code
      // nativeProcessingFunction(ptr, data.length);

      // Automatic cleanup happens when the manager is closed
    }
    System.out.println("✓ Automatic cleanup completed");
  }

  /** Demonstrates handling of edge cases and special values. */
  public static void edgeCasesExample() {
    System.out.println("\n=== Edge Cases Example ===");

    // Empty arrays
    int[] emptyInts = {};
    String[] emptyStrings = {};

    long emptyIntPtr = Native.exportIntArray(emptyInts);
    long emptyStringPtr = Native.exportStringArray(emptyStrings);

    System.out.println("Empty arrays exported successfully:");
    System.out.println("  Empty int array: 0x" + Long.toHexString(emptyIntPtr));
    System.out.println("  Empty string array: 0x" + Long.toHexString(emptyStringPtr));

    // Arrays with special values
    double[] specialDoubles = {
      Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 0.0, -0.0
    };
    String[] specialStrings = {"", null, "Unicode: 🦀", "Special chars: \n\t\r"};

    long specialDoublePtr = Native.exportDoubleArray(specialDoubles);
    long specialStringPtr = Native.exportStringArray(specialStrings);

    try {
      System.out.println("Special value arrays exported:");
      System.out.println("  Special doubles: 0x" + Long.toHexString(specialDoublePtr));
      System.out.println("  Special strings: 0x" + Long.toHexString(specialStringPtr));

    } finally {
      // Clean up all exported arrays
      Native.freeExportedIntArray(emptyIntPtr);
      Native.freeExportedStringArray(emptyStringPtr);
      Native.freeExportedDoubleArray(specialDoublePtr);
      Native.freeExportedStringArray(specialStringPtr);
    }

    System.out.println("✓ Edge cases handled successfully");
  }

  /** Demonstrates performance considerations for large arrays. */
  public static void performanceExample() {
    System.out.println("\n=== Performance Example ===");

    int size = 1000000; // 1 million elements

    long startTime = System.nanoTime();

    // Create large array
    int[] largeArray = new int[size];
    for (int i = 0; i < size; i++) {
      largeArray[i] = i;
    }

    // Export to native
    long ptr = Native.exportIntArray(largeArray);

    long exportTime = System.nanoTime();

    // Free the array
    Native.freeExportedIntArray(ptr);

    long freeTime = System.nanoTime();

    System.out.printf("Performance metrics for %d elements:\n", size);
    System.out.printf("  Array creation: %.2f ms\n", (exportTime - startTime) / 1_000_000.0);
    System.out.printf("  Array export: %.2f ms\n", (freeTime - exportTime) / 1_000_000.0);
    System.out.printf("  Total time: %.2f ms\n", (freeTime - startTime) / 1_000_000.0);
  }

  public static void main(String[] args) {
    try {
      basicArrayExportExample();
      safeArrayExportExample();
      edgeCasesExample();
      performanceExample();

      System.out.println("\n✅ All examples completed successfully!");

    } catch (Exception e) {
      System.err.println("❌ Example failed: " + e.getMessage());
      e.printStackTrace();
    }
  }

  /**
   * Utility class for managing array exports with automatic cleanup. Implements AutoCloseable for
   * use with try-with-resources.
   */
  static class ArrayExportManager implements AutoCloseable {
    private java.util.List<Long> exportedPointers = new java.util.ArrayList<>();
    private java.util.List<String> pointerTypes = new java.util.ArrayList<>();

    public long exportIntArray(int[] array) {
      long ptr = Native.exportIntArray(array);
      exportedPointers.add(ptr);
      pointerTypes.add("int");
      return ptr;
    }

    public long exportLongArray(long[] array) {
      long ptr = Native.exportLongArray(array);
      exportedPointers.add(ptr);
      pointerTypes.add("long");
      return ptr;
    }

    public long exportDoubleArray(double[] array) {
      long ptr = Native.exportDoubleArray(array);
      exportedPointers.add(ptr);
      pointerTypes.add("double");
      return ptr;
    }

    public long exportStringArray(String[] array) {
      long ptr = Native.exportStringArray(array);
      exportedPointers.add(ptr);
      pointerTypes.add("string");
      return ptr;
    }

    @Override
    public void close() {
      for (int i = 0; i < exportedPointers.size(); i++) {
        long ptr = exportedPointers.get(i);
        String type = pointerTypes.get(i);

        switch (type) {
          case "int":
            Native.freeExportedIntArray(ptr);
            break;
          case "long":
            Native.freeExportedLongArray(ptr);
            break;
          case "double":
            Native.freeExportedDoubleArray(ptr);
            break;
          case "string":
            Native.freeExportedStringArray(ptr);
            break;
        }
      }

      System.out.println("Freed " + exportedPointers.size() + " exported arrays");
      exportedPointers.clear();
      pointerTypes.clear();
    }
  }
}

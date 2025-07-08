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

package org.apache.comet.parquet; /*
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

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.permission.FsPermission;

/**
 * JNI bridge for HDFS operations using Hadoop DistributedFileSystem. This class provides the Java
 * side implementation that will be called from Rust via JNI.
 */
public class HdfsBridge {
  private final FileSystem fs;
  private final Configuration conf;

  /**
   * Constructor that initializes the HDFS filesystem connection.
   *
   * @param hdfsUri HDFS URI (e.g., "hdfs://localhost:8020")
   * @param conf Hadoop Configuration object (can be null for default)
   * @throws IOException if filesystem initialization fails
   */
  public HdfsBridge(String hdfsUri, Configuration conf) throws IOException {
    this.conf = conf != null ? conf : new Configuration();
    this.fs = FileSystem.get(URI.create(hdfsUri), this.conf);
  }

  /**
   * Read a file from HDFS and return its contents as a byte array.
   *
   * @param path HDFS path to read
   * @return byte array containing file contents
   * @throws IOException if read operation fails
   */
  public byte[] readFile(String path) throws IOException {
    Path hdfsPath = new Path(path);
    try (FSDataInputStream in = fs.open(hdfsPath)) {
      return IOUtils.toByteArray(in);
    }
  }

  /**
   * Write data to a file in HDFS.
   *
   * @param path HDFS path to write to
   * @param data byte array containing data to write
   * @throws IOException if write operation fails
   */
  public void writeFile(String path, byte[] data) throws IOException {
    Path hdfsPath = new Path(path);
    try (FSDataOutputStream out = fs.create(hdfsPath, true)) {
      out.write(data);
      out.flush();
    }
  }

  /**
   * Delete a file from HDFS.
   *
   * @param path HDFS path to delete
   * @throws IOException if delete operation fails
   */
  public void deleteFile(String path) throws IOException {
    Path hdfsPath = new Path(path);
    if (fs.exists(hdfsPath)) {
      fs.delete(hdfsPath, false);
    }
  }

  /**
   * Get metadata for a file in HDFS.
   *
   * @param path HDFS path
   * @return FileMetadata object containing file information
   * @throws IOException if metadata retrieval fails
   */
  public FileMetadata getFileMetadata(String path) throws IOException {
    Path hdfsPath = new Path(path);
    FileStatus status = fs.getFileStatus(hdfsPath);
    return new FileMetadata(path, status.getLen(), status.getModificationTime());
  }

  /**
   * List files in a directory with optional prefix filtering.
   *
   * @param prefix directory prefix to list (can be empty for root)
   * @return array of FileMetadata objects
   * @throws IOException if listing operation fails
   */
  public FileMetadata[] listFiles(String prefix) throws IOException {
    Path prefixPath = prefix.isEmpty() ? new Path("/") : new Path(prefix);
    List<FileMetadata> files = new ArrayList<>();

    if (fs.exists(prefixPath)) {
      FileStatus[] statuses = fs.listStatus(prefixPath);
      for (FileStatus status : statuses) {
        if (!status.isDirectory()) {
          files.add(
              new FileMetadata(
                  status.getPath().toString(), status.getLen(), status.getModificationTime()));
        }
      }
    }

    return files.toArray(new FileMetadata[0]);
  }

  /**
   * Check if a file exists in HDFS.
   *
   * @param path HDFS path to check
   * @return true if file exists, false otherwise
   * @throws IOException if check operation fails
   */
  public boolean fileExists(String path) throws IOException {
    Path hdfsPath = new Path(path);
    return fs.exists(hdfsPath);
  }

  /**
   * Create a directory in HDFS.
   *
   * @param path HDFS path to create
   * @throws IOException if directory creation fails
   */
  public void createDirectory(String path) throws IOException {
    Path hdfsPath = new Path(path);
    if (!fs.exists(hdfsPath)) {
      fs.mkdirs(hdfsPath, FsPermission.valueOf("-rwxrwxrwx"));
    }
  }

  /**
   * Copy a file within HDFS.
   *
   * @param from source path
   * @param to destination path
   * @throws IOException if copy operation fails
   */
  public void copyFile(String from, String to) throws IOException {
    Path fromPath = new Path(from);
    Path toPath = new Path(to);
    FileUtil.copy(fs, fromPath, fs, toPath, false, conf);
  }

  /**
   * Get the filesystem instance.
   *
   * @return FileSystem instance
   */
  public FileSystem getFileSystem() {
    return fs;
  }

  /**
   * Close the filesystem connection.
   *
   * @throws IOException if close operation fails
   */
  public void close() throws IOException {
    if (fs != null) {
      fs.close();
    }
  }

  /** Inner class to hold file metadata information. */
  public static class FileMetadata {
    private final String path;
    private final long size;
    private final long lastModified;

    public FileMetadata(String path, long size, long lastModified) {
      this.path = path;
      this.size = size;
      this.lastModified = lastModified;
    }

    public String getPath() {
      return path;
    }

    public long getSize() {
      return size;
    }

    public long getLastModified() {
      return lastModified;
    }

    @Override
    public String toString() {
      return String.format(
          "FileMetadata{path='%s', size=%d, lastModified=%d}", path, size, lastModified);
    }
  }
}

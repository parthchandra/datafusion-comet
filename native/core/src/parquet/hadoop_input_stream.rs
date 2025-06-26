// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.


use jni::{
    objects::{JObject, JValue, JByteArray},
    sys::{jlong, jint, jbyteArray},
    JNIEnv,
};

// Function to call the seek method of a SeekableInputStream
pub fn call_seek_on_seekable_input_stream<'local>(
    env: &mut JNIEnv<'local>,
    seekable_stream_obj: JObject<'local>,
    position: jlong,
) -> Result<(), jni::errors::Error> {
    // 1. Get the class of the SeekableInputStream object
    let seekable_input_stream_class = env.get_object_class(seekable_stream_obj)?;

    // 2. Get the method ID for the "seek" method (assuming void seek(long))
    let seek_method_id = env.get_method_id(seekable_input_stream_class, "seek", "(J)V")?;

    // 3. Call the "seek" method
    env.call_method(
        seekable_stream_obj,
        seek_method_id,
        &[JValue::Long(position)],
    )?;

    Ok(())
}

// Function to call the getPos method of a SeekableInputStream
pub fn call_get_pos_on_seekable_input_stream<'local>(
    env: &mut JNIEnv<'local>,
    seekable_stream_obj: JObject<'local>,
) -> Result<jlong, jni::errors::Error> {
    // 1. Get the class
    let seekable_input_stream_class = env.get_object_class(seekable_stream_obj)?;

    // 2. Get the method ID for "getPos()" (signature "()J")
    let get_pos_method_id = env.get_method_id(seekable_input_stream_class, "getPos", "()J")?;

    // 3. Call the method and get the long return value
    let result = env.call_method(seekable_stream_obj, get_pos_method_id, &[])?;

    // 4. Convert the JValue result to a jlong
    let pos = result.j()?;

    Ok(pos)
}

// Function to call the read method of an InputStream (or SeekableInputStream)
pub fn call_read_on_input_stream<'local>(
    env: &mut JNIEnv<'local>,
    input_stream_obj: JObject<'local>,
    buffer: &mut [u8], // Rust buffer to read into
) -> Result<jint, jni::errors::Error> {
    // 1. Get the class
    let input_stream_class = env.get_object_class(input_stream_obj)?;

    // 2. Get the method ID for "read(byte[] b)" (signature "([B)I")
    let read_method_id = env.get_method_id(input_stream_class, "read", "([B)I")?;

    // 3. Create a Java byte array from the Rust buffer
    let java_buffer = env.new_byte_array(buffer.len() as jint)?;

    // 4. Call the read method
    let result = env.call_method(
        input_stream_obj,
        read_method_id,
        &[JValue::Object(java_buffer.into())],
    )?;

    // 5. Convert the JValue result to a jint (number of bytes read)
    let bytes_read = result.i()?;

    // 6. Copy data from Java byte array to Rust buffer
    if bytes_read > 0 {
        env.get_byte_array_region(&java_buffer, 0, &mut buffer[0..bytes_read as usize])?;
    }

    Ok(bytes_read)
}

// Function to call the readFully method of a SeekableInputStream
// (assuming void readFully(byte[] bytes))
pub fn call_read_fully_on_seekable_input_stream<'local>(
    env: &mut JNIEnv<'local>,
    seekable_stream_obj: JObject<'local>,
    buffer: &mut [u8],
) -> Result<(), jni::errors::Error> {
    // 1. Get the class
    let seekable_input_stream_class = env.get_object_class(seekable_stream_obj)?;

    // 2. Get the method ID for "readFully(byte[] bytes)" (signature "([B)V")
    let read_fully_method_id = env.get_method_id(seekable_input_stream_class, "readFully", "([B)V")?;

    // 3. Create a Java byte array from the Rust buffer
    let java_buffer = env.new_byte_array(buffer.len() as jint)?;

    // 4. Call the readFully method
    env.call_method(
        seekable_stream_obj,
        read_fully_method_id,
        &[JValue::Object(java_buffer.into())],
    )?;

    // 5. Copy data from Java byte array to Rust buffer
    env.get_byte_array_region(&java_buffer, 0, buffer)?;

    Ok(())
}

// Function to call the readFully method of a SeekableInputStream
// with offset and length (assuming void readFully(byte[] bytes, int start, int len))
pub fn call_read_fully_with_offset_and_len_on_seekable_input_stream<'local>(
    env: &mut JNIEnv<'local>,
    seekable_stream_obj: JObject<'local>,
    buffer: &mut [u8], // Rust buffer to read into
    start: jint,      // Start offset within the buffer
    len: jint,        // Number of bytes to read
) -> Result<(), jni::errors::Error> {
    // 1. Get the class of the SeekableInputStream object
    let seekable_input_stream_class = env.get_object_class(seekable_stream_obj)?;

    // 2. Get the method ID for "readFully(byte[] bytes, int start, int len)"
    // The signature is "([BII)V" ([B for byte array, I for int, V for void)
    let read_fully_method_id = env.get_method_id(seekable_input_stream_class, "readFully", "([BII)V")?;

    // 3. Create a Java byte array from the Rust buffer (or a portion of it)
    // Note: It's necessary to create a Java byte array that is the same size as the Rust buffer.
    // The 'start' and 'len' parameters will be passed to the Java method to specify
    // the read region within the Java array.
    let java_buffer = env.new_byte_array(buffer.len() as jint)?;

    // 4. Call the readFully method with the byte array, start, and len
    env.call_method(
        seekable_stream_obj,
        read_fully_method_id,
        &[
            JValue::Object(java_buffer.into()),
            JValue::Int(start),
            JValue::Int(len),
        ],
    )?;

    // 5. Copy data from the Java byte array to the Rust buffer
    // Only copy the portion that was actually read, respecting the 'len' parameter.
    env.get_byte_array_region(&java_buffer, start, &mut buffer[start as usize..(start + len) as usize])?;

    Ok(())
}

// Example usage within a native method called from Java:
#[no_mangle]
pub extern "system" fn Java_com_your_package_MyRustBridge_processInputStream(
    env: JNIEnv,
    _class: jni::objects::JClass,
    input_stream: JObject, // The SeekableInputStream passed from Java
) {
    let mut env = env;

    // Call the seek method
    if let Err(e) = call_seek_on_seekable_input_stream(&mut env, input_stream, 1024) {
        eprintln!("Error calling seek on SeekableInputStream: {:?}", e);
    }

    // Call the getPos method
    match call_get_pos_on_seekable_input_stream(&mut env, input_stream) {
        Ok(pos) => println!("Current position: {}", pos),
        Err(e) => eprintln!("Error calling getPos: {:?}", e),
    }

    // Call the read method
    let mut buffer = vec![0u8; 100]; // Buffer to read into
    match call_read_on_input_stream(&mut env, input_stream, &mut buffer) {
        Ok(bytes_read) => {
            println!("Read {} bytes", bytes_read);
            // Process the data in 'buffer'
        }
        Err(e) => eprintln!("Error calling read: {:?}", e),
    }

    // Call the readFully method
    let mut read_fully_buffer = vec![0u8; 50];
    if let Err(e) = call_read_fully_on_seekable_input_stream(&mut env, input_stream, &mut read_fully_buffer) {
        eprintln!("Error calling readFully: {:?}", e);
    }
}

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

use crate::parquet::from_u8_slice;
use jni::{
    objects::{JObject, JValue},
    sys::{jint, jlong},
    JNIEnv,
};

struct HadoopInputStream<'a> {
    env: JNIEnv<'a>,
    seekable_object_stream: JObject<'a>,
}

impl<'a> HadoopInputStream<'a> {
    pub fn new(env: JNIEnv<'a>, seekable_object_stream: JObject<'a>) -> Self {
        Self {
            env,
            seekable_object_stream,
        }
    }
}

impl HadoopInputStream<'_> {
    // long SeekableInputStream.seek(long pos)
    pub fn seek(&mut self, position: jlong) -> Result<jlong, jni::errors::Error> {
        let result = self.env.call_method(
            &self.seekable_object_stream,
            "seek",
            "(J)V",
            &[JValue::Long(position)],
        )?;
        result.j()
    }

    // long SeekableInputStream.getPos()
    pub fn get_pos(&mut self) -> Result<jlong, jni::errors::Error> {
        let result = self
            .env
            .call_method(&self.seekable_object_stream, "getPos", "()J", &[])?;
        let pos = result.j()?;
        Ok(pos)
    }

    // int InputStream.read()
    pub unsafe fn read(
        &mut self,
        buffer: &mut [u8], // Rust buffer to read into
    ) -> Result<jint, jni::errors::Error> {
        let java_buffer = self.env.new_byte_array(buffer.len() as jint)?;
        let result = self.env.call_method(
            &self.seekable_object_stream,
            "read",
            "([B)I",
            &[JValue::Object(&*java_buffer)],
        )?;
        let bytes_read = result.i()?;
        if bytes_read > 0 {
            self.env
                .get_byte_array_region(&java_buffer, 0, from_u8_slice(buffer))?;
        }
        Ok(bytes_read)
    }

    //  void readFully(byte[] bytes))
    pub unsafe fn read_fully(&mut self, buffer: &mut [u8]) -> Result<(), jni::errors::Error> {
        let java_buffer = self.env.new_byte_array(buffer.len() as jint)?;
        self.env.call_method(
            &self.seekable_object_stream,
            "readFully",
            "([B)V",
            &[JValue::Object(&*java_buffer)],
        )?;
        self.env
            .get_byte_array_region(&java_buffer, 0, from_u8_slice(buffer))?;
        Ok(())
    }

    // void readFully(byte[] bytes, int start, int len))
    pub fn read_fully_with_offset_and_len(
        &mut self,
        buffer: &mut [u8], // Rust buffer to read into
        start: jint,       // Start offset within the buffer
        len: jint,         // Number of bytes to read
    ) -> Result<(), jni::errors::Error> {
        let java_buffer = self.env.new_byte_array(buffer.len() as jint)?;
        self.env.call_method(
            &self.seekable_object_stream,
            "readFully",
            "([BII)V",
            &[
                JValue::Object(&*java_buffer),
                JValue::Int(start),
                JValue::Int(len),
            ],
        )?;
        self.env
            .get_byte_array_region(&java_buffer, 0, from_u8_slice(buffer))?;
        Ok(())
    }
}

// Example usage within a native method called from Java:
#[no_mangle]
pub unsafe extern "system" fn Java_org_apache_comet_parquet_HadoopBridge_processInputStream(
    env: JNIEnv,
    _class: jni::objects::JClass,
    input_stream: JObject, // The SeekableInputStream passed from Java
) {
    let mut stream = HadoopInputStream::new(env, input_stream);

    // Call the seek method
    if let Err(e) = stream.seek(1024) {
        println!("Error calling seek on SeekableInputStream: {:?}", e);
    }

    // Call the getPos method
    match stream.get_pos() {
        Ok(pos) => println!("Current position: {}", pos),
        Err(e) => println!("Error calling getPos: {:?}", e),
    }

    // Call the read method
    let mut buffer = vec![0u8; 100]; // Buffer to read into
    match stream.read(&mut buffer) {
        Ok(bytes_read) => {
            println!("Read {} bytes", bytes_read);
            // Process the data in 'buffer'
        }
        Err(e) => println!("Error calling read: {:?}", e),
    }

    // Call the readFully method
    let mut read_fully_buffer = vec![0u8; 50];
    if let Err(e) = stream.read_fully(&mut read_fully_buffer) {
        println!("Error calling readFully: {:?}", e);
    }
}

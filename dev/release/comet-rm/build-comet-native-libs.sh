#!/bin/bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

# builds a comet binary
REPO=$1
BRANCH=$2
ARCH=$3

function usage {
  local NAME=$(basename $0)
  echo "Usage: ${NAME} [git repo] [branch] [arm64 | amd64]"
  exit 1
}

if [ $# -ne 3 ]
then
  usage
fi

if [ "$ARCH" != "arm64" ] && [ "$ARCH" != "amd64" ]
then
  usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

rm -fr comet
#git clone https://github.com/parthchandra/datafusion-comet.git comet
git clone https://github.com:parthchandra/datafusion-comet.git comet

# build comet binaries
cd comet
git checkout binary-build

make  core-${1}-libs

# copy libs to /opt/host_workdir/output
OUTPUT_DIR="/opt/host_workdir/output"

declare -A OUTPUT_LIBS

if [ "$ARCH" == "arm64" ]
then
   OUTPUT_LIBS=(\
    ["native/target/aarch64-apple-darwin/release/libcomet.dylib"]="$OUTPUT_DIR/common/target/classes/org/apache/comet/darwin/aarch64" \
    ["native/target/release/libcomet.so"]="$OUTPUT_DIR/common/target/classes/org/apache/comet/linux/aarch64" \
  )
else
  OUTPUT_LIBS=(\
    ["native/target/x86_64-apple-darwin/release/libcomet.dylib"]="$OUTPUT_DIR/common/target/classes/org/apache/comet/darwin/x86_64" \
    ["native/target/release/libcomet.so"]="$OUTPUT_DIR/common/target/classes/org/apache/comet/linux/amd64" \
  )
fi

for SRC_LIB in ${!OUTPUT_LIBS[@]}
do
  TARGET_DIR=${OUTPUT_LIBS[${SRC_LIB}]}
  if [ -f "$SRC_LIB" ]
  then
    mkdir -p "$TARGET_DIR"
    echo "Copying $SRC_LIB to $TARGET_DIR"
    cp "$SRC_LIB" "$TARGET_DIR"
  fi
done
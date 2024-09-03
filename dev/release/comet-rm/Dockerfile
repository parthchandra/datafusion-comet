#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
ARG HAS_MACOS_SDK="false"
ARG MACOS_SDK_DIR=""
ARG MACOS_SDK=""

FROM ubuntu:20.04 AS base

USER root

# For apt to be noninteractive
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

ENV LC_ALL=C
# RUN apt-get update \
# Install pr-requisites for rust
RUN export LC_ALL=C \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        build-essential \
        curl \
        wget \
        git \
        llvm \
        clang \
        libssl-dev \
        lzma-dev \
        openssh-client

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#RUN cd /tmp && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
ENV PATH="/root/.cargo/bin:${PATH}"
#RUN rustup install nightly && cargo +nightly install cargo2junit
RUN cargo install cargo2junit

# Use Rust toolchain, and seed Rust
#COPY rust-toolchain.toml  /rust-toolchain.toml
#RUN cargo search

#RUN if [[ -z "$MACOS_SDK" ]] ; then \
#        echo MACOS_SDK not provided ; \
#    else \
#        echo Using MACOS_SDK from $MACOS_SDK  \
#        && cd /opt \
#        && git clone --depth 1 https://github.com/tpoechtrager/osxcross.git \
#        && cd /opt/osxcross/tarballs \
#        && cp $MACOS_SDK . \
#        && cd /opt/osxcross \
#        && UNATTENDED=1 ./build.sh \
#    fi \

# Stage to add OSXCross if MacOSSDK is provided
FROM base AS with-macos-sdk-true

WORKDIR /opt/xcode
COPY ${MACOS_SDK_PATH}/${MACOS_SDK} /opt/xcode

# Build OSXCross
RUN cd /opt && git clone --depth 1 https://github.com/tpoechtrager/osxcross.git \
    && cd /opt/osxcross \
    && ./tools/gen_sdk_package_pbzx.sh /opt/xcode/${MACOS_SDK} \
    && cd .. \
    && cp /opt/xcode/*.tar.xz tarballs \
    && UNATTENDED=1 ./build.sh
ENV PATH="/osxcross/target/bin:${PATH}"
# Use osxcross toolchain for cargo
COPY dev/cargo.config /root/.cargo/config
ENV HAS_OSXCROSS="true"

# Placeholder Stage if MacOSSDK is not provided
FROM base AS with-macos-sdk-false
RUN echo "Building without MacOS"
#RUN wget https://developer.apple.com/download/all/?q=xcode
#ENV HAS_OSXCROSS="false"


FROM with-macos-sdk-${HAS_MACOS_SDK} AS final

#RUN mkdir -p  /opt/comet-rm/output
WORKDIR /opt/comet-rm
COPY

#RUN  git clone https://github.com/apache/datafusion-comet.git /opt/comet

#ARG UID
#RUN useradd -m -s /bin/bash -p comet-rm -u $UID comet-rm
#USER comet-rm:comet-rm

ENTRYPOINT [ "/opt/comet-rm/entrypoint.sh" ]

# maven \
#    xz \
#    patch \
#    openssl-devel \
#    autoconf \
#    libtool \
#    automake \
#    python39-devel \
#    neon-devel \
#    libxml2-devel \
#    cmake \
#    llvm-toolset \
# libxml2-dev
# glibc-locale-source glibc-langpack-en

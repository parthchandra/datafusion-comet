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

package org.apache.comet

import scala.collection.mutable

import org.apache.spark.sql.catalyst.trees.TreeNodeTag

class CometExplainInfo extends Serializable {
  var info: String = ""
  var children: Seq[CometExplainInfo] = Seq.empty

  override def toString: String =
    if (children.isEmpty) info else s"$info(${children.mkString("", ", ", "")})"

  def toSimpleString: String = {
    val sb = mutable.Set[String]()
    dedup(sb)
    sb.mkString("\n")
  }

  private def dedup(all: mutable.Set[String]): mutable.Set[String] = {
    if (children.isEmpty) {
      all += info
    } else {
      children
        .filter(o => (o != CometExplainInfo.none && o != CometExplainInfo.subTreeIsNotNative))
        .map(c => c.dedup(all))
      // return only the child node. Parent nodes clutter up the displayed info and in practice
      // do not have very useful information.
      all
    }
  }
}

case class CometExplainSubtreeIsNotNative() extends CometExplainInfo

object CometExplainInfo {
  val none: CometExplainInfo = null
  val subTreeIsNotNative: CometExplainSubtreeIsNotNative = CometExplainSubtreeIsNotNative()
  val EXTENSION_INFO = new TreeNodeTag[String]("CometExtensionInfo")

  def apply(info: String): CometExplainInfo = {
    val b = new CometExplainInfo
    b.info = info
    b
  }

  def apply(info: String, child: CometExplainInfo): CometExplainInfo = {
    val b = new CometExplainInfo
    b.info = info
    if (child != null) {
      b.children = Seq(child)
    }
    b
  }

  def apply(info: String, children: Seq[CometExplainInfo]): CometExplainInfo = {
    val b = new CometExplainInfo
    b.info = info
    b.children = children
    b
  }
}

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

package org.apache.spark.sql

import java.io.{File, PrintWriter}
import java.nio.file.Files
import java.nio.file.Path

import scala.sys.process._
import scala.util.Try

import org.apache.commons.io.FileUtils
import org.apache.spark.SparkConf
import org.apache.spark.deploy.SparkHadoopUtil

/**
 * This class generates TPCH table data by using tpch-dbgen:
 *   - https://github.com/databricks/tpch-dbgen
 *
 * To run this:
 * {{{
 *   make benchmark-org.apache.spark.sql.GenTPCHData -- --location <path> --scaleFactor 1
 * }}}
 */
object GenTPCHDataIceberg {
  val TEMP_DBGEN_DIR: Path = new File("/tmp").toPath
  val DBGEN_DIR_PREFIX = "tempTPCHGen"

  def main(args: Array[String]): Unit = {
    val config = new GenTPCHDataConfig(args)

    val conf = new SparkConf()
      .setMaster(System.getProperty("spark.sql.test.master", "local[*]"))
      .setAppName(this.getClass.getSimpleName.stripSuffix("$"))
      .set("spark.sql.parquet.compression.codec", "snappy")
      .set(
        "spark.sql.shuffle.partitions",
        System.getProperty("spark.sql.shuffle.partitions", "4"))
      .set("spark.driver.memory", "3g")
      .set("spark.executor.memory", "3g")
      .set("spark.sql.autoBroadcastJoinThreshold", (20 * 1024 * 1024).toString)
      .set("spark.sql.crossJoin.enabled", "true")
      .set("spark.sql.crossJoin.enabled", "true")
      .set("spark.comet.exec.replaceSortMergeJoin", "true")
      .set("spark.comet.exec.shuffle.enabled", "true")
      .set("spark.comet.exec.shuffle.mode", "native")
      .set("spark.comet.exec.shuffle.fallbackToColumnar", "true")
      .set("spark.comet.exec.shuffle.compression.codec", "lz4")
      .set(
        "spark.sql.extensions",
        "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
      .set("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
      .set("spark.sql.catalog.iceberg.type", "hadoop")
      .setIfMissing("parquet.enable.dictionary", "true")
      .set(
        "spark.shuffle.manager",
        "org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager")

    val spark = SparkSession
      .builder()
      .config(conf)
      .appName(getClass.getName)
      .master(config.master)
      .getOrCreate()

    setScaleConfig(spark, config.scaleFactor)

    // Number of worker nodes
    val workers = spark.sparkContext.getExecutorMemoryStatus.size

    var defaultDbgenDir: File = null

    val dbgenDir = if (config.dbgenDir == null) {
      defaultDbgenDir = Files.createTempDirectory(TEMP_DBGEN_DIR, DBGEN_DIR_PREFIX).toFile
      val baseDir = defaultDbgenDir.getAbsolutePath
      defaultDbgenDir.delete()
      // Install the data generators in all nodes
      // TODO: think a better way to install on each worker node
      //       such as https://stackoverflow.com/a/40876671
      spark.range(0, workers, 1, workers).foreach(worker => installDBGEN(baseDir)(worker))
      s"${baseDir}/dbgen"
    } else {
      config.dbgenDir
    }

    val tables = new TPCHTables(spark.sqlContext, dbgenDir, config.scaleFactor.toString)

    // Generate data
    // Since dbgen may uses stdout to output the data, tables.genData needs to run table by table
    val tableNames =
      if (config.tableFilter.trim.isEmpty) tables.tables.map(_.name) else Seq(config.tableFilter)
    tableNames.foreach { tableName =>
      tables.genData(
        location = s"${config.location}/tpch/sf${config.scaleFactor}_${config.format}",
        format = config.format,
        overwrite = config.overwrite,
        partitionTables = config.partitionTables,
        clusterByPartitionColumns = config.clusterByPartitionColumns,
        filterOutNullPartitionValues = config.filterOutNullPartitionValues,
        tableFilter = tableName,
        numPartitions = config.numPartitions)
    }

    // Clean up
    if (defaultDbgenDir != null) {
      spark.range(0, workers, 1, workers).foreach { _ =>
        val _ = FileUtils.deleteQuietly(defaultDbgenDir)
      }
    }

    spark.stop()
  }

  def setScaleConfig(spark: SparkSession, scaleFactor: Int): Unit = {
    // Avoid OOM when shuffling large scale factors and errors like 2GB shuffle limit at 10TB like:
    // org.apache.spark.shuffle.FetchFailedException: Too large frame: 9640891355
    // For 10TB 16x4core nodes were needed with the config below, 8x for 1TB and below.
    // About 24hrs. for SF 1 to 10,000.
    if (scaleFactor >= 10000) {
      spark.conf.set("spark.sql.shuffle.partitions", "20000")
      SparkHadoopUtil.get.conf.set("parquet.memory.pool.ratio", "0.1")
    } else if (scaleFactor >= 1000) {
      spark.conf.set(
        "spark.sql.shuffle.partitions",
        "2001"
      ) // one above 2000 to use HighlyCompressedMapStatus
      SparkHadoopUtil.get.conf.set("parquet.memory.pool.ratio", "0.3")
    } else {
      spark.conf.set("spark.sql.shuffle.partitions", "200") // default
      SparkHadoopUtil.get.conf.set("parquet.memory.pool.ratio", "0.5")
    }
  }

  // Install tpch-dbgen (with the stdout patch)
  def installDBGEN(
      baseDir: String,
      url: String = "https://github.com/databricks/tpch-dbgen.git",
      useStdout: Boolean = true)(i: java.lang.Long): Unit = {
    // Check if we want the revision which makes dbgen output to stdout
    val checkoutRevision: String =
      if (useStdout) "git checkout 0469309147b42abac8857fa61b4cf69a6d3128a8 -- bm_utils.c" else ""

    Seq("mkdir", "-p", baseDir).!
    val pw = new PrintWriter(s"${baseDir}/dbgen_$i.sh")
    pw.write(s"""
      |rm -rf ${baseDir}/dbgen
      |rm -rf ${baseDir}/dbgen_install_$i
      |mkdir ${baseDir}/dbgen_install_$i
      |cd ${baseDir}/dbgen_install_$i
      |git clone '$url'
      |cd tpch-dbgen
      |$checkoutRevision
      |sed -i'' -e 's/#include <malloc\\.h>/#ifndef __APPLE__\\n#include <malloc\\.h>\\n#endif/' bm_utils.c
      |sed -i'' -e 's/#include <malloc\\.h>/#if defined(__MACH__)\\n#include <stdlib\\.h>\\n#else\\n#include <malloc\\.h>\\n#endif/' varsub.c
      |make
      |ln -sf ${baseDir}/dbgen_install_$i/tpch-dbgen ${baseDir}/dbgen || echo "ln -sf failed"
      |test -e ${baseDir}/dbgen/dbgen
      |echo "OK"
      """.stripMargin)
    pw.close
    Seq("chmod", "+x", s"${baseDir}/dbgen_$i.sh").!
    Seq(s"${baseDir}/dbgen_$i.sh").!!
  }

  def createTables(spark: SparkSession): Unit = {
    val script: Array[String] = Array(
      "drop database if exists iceberg.`comet-test`",
      "create database if not exists iceberg.`comet-test`",
      "use iceberg.`comet-test`",
      "drop table if exists part",
      "drop table if exists supplier",
      "drop table if exists partsupp",
      "drop table if exists customer",
      "drop table if exists orders",
      "drop table if exists lineitem",
      "drop table if exists nation",
      "drop table if exists region",
      """
         CREATE TABLE
        `part`
      (
        p_partkey     BIGINT ,
        p_name        VARCHAR(55) ,
        p_mfgr        CHAR(25) ,
        p_brand       CHAR(10) ,
        p_type        VARCHAR(25) ,
        p_size        BIGINT ,
        p_container   CHAR(10) ,
        p_retailprice DECIMAL(12,2) ,
        p_comment     VARCHAR(23)
      )
      USING ICEBERG
        TBLPROPERTIES (
          'write.parquet.row-group-size-bytes'='536870912',
      'write.parquet.compression-codec'='zstd'
      )
      PARTITIONED BY (`p_brand`)
      DISTRIBUTED BY PARTITION
      """,
      """
         CREATE TABLE
        `supplier`
      (
        s_suppkey     BIGINT ,
        s_name        CHAR(25) ,
        s_address     VARCHAR(40) ,
        s_nationkey   BIGINT ,
        s_phone       CHAR(15) ,
        s_acctbal     DECIMAL(12,2) ,
        s_comment     VARCHAR(101)
      )
      USING ICEBERG
        TBLPROPERTIES (
          'write.parquet.row-group-size-bytes'='536870912',
      'write.parquet.compression-codec'='zstd'
      )
      """,
      """
         CREATE TABLE
        `partsupp`
      (
        ps_partkey     BIGINT ,
        ps_suppkey     BIGINT ,
        ps_availqty    INT ,
        ps_supplycost  DECIMAL(12,2)  ,
        ps_comment     VARCHAR(199)
      )
      USING ICEBERG
        TBLPROPERTIES (
          'write.parquet.row-group-size-bytes'='536870912',
      'write.parquet.compression-codec'='zstd'
      )
      """,
      """
         CREATE TABLE
        `customer`
      (
        c_custkey     BIGINT ,
        c_name        VARCHAR(25) ,
        c_address     VARCHAR(40) ,
        c_nationkey   BIGINT ,
        c_phone       CHAR(15) ,
        c_acctbal     DECIMAL(12,2)   ,
        c_mktsegment  CHAR(10) ,
        c_comment     VARCHAR(117)
      )
      USING ICEBERG
        TBLPROPERTIES (
          'write.parquet.row-group-size-bytes'='536870912',
      'write.parquet.compression-codec'='zstd'
      )
      PARTITIONED BY (`c_mktsegment`)
      DISTRIBUTED BY PARTITION
      """,
      """
         CREATE TABLE
        `orders`
      (
        o_orderkey       BIGINT ,
        o_custkey        BIGINT ,
        o_orderstatus    CHAR(1) ,
        o_totalprice     DECIMAL(12,2) ,
        o_orderdate      DATE ,
        o_orderpriority  CHAR(15) ,
        o_clerk          CHAR(15) ,
        o_shippriority   INT ,
        o_comment        VARCHAR(79)
      )
      USING ICEBERG
        TBLPROPERTIES (
          'write.parquet.row-group-size-bytes'='536870912',
      'write.parquet.compression-codec'='zstd'
      )
      PARTITIONED BY (`o_orderdate`)
      DISTRIBUTED BY PARTITION
      """,
      """
      CREATE TABLE
        `lineitem`
      (
        l_orderkey    BIGINT ,
        l_partkey     BIGINT ,
        l_suppkey     BIGINT ,
        l_linenumber  INT ,
        l_quantity    DECIMAL(12,2) ,
        l_extendedprice  DECIMAL(12,2) ,
        l_discount    DECIMAL(12,2) ,
        l_tax         DECIMAL(12,2) ,
        l_returnflag  CHAR(1) ,
        l_linestatus  CHAR(1) ,
        l_shipdate    DATE ,
        l_commitdate  DATE ,
        l_receiptdate DATE ,
        l_shipinstruct CHAR(25) ,
        l_shipmode     CHAR(10) ,
        l_comment      VARCHAR(44)
      )
      USING ICEBERG
        TBLPROPERTIES (
          'write.parquet.row-group-size-bytes'='536870912',
      'write.parquet.compression-codec'='zstd'
      )
      PARTITIONED BY (`l_shipdate`)
      DISTRIBUTED BY PARTITION
     """,
      """
      CREATE TABLE
        `nation`
      (
        n_nationkey  BIGINT ,
        n_name       CHAR(25) ,
        n_regionkey  BIGINT ,
        n_comment    VARCHAR(152)
      )
      USING ICEBERG
        TBLPROPERTIES (
          'write.parquet.row-group-size-bytes'='536870912',
      'write.parquet.compression-codec'='zstd'
      )
      """,
      """
      CREATE TABLE
        `region`
      (
        r_regionkey  BIGINT ,
        r_name       CHAR(25) ,
        r_comment    VARCHAR(152)
      )
      USING ICEBERG
        TBLPROPERTIES (
          'write.parquet.row-group-size-bytes'='536870912',
      'write.parquet.compression-codec'='zstd'
      )
      """)

    for (s <- script) {
      println(s"Creating $s")
      spark.sql(s)
    }

  }
}

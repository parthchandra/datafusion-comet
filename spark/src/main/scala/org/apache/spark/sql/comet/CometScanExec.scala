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

package org.apache.spark.sql.comet

import scala.concurrent.duration.NANOSECONDS
import scala.reflect.ClassTag

import org.apache.hadoop.fs.Path
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst._
import org.apache.spark.sql.catalyst.catalog.BucketSpec
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.plans.QueryPlan
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.catalyst.util.CaseInsensitiveMap
import org.apache.spark.sql.comet.shims.ShimCometScanExec
import org.apache.spark.sql.errors.QueryExecutionErrors
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.datasources._
import org.apache.spark.sql.execution.metric._
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.util.ArrayImplicits.SparkArrayOps
import org.apache.spark.util.collection._

import org.apache.comet.MetricsSupport
import org.apache.comet.parquet.CometParquetFileFormat

/**
 * Comet physical scan node for DataSource V1. Most of the code here follow Spark's
 * [[FileSourceScanExec]],
 */
case class CometScanExec(
    @transient relation: HadoopFsRelation,
    output: Seq[Attribute],
    requiredSchema: StructType,
    partitionFilters: Seq[Expression],
    optionalBucketSet: Option[BitSet],
    optionalNumCoalescedBuckets: Option[Int],
    dataFilters: Seq[Expression],
    tableIdentifier: Option[TableIdentifier],
    disableBucketedScan: Boolean = false,
    wrapped: FileSourceScanExec)
    extends FileSourceScanLike
    with ShimCometScanExec
    with CometPlan {

  // FIXME: ideally we should reuse wrapped.supportsColumnar, however that fails many tests
  override lazy val supportsColumnar: Boolean =
    relation.fileFormat.supportBatch(relation.sparkSession, schema)

  override def vectorTypes: Option[Seq[String]] = wrapped.vectorTypes

//  lazy val driverMetrics: HashMap[String, Long] = HashMap.empty

  /**
   * Send the driver-side metrics. Before calling this function, selectedPartitions has been
   * initialized. See SPARK-26327 for more details.
   */
//  private def sendDriverMetrics(): Unit = {
//    driverMetrics.foreach(e => metrics(e._1).add(e._2))
//    val executionId = sparkContext.getLocalProperty(SQLExecution.EXECUTION_ID_KEY)
//    SQLMetrics.postDriverMetricUpdates(
//      sparkContext,
//      executionId,
//      metrics.filter(e => driverMetrics.contains(e._1)).values.toSeq)
//  }

  private def isDynamicPruningFilter(e: Expression): Boolean =
    e.find(_.isInstanceOf[PlanExpression[_]]).isDefined

  @transient override lazy val selectedPartitions: ScanFileListing = {
    wrapped.selectedPartitions
  }

  // We can only determine the actual partitions at runtime when a dynamic partition filter is
  // present. This is because such a filter relies on information that is only available at run
  // time (for instance the keys used in the other side of a join).
  @transient override protected lazy val dynamicallySelectedPartitions: ScanFileListing = {
    val dynamicDataFilters = dataFilters.filter(isDynamicPruningFilter)
    val dynamicPartitionFilters = partitionFilters.filter(isDynamicPruningFilter)

    if (dynamicPartitionFilters.nonEmpty) {
      val startTime = System.nanoTime()
      // call the file index for the files matching all filters except dynamic partition filters
      val predicate = dynamicPartitionFilters.reduce(And)
      val partitionColumns = relation.partitionSchema
      val boundPredicate = Predicate.create(
        predicate.transform { case a: AttributeReference =>
          val index = partitionColumns.indexWhere(a.name == _.name)
          BoundReference(index, partitionColumns(index).dataType, nullable = true)
        },
        Nil)
      val returnedFiles =
        selectedPartitions.filterAndPruneFiles(boundPredicate, dynamicDataFilters)
      setFilesNumAndSizeMetric(returnedFiles, false)
      val timeTakenMs = NANOSECONDS.toMillis(System.nanoTime() - startTime)
      driverMetrics("pruningTime").set(timeTakenMs)
      returnedFiles
    } else {
      selectedPartitions
    }
  }

  // exposed for testing
  override lazy val bucketedScan: Boolean = wrapped.bucketedScan

  override lazy val (outputPartitioning, outputOrdering): (Partitioning, Seq[SortOrder]) =
    (wrapped.outputPartitioning, wrapped.outputOrdering)

//  def translateToV1Filters(
//      dataFilters: Seq[Expression],
//      scalarSubqueryToLiteral: execution.ScalarSubquery => Literal): Seq[Filter] = {
//    val scalarSubqueryReplaced = dataFilters.map(_.transform {
//      // Replace scalar subquery to literal so that `DataSourceStrategy.translateFilter` can
//      // support translating it.
//      case scalarSubquery: execution.ScalarSubquery => scalarSubqueryToLiteral(scalarSubquery)
//    })
//
//    val supportNestedPredicatePushdown = DataSourceUtils.supportNestedPredicatePushdown(relation)
//    // `dataFilters` should not include any constant metadata col filters
//    // because the metadata struct has been flatted in FileSourceStrategy
//    // and thus metadata col filters are invalid to be pushed down. Metadata that is generated
//    // during the scan can be used for filters.
//    scalarSubqueryReplaced
//      .filterNot(_.references.exists {
//        case FileSourceConstantMetadataAttribute(_) => true
//        case _ => false
//      })
//      .flatMap(DataSourceStrategy.translateFilter(_, supportNestedPredicatePushdown))
//  }
//  @transient
//  private lazy val pushedDownFilters = translateToV1Filters(dataFilters, _.toLiteral)

  override lazy val metadata: Map[String, String] =
    if (wrapped == null) Map.empty else wrapped.metadata

  override def verboseStringWithOperatorId(): String = {
    getTagValue(QueryPlan.OP_ID_TAG).foreach(id => wrapped.setTagValue(QueryPlan.OP_ID_TAG, id))
    wrapped.verboseStringWithOperatorId()
  }

  lazy val inputRDD: RDD[InternalRow] = {
    val options = relation.options +
      (FileFormat.OPTION_RETURNING_BATCH -> supportsColumnar.toString)
    val readFile: (PartitionedFile) => Iterator[InternalRow] =
      relation.fileFormat.buildReaderWithPartitionValues(
        sparkSession = relation.sparkSession,
        dataSchema = relation.dataSchema,
        partitionSchema = relation.partitionSchema,
        requiredSchema = requiredSchema,
        filters = pushedDownFilters,
        options = options,
        hadoopConf =
          relation.sparkSession.sessionState.newHadoopConfWithOptions(relation.options))

    val readRDD = if (bucketedScan) {
      createBucketedReadRDD(relation.bucketSpec.get, readFile, dynamicallySelectedPartitions)
    } else {
      createReadRDD(readFile, dynamicallySelectedPartitions)
    }
    sendDriverMetrics()
    readRDD
  }

  override def inputRDDs(): Seq[RDD[InternalRow]] = {
    inputRDD :: Nil
  }

  /** Helper for computing total number and size of files in selected partitions. */
  private def setFilesNumAndSizeMetric(partitions: ScanFileListing, static: Boolean): Unit = {
    val filesNum = partitions.totalNumberOfFiles
    val filesSize = partitions.totalFileSize
    if (!static || !partitionFilters.exists(isDynamicPruningFilter)) {
      driverMetrics("numFiles").set(filesNum)
      driverMetrics("filesSize").set(filesSize)
    } else {
      driverMetrics("staticFilesNum").set(filesNum)
      driverMetrics("staticFilesSize").set(filesSize)
    }
    if (relation.partitionSchema.nonEmpty) {
      driverMetrics("numPartitions").set(partitions.partitionCount)
    }
  }

  override lazy val metrics: Map[String, SQLMetric] = wrapped.metrics ++ {
    // Tracking scan time has overhead, we can't afford to do it for each row, and can only do
    // it for each batch.
    if (supportsColumnar) {
      Some("scanTime" -> SQLMetrics.createTimingMetric(sparkContext, "scan time"))
    } else {
      None
    }
  } ++ {
    relation.fileFormat match {
      case f: MetricsSupport => f.initMetrics(sparkContext)
      case _ => Map.empty
    }
  }

  protected override def doExecute(): RDD[InternalRow] = {
    ColumnarToRowExec(this).doExecute()
  }

  protected override def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val numOutputRows = longMetric("numOutputRows")
    val scanTime = longMetric("scanTime")
    inputRDD.asInstanceOf[RDD[ColumnarBatch]].mapPartitionsInternal { batches =>
      new Iterator[ColumnarBatch] {

        override def hasNext: Boolean = {
          // The `FileScanRDD` returns an iterator which scans the file during the `hasNext` call.
          val startNs = System.nanoTime()
          val res = batches.hasNext
          scanTime += NANOSECONDS.toMillis(System.nanoTime() - startNs)
          res
        }

        override def next(): ColumnarBatch = {
          val batch = batches.next()
          numOutputRows += batch.numRows()
          batch
        }
      }
    }
  }

  override def executeCollect(): Array[InternalRow] = {
    ColumnarToRowExec(this).executeCollect()
  }

  override val nodeName: String =
    s"CometScan $relation ${tableIdentifier.map(_.unquotedString).getOrElse("")}"

  /**
   * Create an RDD for bucketed reads. The non-bucketed variant of this function is
   * [[createReadRDD]].
   *
   * The algorithm is pretty simple: each RDD partition being returned should include all the
   * files with the same bucket id from all the given Hive partitions.
   *
   * @param bucketSpec
   *   the bucketing spec.
   * @param readFile
   *   a function to read each (part of a) file.
   * @param selectedPartitions
   *   Hive-style partition that are part of the read.
   */
  private def createBucketedReadRDD(
      bucketSpec: BucketSpec,
      readFile: (PartitionedFile) => Iterator[InternalRow],
      selectedPartitions: ScanFileListing): RDD[InternalRow] = {
    logInfo("Planning with ${bucketSpec.numBuckets} buckets")
    val partitionArray = selectedPartitions.toPartitionArray
    val filesGroupedToBuckets = partitionArray.groupBy { f =>
      BucketingUtils
        .getBucketId(f.toPath.getName)
        .getOrElse(throw QueryExecutionErrors.invalidBucketFile(f.urlEncodedPath))
    }

    val prunedFilesGroupedToBuckets = if (optionalBucketSet.isDefined) {
      val bucketSet = optionalBucketSet.get
      filesGroupedToBuckets.filter { f =>
        bucketSet.get(f._1)
      }
    } else {
      filesGroupedToBuckets
    }

    val filePartitions = optionalNumCoalescedBuckets
      .map { numCoalescedBuckets =>
        logInfo("Coalescing to $numCoalescedBuckets buckets")
        val coalescedBuckets = prunedFilesGroupedToBuckets.groupBy(_._1 % numCoalescedBuckets)
        Seq.tabulate(numCoalescedBuckets) { bucketId =>
          val partitionedFiles = coalescedBuckets
            .get(bucketId)
            .map {
              _.values.flatten.toArray
            }
            .getOrElse(Array.empty)
          FilePartition(bucketId, partitionedFiles)
        }
      }
      .getOrElse {
        Seq.tabulate(bucketSpec.numBuckets) { bucketId =>
          FilePartition(bucketId, prunedFilesGroupedToBuckets.getOrElse(bucketId, Array.empty))
        }
      }

    new FileScanRDD(
      relation.sparkSession,
      readFile,
      filePartitions,
      new StructType(requiredSchema.fields ++ relation.partitionSchema.fields),
      fileConstantMetadataColumns,
      relation.fileFormat.fileConstantMetadataExtractors,
      new FileSourceOptions(CaseInsensitiveMap(relation.options)))
  }

  /**
   * Create an RDD for non-bucketed reads. The bucketed variant of this function is
   * [[createBucketedReadRDD]].
   *
   * @param readFile
   *   a function to read each (part of a) file.
   * @param selectedPartitions
   *   Hive-style partition that are part of the read.
   */
  private def createReadRDD(
      readFile: PartitionedFile => Iterator[InternalRow],
      selectedPartitions: ScanFileListing): RDD[InternalRow] = {
    val openCostInBytes = relation.sparkSession.sessionState.conf.filesOpenCostInBytes
    val maxSplitBytes =
      FilePartition.maxSplitBytes(relation.sparkSession, selectedPartitions)
    logInfo(
      "Planning scan with bin packing, max size: $maxSplitBytes " +
        "bytes, open cost is considered as scanning $openCostInBytes " +
        "bytes.")

    // Filter files with bucket pruning if possible
    val bucketingEnabled = relation.sparkSession.sessionState.conf.bucketingEnabled
    val shouldProcess: Path => Boolean = optionalBucketSet match {
      case Some(bucketSet) if bucketingEnabled =>
        // Do not prune the file if bucket file name is invalid
        filePath => BucketingUtils.getBucketId(filePath.getName).forall(bucketSet.get)
      case _ =>
        _ => true
    }

    val splitFiles = selectedPartitions.filePartitionIterator
      .flatMap { partition =>
        val ListingPartition(partitionVals, _, fileStatusIterator) = partition
        fileStatusIterator.flatMap { file =>
          if (shouldProcess(file.getPath)) {
            val isSplitable = relation.fileFormat.isSplitable(
              relation.sparkSession,
              relation.options,
              file.getPath)
            PartitionedFileUtil.splitFiles(
              file = file,
              isSplitable = isSplitable,
              maxSplitBytes = maxSplitBytes,
              partitionValues = partitionVals)
          } else {
            Seq.empty
          }
        }
      }
      .toArray
      .sortBy(_.length)(implicitly[Ordering[Long]].reverse)

    val partitions = FilePartition
      .getFilePartitions(relation.sparkSession, splitFiles.toImmutableArraySeq, maxSplitBytes)

    new FileScanRDD(
      relation.sparkSession,
      readFile,
      partitions,
      new StructType(requiredSchema.fields ++ relation.partitionSchema.fields),
      fileConstantMetadataColumns,
      relation.fileFormat.fileConstantMetadataExtractors,
      new FileSourceOptions(CaseInsensitiveMap(relation.options)))
  }

//  private def prepareRDD(
//      fsRelation: HadoopFsRelation,
//      readFile: (PartitionedFile) => Iterator[InternalRow],
//      partitions: Seq[FilePartition]): RDD[InternalRow] = {
//    val hadoopConf = relation.sparkSession.sessionState.newHadoopConfWithOptions(relation.options)
//    val prefetchEnabled = hadoopConf.getBoolean(
//      CometConf.COMET_SCAN_PREFETCH_ENABLED.key,
//      CometConf.COMET_SCAN_PREFETCH_ENABLED.defaultValue.get)
//
//    val sqlConf = fsRelation.sparkSession.sessionState.conf
//    if (prefetchEnabled) {
//      CometParquetFileFormat.populateConf(sqlConf, hadoopConf)
//      val broadcastedConf =
//        fsRelation.sparkSession.sparkContext.broadcast(new SerializableConfiguration(hadoopConf))
//      val partitionReaderFactory = CometParquetPartitionReaderFactory(
//        sqlConf,
//        broadcastedConf,
//        requiredSchema,
//        relation.partitionSchema,
//        pushedDownFilters.toArray,
//        new ParquetOptions(CaseInsensitiveMap(relation.options), sqlConf),
//        metrics)
//
//      new DataSourceRDD(
//        fsRelation.sparkSession.sparkContext,
//        partitions.map(Seq(_)),
//        partitionReaderFactory,
//        true,
//        Map.empty)
//    } else {
//      newFileScanRDD(
//        fsRelation,
//        readFile,
//        partitions,
//        new StructType(requiredSchema.fields ++ fsRelation.partitionSchema.fields),
//        new ParquetOptions(CaseInsensitiveMap(relation.options), sqlConf))
//    }
//  }

  // Filters unused DynamicPruningExpression expressions - one which has been replaced
  // with DynamicPruningExpression(Literal.TrueLiteral) during Physical Planning
  private def filterUnusedDynamicPruningExpressions(
      predicates: Seq[Expression]): Seq[Expression] = {
    predicates.filterNot(_ == DynamicPruningExpression(Literal.TrueLiteral))
  }

  override def doCanonicalize(): CometScanExec = {
    CometScanExec(
      relation,
      output.map(QueryPlan.normalizeExpressions(_, output)),
      requiredSchema,
      QueryPlan.normalizePredicates(
        filterUnusedDynamicPruningExpressions(partitionFilters),
        output),
      optionalBucketSet,
      optionalNumCoalescedBuckets,
      QueryPlan.normalizePredicates(dataFilters, output),
      None,
      disableBucketedScan,
      null)
  }
}

object CometScanExec {
  def apply(scanExec: FileSourceScanExec, session: SparkSession): CometScanExec = {
    // TreeNode.mapProductIterator is protected method.
    def mapProductIterator[B: ClassTag](product: Product, f: Any => B): Array[B] = {
      val arr = Array.ofDim[B](product.productArity)
      var i = 0
      while (i < arr.length) {
        arr(i) = f(product.productElement(i))
        i += 1
      }
      arr
    }

    // Replacing the relation in FileSourceScanExec by `copy` seems causing some issues
    // on other Spark distributions if FileSourceScanExec constructor is changed.
    // Using `makeCopy` to avoid the issue.
    // https://github.com/apache/arrow-datafusion-comet/issues/190
    def transform(arg: Any): AnyRef = arg match {
      case _: HadoopFsRelation =>
        scanExec.relation.copy(fileFormat = new CometParquetFileFormat)(session)
      case other: AnyRef => other
      case null => null
    }
    val newArgs = mapProductIterator(scanExec, transform(_))
    val wrapped = scanExec.makeCopy(newArgs).asInstanceOf[FileSourceScanExec]
    val batchScanExec = CometScanExec(
      wrapped.relation,
      wrapped.output,
      wrapped.requiredSchema,
      wrapped.partitionFilters,
      wrapped.optionalBucketSet,
      wrapped.optionalNumCoalescedBuckets,
      wrapped.dataFilters,
      wrapped.tableIdentifier,
      wrapped.disableBucketedScan,
      wrapped)
    scanExec.logicalLink.foreach(batchScanExec.setLogicalLink)
    batchScanExec
  }

}

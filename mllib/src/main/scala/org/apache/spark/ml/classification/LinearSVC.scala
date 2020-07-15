/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.classification


import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{CachedDiffFunction, OWLQN => BreezeOWLQN}
import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.optim.aggregator.HingeAggregator
import org.apache.spark.ml.optim.loss.{L2Regularization, RDDLossFunction}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.{SparkEnv, SparkException}

import scala.collection.mutable

/** Params for linear SVM Classifier. */
private[classification] trait LinearSVCParams extends ClassifierParams with HasRegParam
  with HasMaxIter with HasFitIntercept with HasTol with HasStandardization with HasWeightCol
  with HasAggregationDepth with HasThreshold {
  /**
    * Param for threshold in binary classification prediction.
    * For LinearSVC, this threshold is applied to the rawPrediction, rather than a probability.
    * This threshold can be any real number, where Inf will make all predictions 0.0
    * and -Inf will make all predictions 1.0.
    * Default: 0.0
    *
    * @group param
    */
  final override val threshold: DoubleParam = new DoubleParam(this, "threshold",
    "threshold in binary classification prediction applied to rawPrediction")
}

/**
  * :: Experimental ::
  *
  * <a href = "https://en.wikipedia.org/wiki/Support_vector_machine#Linear_SVM">
  * Linear SVM Classifier</a>
  *
  * This binary classifier optimizes the Hinge Loss using the OWLQN optimizer.
  * Only supports L2 regularization currently.
  *
  */
@Since("2.2.0")
@Experimental
class LinearSVC @Since("2.2.0")(
    @Since("2.2.0") override val uid: String)
  extends Classifier[Vector, LinearSVC, LinearSVCModel]
    with LinearSVCParams with DefaultParamsWritable {

  @Since("2.2.0")
  def this() = this(Identifiable.randomUID("linearsvc"))


  /**
    * Set the regularization parameter.
    * Default is 0.0.
    *
    * @group setParam
    */
  @Since("2.2.0")
  def setRegParam(value: Double): this.type = set(regParam, value)
  setDefault(regParam -> 0.0)

  /**
    * Set the maximum number of iterations.
    * Default is 100.
    *
    * @group setParam
    */
  @Since("2.2.0")
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  /**
    * Whether to fit an intercept term.
    * Default is true.
    *
    * @group setParam
    */
  @Since("2.2.0")
  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)
  setDefault(fitIntercept -> true)

  /**
    * Set the convergence tolerance of iterations.
    * Smaller values will lead to higher accuracy at the cost of more iterations.
    * Default is 1E-6.
    *
    * @group setParam
    */
  @Since("2.2.0")
  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  /**
    * Whether to standardize the training features before fitting the model.
    * Default is true.
    *
    * @group setParam
    */
  @Since("2.2.0")
  def setStandardization(value: Boolean): this.type = set(standardization, value)
  setDefault(standardization -> true)

  /**
    * Set the value of param [[weightCol]].
    * If this is not set or empty, we treat all instance weights as 1.0.
    * Default is not set, so all instances have weight one.
    *
    * @group setParam
    */
  @Since("2.2.0")
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /**
    * Set threshold in binary classification.
    *
    * @group setParam
    */
  @Since("2.2.0")
  def setThreshold(value: Double): this.type = set(threshold, value)
  setDefault(threshold -> 0.0)

  /**
    * Suggested depth for treeAggregate (greater than or equal to 2).
    * If the dimensions of features or the number of partitions are large,
    * this param could be adjusted to a larger size.
    * Default is 2.
    *
    * @group expertSetParam
    */
  @Since("2.2.0")
  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)
  setDefault(aggregationDepth -> 2)


  var initialModel: Array[Double] = null
  def setInitialModel(initModel: Array[Double]): Unit = {
    initialModel = initModel
  }

  /**
    * get initial model with size numFeatures. Here for comparision with SGD, we
    * set intercept to be false by default.
    * ZHIPENG == If this is some bug, please make sure that you check this.
    *
    * @param numFeatures
    * @return
    */
  def getInitialModel(numFeatures: Int, featureStd: Array[Double]): OldVector = {
    if (initialModel == null) {
      OldVectors.zeros(numFeatures)
    }
    else {
      if (initialModel.size == numFeatures) {
        // model already initialized, and whether plus intercept is the same
        // be careful here! Since by default we set standization to be false, so the weight has to be:
        // weight = weight * std
        require(initialModel.size == featureStd.size)
        for (i <- 0 to initialModel.size - 1) {
          initialModel(i) = initialModel(i) * featureStd(i)
        }
        OldVectors.dense(initialModel)
      }
      else {
        throw new IllegalArgumentException("ghand: the dimension of initial model should be the same as number of features. Special" +
          "cases should be considered like fitIntercept or not. By default, we don't use intersecpt.")
      }
    }

  }

  var ghandbudget: Int = 10
  /**
    * zhipeng
    *
    * @param value set budget, i.e., how many history data do you want.
    */
  def SetBudget(value: Int): Unit = {
    ghandbudget = value
  }

  @Since("2.2.0")
  override def copy(extra: ParamMap): LinearSVC = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): LinearSVCModel = {
    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val instances: RDD[Instance] =
      dataset.select(col($(labelCol)), w, col($(featuresCol))).rdd.map {
        case Row(label: Double, weight: Double, features: Vector) =>
          Instance(label, weight, features)
      }

    val instr = Instrumentation.create(this, instances)
    instr.logParams(regParam, maxIter, fitIntercept, tol, standardization, threshold,
      aggregationDepth)

    // ghandzhipeng: this function will aggregate statistical information from each executors.
    // each element is 8 * model_size, as a result, for kdd12 dataset, statistical information on
    // each worker is bigger than 2 GB(55m * 8 * 8 ~ 3GB), thus it fails.

    val summarizer = {
      val seqOp = (c: MultivariateOnlineSummarizer,
                   instance: Instance) => {
        (c.add(instance.features, instance.weight))
      }
      val combOp = (c1: MultivariateOnlineSummarizer,
                    c2: MultivariateOnlineSummarizer) => {
        c1.merge(c2)
      }
      instances.treeAggregate(
        (new MultivariateOnlineSummarizer)
      )(seqOp, combOp, 2)
    }

//    val numFeatures = SparkEnv.get.conf.get("spark.ml.numFeature", "-1").toInt
//    if (numFeatures == -1) {
//      throw new IllegalArgumentException("ghand: please set -spark.ml.numFeature- for LinearSVC class.")
//    }
    val numFeatures: Int = instances.map(_.features.size).first()
    val numFeaturesPlusIntercept = if (getFitIntercept) numFeatures + 1 else numFeatures
    val numClasses = SparkEnv.get.conf.get("spark.ml.numClasses", "2").toInt
    logInfo(s"ghand=We assume binary classification")

    instr.logNumClasses(numClasses)
    instr.logNumFeatures(numFeatures)

    val (coefficientVector, interceptVector, objectiveHistory) = {

      var featuresStd: Array[Double] = null
      val useFeatureScaling = SparkEnv.get.conf.get("spark.ml.useFeatureScaling", "true").toBoolean
      if (useFeatureScaling) {
        featuresStd = summarizer.variance.toArray.map(math.sqrt)
      }
      else {
        featuresStd = Array.fill(numFeatures)(1.0)
      }

      val getFeaturesStd = (j: Int) => featuresStd(j)
      val regParamL2 = $(regParam)
      val bcFeaturesStd = instances.context.broadcast(featuresStd)
      val regularization = if (regParamL2 != 0.0) {
        val shouldApply = (idx: Int) => idx >= 0 && idx < numFeatures
        Some(new L2Regularization(regParamL2, shouldApply,
          if ($(standardization)) None else Some(getFeaturesStd)))
      } else {
        None
      }

      val getAggregatorFunc = new HingeAggregator(bcFeaturesStd, $(fitIntercept))(_)
      val costFun = new RDDLossFunction(instances, getAggregatorFunc, regularization,
        $(aggregationDepth))

      def regParamL1Fun = (index: Int) => 0D
      val optimizer = new BreezeOWLQN[Int, BDV[Double]]($(maxIter), ghandbudget, regParamL1Fun, $(tol))
      val initialCoefWithIntercept = getInitialModel(numFeaturesPlusIntercept, featuresStd)

      // zhipeng == calculate the loss given the initial model
      costFun.calculate(initialCoefWithIntercept.asBreeze.toDenseVector)

      val states = optimizer.iterations(new CachedDiffFunction(costFun),
        initialCoefWithIntercept.asBreeze.toDenseVector)

      val scaledObjectiveHistory = mutable.ArrayBuilder.make[Double]
      var state: optimizer.State = null

      // add iterations control
      var iter: Int = 0
      var startTime: Long = 0
      var endTime: Long = 0
      while (states.hasNext) {
        startTime = System.currentTimeMillis()
        state = states.next()
        scaledObjectiveHistory += state.adjustedValue
        endTime = System.currentTimeMillis()
        logInfo(s"ghand=LBFGS=" +
          s"start:${startTime}=end:${endTime}=" +
          s"duration:${endTime - startTime}=loss=${state.adjustedValue}")
      }

      bcFeaturesStd.destroy(blocking = false)
      if (state == null) {
        val msg = s"${optimizer.getClass.getName} failed."
        logError(msg)
        throw new SparkException(msg)
      }

      /*
         The coefficients are trained in the scaled space; we're converting them back to
         the original space.
         Note that the intercept in scaled space and original space is the same;
         as a result, no scaling is needed.
       */
      val rawCoefficients = state.x.toArray
      val coefficientArray = Array.tabulate(numFeatures) { i =>
        if (featuresStd(i) != 0.0) {
          rawCoefficients(i) / featuresStd(i)
        } else {
          0.0
        }
      }

      val intercept = if ($(fitIntercept)) {
        rawCoefficients(numFeaturesPlusIntercept - 1)
      } else {
        0.0
      }
      (Vectors.dense(coefficientArray), intercept, scaledObjectiveHistory.result())
    }

    val model = copyValues(new LinearSVCModel(uid, coefficientVector, interceptVector))
    instr.logSuccess(model)
    model
  }
}

@Since("2.2.0")
object LinearSVC extends DefaultParamsReadable[LinearSVC] {

  @Since("2.2.0")
  override def load(path: String): LinearSVC = super.load(path)
}

/**
  * :: Experimental ::
  * Linear SVM Model trained by [[LinearSVC]]
  */
@Since("2.2.0")
@Experimental
class LinearSVCModel private[classification](
                                              @Since("2.2.0") override val uid: String,
                                              @Since("2.2.0") val coefficients: Vector,
                                              @Since("2.2.0") val intercept: Double)
  extends ClassificationModel[Vector, LinearSVCModel]
    with LinearSVCParams with MLWritable {

  @Since("2.2.0")
  override val numClasses: Int = 2

  @Since("2.2.0")
  override val numFeatures: Int = coefficients.size

  @Since("2.2.0")
  def setThreshold(value: Double): this.type = set(threshold, value)

  setDefault(threshold, 0.0)

  @Since("2.2.0")
  def setWeightCol(value: Double): this.type = set(threshold, value)

  private val margin: Vector => Double = (features) => {
    BLAS.dot(features, coefficients) + intercept
  }

  override protected def predict(features: Vector): Double = {
    if (margin(features) > $(threshold)) 1.0 else 0.0
  }

  override protected def predictRaw(features: Vector): Vector = {
    val m = margin(features)
    Vectors.dense(-m, m)
  }

  override protected def raw2prediction(rawPrediction: Vector): Double = {
    if (rawPrediction(1) > $(threshold)) 1.0 else 0.0
  }

  @Since("2.2.0")
  override def copy(extra: ParamMap): LinearSVCModel = {
    copyValues(new LinearSVCModel(uid, coefficients, intercept), extra).setParent(parent)
  }

  @Since("2.2.0")
  override def write: MLWriter = new LinearSVCModel.LinearSVCWriter(this)

}


@Since("2.2.0")
object LinearSVCModel extends MLReadable[LinearSVCModel] {

  @Since("2.2.0")
  override def read: MLReader[LinearSVCModel] = new LinearSVCReader

  @Since("2.2.0")
  override def load(path: String): LinearSVCModel = super.load(path)

  /** [[MLWriter]] instance for [[LinearSVCModel]] */
  private[LinearSVCModel]
  class LinearSVCWriter(instance: LinearSVCModel)
    extends MLWriter with Logging {

    private case class Data(coefficients: Vector, intercept: Double)

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.coefficients, instance.intercept)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class LinearSVCReader extends MLReader[LinearSVCModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[LinearSVCModel].getName

    override def load(path: String): LinearSVCModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.format("parquet").load(dataPath)
      val Row(coefficients: Vector, intercept: Double) =
        data.select("coefficients", "intercept").head()
      val model = new LinearSVCModel(metadata.uid, coefficients, intercept)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

}

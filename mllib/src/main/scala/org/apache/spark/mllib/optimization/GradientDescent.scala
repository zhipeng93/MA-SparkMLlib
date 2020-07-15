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

package org.apache.spark.mllib.optimization

import breeze.linalg.{DenseVector => BDV, _}
import org.apache.spark.SparkEnv
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Instance
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.rdd.RDD

/**
  * Class used to solve an optimization problem using Gradient Descent.
  *
  * @param gradient Gradient function to be used.
  * @param updater  Updater to be used to update weights after every iteration.
  */
class GradientDescent private[spark](private var gradient: Gradient, private var updater: Updater)
  extends Optimizer with Logging {

  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
  private var convergenceTol: Double = 0.0

  /**
    * Set the initial step size of SGD for the first step. Default 1.0.
    * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
    */
  def setStepSize(step: Double): this.type = {
    require(step > 0,
      s"Initial step size must be positive but got ${step}")
    this.stepSize = step
    this
  }

  /**
    * Set fraction of data to be used for each SGD iteration.
    * Default 1.0 (corresponding to deterministic/classical gradient descent)
    */
  def setMiniBatchFraction(fraction: Double): this.type = {
    require(fraction > 0 && fraction <= 1.0,
      s"Fraction for mini-batch SGD must be in range (0, 1] but got ${fraction}")
    this.miniBatchFraction = fraction
    this
  }

  /**
    * Set the number of iterations for SGD. Default 100.
    */
  def setNumIterations(iters: Int): this.type = {
    require(iters >= 0,
      s"Number of iterations must be nonnegative but got ${iters}")
    this.numIterations = iters
    this
  }

  /**
    * Set the regularization parameter. Default 0.0.
    */
  def setRegParam(regParam: Double): this.type = {
    require(regParam >= 0,
      s"Regularization parameter must be nonnegative but got ${regParam}")
    this.regParam = regParam
    this
  }

  /**
    * Set the convergence tolerance. Default 0.001
    * convergenceTol is a condition which decides iteration termination.
    * The end of iteration is decided based on below logic.
    *
    * - If the norm of the new solution vector is greater than 1, the diff of solution vectors
    * is compared to relative tolerance which means normalizing by the norm of
    * the new solution vector.
    * - If the norm of the new solution vector is less than or equal to 1, the diff of solution
    * vectors is compared to absolute tolerance which is not normalizing.
    *
    * Must be between 0.0 and 1.0 inclusively.
    */
  def setConvergenceTol(tolerance: Double): this.type = {
    require(tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
    this.convergenceTol = tolerance
    this
  }

  /**
    * Set the gradient function (of the loss function of one single data example)
    * to be used for SGD.
    */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
    * Set the updater function to actually perform a gradient step in a given direction.
    * The updater is responsible to perform the update from the regularization term as well,
    * and therefore determines what kind or regularization is used, if any.
    */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  /**
    * :: DeveloperApi ::
    * Runs gradient descent on the given training data.
    *
    * @param data           training data
    * @param initialWeights initial weights
    * @return solution vector
    */
  @DeveloperApi
  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val optim = SparkEnv.get.conf.get("spark.ml.sgd.optim", "sgd")

    val (weights, _) = optim match {
      case "sgd" => GradientDescent.runMiniBatchSGD(
        data,
        gradient,
        updater,
        stepSize,
        numIterations,
        regParam,
        miniBatchFraction,
        initialWeights,
        convergenceTol
      )
      case "momentum" => GradientDescent.runMiniBatchMomentumSGD(
        data,
        gradient,
        updater,
        stepSize,
        numIterations,
        regParam,
        miniBatchFraction,
        initialWeights,
        convergenceTol
      )
      case "rmsprop" => GradientDescent.runMiniBatchRMSPropSGD(
        data,
        gradient,
        updater,
        stepSize,
        numIterations,
        regParam,
        miniBatchFraction,
        initialWeights,
        convergenceTol
      )
      case "adam" => GradientDescent.runMiniBatchAdamSGD(
        data,
        gradient,
        updater,
        stepSize,
        numIterations,
        regParam,
        miniBatchFraction,
        initialWeights,
        convergenceTol
      )
      case "amsgrad" => GradientDescent.runMiniBatchAMSGradSGD(
        data,
        gradient,
        updater,
        stepSize,
        numIterations,
        regParam,
        miniBatchFraction,
        initialWeights,
        convergenceTol
      )
      case _ => null
    }
    weights
  }
}

/**
  * :: DeveloperApi ::
  * Top-level method to run gradient descent.
  */
@DeveloperApi
object GradientDescent extends Logging {

  /**
    * return featureStd, if "spark.ml.useFeatureScaling" is true, return featureStd,
    * else return [1.0] * num_features
    *
    * @param data
    * @return
    */
  def getFeatureStd(data: RDD[(Double, Vector)], num_features: Int): Array[Double] = {
    val instances: RDD[Instance] = data.map(
      x =>
        Instance(x._1, 1, x._2.asML)
    )
    val summarizer = {
      val seqOp = (c: MultivariateOnlineSummarizer,
                   instance: Instance) => {
        c.add(Vectors.fromML(instance.features), instance.weight)
      }

      val combOp = (c1: MultivariateOnlineSummarizer,
                    c2: MultivariateOnlineSummarizer) => {
        c1.merge(c2)
      }

      instances.treeAggregate(
        (new MultivariateOnlineSummarizer)
      )(seqOp, combOp, 3)
    }

    var featuresStd: Array[Double] = null
    if (SparkEnv.get.conf.get("spark.ml.useFeatureScaling", "true").toBoolean) {
      featuresStd = summarizer.variance.toArray.map(math.sqrt)
    }
    else {
      featuresStd = Array.fill(num_features)(1.0)
    }

    featuresStd
  }

  /**
    * return the L2 regularization value given a model
    * @param weight_array
    * @param regParam
    * @return
    */
  def L2reg(weight_array: Array[Double], regParam: Double): Double = {
    val size = weight_array.length
    var k = 0
    var result = 0.0
    while (k < size) {
      result += weight_array(k) * weight_array(k)
      k += 1
    }
    0.5 * regParam * result
  }

  /**
    * return the grddient, batch loss and batch size.
    * If spark.ml.debug=true, also compute the loss per epoch.
    * @param data
    * @param gradient
    * @param miniBatchFraction
    * @param weights
    * @param regParam
    * @param numExamples
    * @param iterationId
    * @return
    */
  def computeGradient(data: RDD[(Double, Vector)],
                      gradient: Gradient,
                      miniBatchFraction: Double,
                      weights: Vector,
                      regParam: Double,
                      numExamples: Int,
                      iterationId: Int
                     ): (Array[Double], Double, Double) = {
    val bcWeights = data.context.broadcast(weights)

    var time_cal_loss: Double = 0.0
    if (SparkEnv.get.conf.get("spark.ml.debug", "false").toBoolean) {
      val start_time = System.currentTimeMillis()
      val train_loss: Double = data.map(x => gradient.computeLoss(x._2, x._1, bcWeights.value))
        .reduce((x, y) => x + y)
      val end_time = System.currentTimeMillis()

      val reg_val = L2reg(bcWeights.value.toArray, regParam)
      time_cal_loss = (end_time - start_time) / 1000.0
      logInfo(s"ghand=Iteration:${iterationId}=TimeCalLoss:${time_cal_loss}")
      logInfo(s"ghandTrainLoss=Iteration:${iterationId}=trainLoss:${(train_loss) / numExamples + reg_val}")
    }
    train_end_time = System.currentTimeMillis()
    logInfo(s"ghand=Iteration:${iterationId}=" +
      s"TimeWithOutLoss:${(train_end_time - train_start_time) / 1000.0 - time_cal_loss}")

    val feature_dim: Int = weights.size
    train_start_time = System.currentTimeMillis()
    val (gradientSum, lossSum, miniBatchSize) = data.sample(false, miniBatchFraction, 42 + iterationId)
      .treeAggregate((BDV.zeros[Double](feature_dim), 0.0, 0L))(
        seqOp = (c, v) => {
          // c: (grad, loss, count), v: (label, features)
          val l = gradient.compute(v._2, v._1, bcWeights.value, Vectors.fromBreeze(c._1))
          (c._1, c._2 + l, c._3 + 1)
        },
        combOp = (c1, c2) => {
          // c: (grad, loss, count)
          (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
        }
      )
    bcWeights.destroy(blocking = false)
    (gradientSum.toArray, lossSum, miniBatchSize)
  }

  /**
    * Run stochastic gradient descent (SGD) in parallel using mini batches.
    * In each iteration, we sample a subset (fraction miniBatchFraction) of the total data
    * in order to compute a gradient estimate.
    * Sampling, and averaging the subgradients over this subset is performed using one standard
    * spark map-reduce in each iteration.
    *
    * @param data              Input data for SGD. RDD of the set of data examples, each of
    *                          the form (label, [feature values]).
    * @param gradient          Gradient object (used to compute the gradient of the loss function of
    *                          one single data example)
    * @param updater           Updater function to actually perform a gradient step in a given direction.
    * @param stepSize          initial step size for the first step
    * @param numIterations     number of iterations that SGD should be run.
    * @param regParam          regularization parameter
    * @param miniBatchFraction fraction of the input data set that should be used for
    *                          one iteration of SGD. Default value 1.0.
    * @param convergenceTol    Minibatch iteration will end before numIterations if the relative
    *                          difference between the current weight and the previous weight is less
    *                          than this value. In measuring convergence, L2 norm is calculated.
    *                          Default value 0.001. Must be between 0.0 and 1.0 inclusively.
    * @return A tuple containing two elements. The first element is a column matrix containing
    *         weights for every feature, and the second element is an array containing the
    *         stochastic loss computed for every iteration.
    */
  def runMiniBatchSGD(
                       data: RDD[(Double, Vector)],
                       gradient: Gradient,
                       updater: Updater,
                       stepSize: Double,
                       numIterations: Int,
                       regParam: Double,
                       miniBatchFraction: Double,
                       initialWeights: Vector,
                       convergenceTol: Double): (Vector, Array[Double]) = {

    val numExamples: Int = data.count().toInt
    val featuresStd: Array[Double] = getFeatureStd(data, initialWeights.size)
    var weights = initialWeights

    var i = 1
    train_start_time = System.currentTimeMillis() // initial value, will be updated
    while (i <= numIterations) {
      val (gradientSum, lossSum, miniBatchSize) = computeGradient(data, gradient, miniBatchFraction,
        weights, regParam, numExamples, i)

      if (miniBatchSize > 0) {
        val batch_loss = lossSum / miniBatchSize + L2reg(weights.toArray, regParam)
        logInfo(s"ghand=Iteration:${i}=trainBatchLoss=${batch_loss}")
        // we assume L2 regularization
        // weight = weight - stepSize * (gradient / scaleVector) - stepSize * regParam * (weight / scaleVector)
        val feature_dim = featuresStd.size
        var k = 0
        val weight_array = weights.toArray
        while (k < feature_dim) {
          if (featuresStd(k) != 0) {
            val tmp_grad = gradientSum(k) / miniBatchSize.toDouble + weight_array(k) * regParam
            weight_array(k) -= stepSize * tmp_grad / (featuresStd(k) * featuresStd(k))
          }
          k += 1
        }
        weights = Vectors.dense(weight_array)
      }

      i += 1
    }

    (weights, null)
  }

  def runMiniBatchMomentumSGD(
                               data: RDD[(Double, Vector)],
                               gradient: Gradient,
                               updater: Updater,
                               stepSize: Double,
                               numIterations: Int,
                               regParam: Double,
                               miniBatchFraction: Double,
                               initialWeights: Vector,
                               convergenceTol: Double): (Vector, Array[Double]) = {

    val numExamples: Int = data.count().toInt
    val featuresStd: Array[Double] = getFeatureStd(data, initialWeights.size)
    var weights = initialWeights
    var i = 1
    train_start_time = System.currentTimeMillis()
    val momentum = SparkEnv.get.conf.getDouble("spark.ml.sgd.momentum", 0.9)
    val velocity: Array[Double] = Array.fill(weights.size)(0)

    while (i <= numIterations) {
      val (gradientSum, lossSum, miniBatchSize) = computeGradient(data, gradient, miniBatchFraction,
        weights, regParam, numExamples, i)

      if (miniBatchSize > 0) {
        val batch_loss = lossSum / miniBatchSize + L2reg(weights.toArray, regParam)
        logInfo(s"ghand=Iteration:${i}=trainBatchLoss=${batch_loss}")

        // we assume L2 regularization
        // weight = weight - stepSize * (gradient / scaleVector) - stepSize * regParam * (weight / scaleVector)
        val feature_dim = featuresStd.size
        var k = 0
        val weight_array = weights.toArray
        while (k < velocity.length) {
          velocity(k) = velocity(k) * momentum +
            stepSize * (gradientSum(k) / miniBatchSize + weight_array(k) * regParam)
          k += 1
        }

        k = 0
        while (k < feature_dim) {
          if (featuresStd(k) != 0) {
            weight_array(k) -= velocity(k) / (featuresStd(k) * featuresStd(k))
          }
          k += 1
        }
        weights = Vectors.dense(weight_array)
      }

      i += 1
    }

    (weights, null)
  }


  def runMiniBatchRMSPropSGD(
                              data: RDD[(Double, Vector)],
                              gradient: Gradient,
                              updater: Updater,
                              stepSize: Double,
                              numIterations: Int,
                              regParam: Double,
                              miniBatchFraction: Double,
                              initialWeights: Vector,
                              convergenceTol: Double): (Vector, Array[Double]) = {

    val numExamples: Int = data.count().toInt
    val featuresStd: Array[Double] = getFeatureStd(data, initialWeights.size)
    var weights = initialWeights
    var i = 1
    train_start_time = System.currentTimeMillis()
    val epsilon = SparkEnv.get.conf.getDouble("spark.ml.sgd.rmsprop.epsilon", 1e-7)
    val forget = SparkEnv.get.conf.getDouble("spark.ml.sgd.rmsprop.forget", 0.9)
    val expectation_g2: Array[Double] = Array.fill(weights.size)(0)

    while (i <= numIterations) {

      val (gradientSum, lossSum, miniBatchSize) = computeGradient(data, gradient, miniBatchFraction,
        weights, regParam, numExamples, i)

      if (miniBatchSize > 0) {
        val batch_loss = lossSum / miniBatchSize + L2reg(weights.toArray, regParam)
        logInfo(s"ghand=Iteration:${i}=trainBatchLoss=${batch_loss}")
        // we assume L2 regularization
        // weight = weight - stepSize * (gradient / scaleVector) - stepSize * regParam * (weight / scaleVector)
        val feature_dim = featuresStd.size
        var k = 0
        val weight_array = weights.toArray
        while (k < expectation_g2.length) {
          expectation_g2(k) = expectation_g2(k) * forget +
              (1 - forget) * math.pow(gradientSum(k) / miniBatchSize + weight_array(k) * regParam, 2)
          k += 1
        }

        k = 0
        while (k < feature_dim) {
          if (featuresStd(k) != 0) {
            weight_array(k) -=
              stepSize / math.sqrt(epsilon + expectation_g2(k)) * gradientSum(k) / miniBatchSize
          }
          k += 1
        }
        weights = Vectors.dense(weight_array)
      }

      i += 1
    }

    (weights, null)
  }


  def runMiniBatchAdamSGD(
                           data: RDD[(Double, Vector)],
                           gradient: Gradient,
                           updater: Updater,
                           stepSize: Double,
                           numIterations: Int,
                           regParam: Double,
                           miniBatchFraction: Double,
                           initialWeights: Vector,
                           convergenceTol: Double): (Vector, Array[Double]) = {

    val numExamples: Int = data.count().toInt
    var weights = initialWeights
    val featuresStd: Array[Double] = getFeatureStd(data, weights.size)
    train_start_time = System.currentTimeMillis()
    val epsilon = SparkEnv.get.conf.getDouble("spark.ml.sgd.adam.epsilon", 1e-7)
    val beta1 = SparkEnv.get.conf.getDouble("spark.ml.sgd.adam.beta1", 0.9)
    val beta2 = SparkEnv.get.conf.getDouble("spark.ml.sgd.adam.beta2", 0.99)
    val expectation_g2: Array[Double] = Array.fill(weights.size)(0)
    val velocity: Array[Double] = Array.fill(weights.size)(0)
    var i = 1
    var k = 0
    while (i <= numIterations) {

      val (gradientSum, lossSum, miniBatchSize) = computeGradient(data, gradient, miniBatchFraction,
        weights, regParam, numExamples, i)

      if (miniBatchSize > 0) {
        val batch_loss = lossSum / miniBatchSize + L2reg(weights.toArray, regParam)
        logInfo(s"ghand=Iteration:${i}=trainBatchLoss=${batch_loss}")

        // we assume L2 regularization
        // weight = weight - stepSize * (gradient / scaleVector) - stepSize * regParam * (weight / scaleVector)
        val feature_dim = featuresStd.size
        val weight_array = weights.toArray
        k = 0
        while (k < velocity.length) {
          velocity(k) =
            beta1 * velocity(k) + (1 - beta1) * (gradientSum(k) / miniBatchSize + weight_array(k) * regParam)
          expectation_g2(k) =
            beta2 * expectation_g2(k) + (1 - beta2) * math.pow(gradientSum(k) / miniBatchSize + weight_array(k) * regParam, 2)
          k += 1
        }

        val power_beta1 = 1 - math.pow(beta1, i)
        val power_beta2 = 1 - math.pow(beta2, i)
        k = 0
        while (k < feature_dim) {
          if (featuresStd(k) != 0) {
            weight_array(k) -=
              stepSize / math.sqrt(epsilon + expectation_g2(k) / power_beta2) * velocity(k) / power_beta1
          }
          k += 1
        }
        weights = Vectors.dense(weight_array)
      }

      i += 1
    }

    (weights, null)
  }


  def runMiniBatchAMSGradSGD(
                           data: RDD[(Double, Vector)],
                           gradient: Gradient,
                           updater: Updater,
                           stepSize: Double,
                           numIterations: Int,
                           regParam: Double,
                           miniBatchFraction: Double,
                           initialWeights: Vector,
                           convergenceTol: Double): (Vector, Array[Double]) = {

    val numExamples: Int = data.count().toInt
    val featuresStd: Array[Double] = getFeatureStd(data, initialWeights.size)
    var weights = initialWeights
    var i = 1
    train_start_time = System.currentTimeMillis()
    val epsilon = SparkEnv.get.conf.getDouble("spark.ml.sgd.adam.epsilon", 1e-7)
    val beta1 = SparkEnv.get.conf.getDouble("spark.ml.sgd.adam.beta1", 0.9)
    val beta2 = SparkEnv.get.conf.getDouble("spark.ml.sgd.adam.beta2", 0.99)
    val expectation_g2: Array[Double] = Array.fill(weights.size)(0)
    val velocity: Array[Double] = Array.fill(weights.size)(0)
    val expectation_g2_head: Array[Double] = Array.fill(weights.size)(0)

    while (i <= numIterations) {

      val (gradientSum, lossSum, miniBatchSize) = computeGradient(data, gradient, miniBatchFraction,
        weights, regParam, numExamples, i)

      if (miniBatchSize > 0) {
        val batch_loss = lossSum / miniBatchSize + L2reg(weights.toArray, regParam)
        logInfo(s"ghand=Iteration:${i}=trainBatchLoss=${batch_loss}")

        // we assume L2 regularization
        // weight = weight - stepSize * (gradient / scaleVector) - stepSize * regParam * (weight / scaleVector)
        val feature_dim = featuresStd.size
        var k = 0
        val weight_array = weights.toArray
        while (k < velocity.length) {
          velocity(k) = beta1 * velocity(k) + (1 - beta1) * (gradientSum(k) / miniBatchSize + weight_array(k) * regParam)
          expectation_g2(k) = beta2 * expectation_g2(k) +(1 - beta2) * math.pow(gradientSum(k) / miniBatchSize + weight_array(k) * regParam, 2)
          expectation_g2_head(k) = math.max(expectation_g2(k), expectation_g2_head(k))
          k += 1
        }

        k = 0
        val power_beta1 = 1 - math.pow(beta1, i)
        val power_beta2 = 1 - math.pow(beta2, i)
        while (k < feature_dim) {
          if (featuresStd(k) != 0) {
            weight_array(k) -= stepSize / math.sqrt(epsilon + expectation_g2_head(k) /
              power_beta2) * velocity(k) / power_beta1
          }
          k += 1
        }
        weights = Vectors.dense(weight_array)
      }

      i += 1
    }

    (weights, null)
  }

  var train_start_time: Long = 0
  var train_end_time: Long = 0
}

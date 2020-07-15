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

package org.apache.spark.mllib.classification

import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD

/**
 * Train a classification model for Binary Logistic Regression
 * using Stochastic Gradient Descent. By default L2 regularization is used,
 * which can be changed via `LogisticRegressionWithSGD.optimizer`.
 *
 * Using [[LogisticRegressionWithLBFGS]] is recommended over this.
 *
 * @note Labels used in Logistic Regression should be {0, 1, ..., k - 1}
 * for k classes multi-label classification problem.
 */
class GhandLogRSGDShuffleModel private[mllib] (
    private var stepSize: Double,
    private var numIterations: Int,
    private var regParam: Double,
    private var miniBatchFraction: Double)
  extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {

  private val gradient = new LogisticGradient()
  private val updater = new SquaredL2Updater()
  @Since("0.8.0")
  override val optimizer = new GhandLogRSGDShuffle(gradient, updater)
    .setStepSize(stepSize)
    .setNumIterations(numIterations)
    .setRegParam(regParam)
    .setMiniBatchFraction(miniBatchFraction)
  override protected val validators = List(DataValidators.binaryLabelValidator)

  /**
   * Construct a LogisticRegression object with default parameters: {stepSize: 1.0,
   * numIterations: 100, regParm: 0.01, miniBatchFraction: 1.0}.
   */
  @Since("0.8.0")
  @deprecated("Use ml.classification.LogisticRegression or LogisticRegressionWithLBFGS", "2.0.0")
  def this() = this(1.0, 100, 0.01, 1.0)

  override protected[mllib] def createModel(weights: Vector, intercept: Double) = {
    new LogisticRegressionModel(weights, intercept)
  }
}

/**
 * Top-level methods for calling Logistic Regression using Stochastic Gradient Descent.
 *
 * @note Labels used in Logistic Regression should be {0, 1}
 */
object GhandLogRSGDShuffleModel {
  // NOTE(shivaram): We use multiple train methods instead of default arguments to support
  // Java programs.

  @Since("1.0.0")
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             regParam: Double,
             miniBatchFraction: Double): LogisticRegressionModel = {
    new GhandLogRSGDShuffleModel(stepSize, numIterations, regParam, miniBatchFraction)
      .run(input)
  }


  @Since("1.0.0")
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             regParam: Double,
             miniBatchFraction: Double,
             initialWeights: Vector): LogisticRegressionModel = {
    new GhandLogRSGDShuffleModel(stepSize, numIterations, regParam, miniBatchFraction)
      .run(input, initialWeights)
  }
  /**
   * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
   * number of iterations of gradient descent using the specified step size. Each iteration uses
   * `miniBatchFraction` fraction of the data to calculate the gradient. The weights used in
   * gradient descent are initialized using the initial weights provided.
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @param stepSize Step size to be used for each iteration of gradient descent.
   * @param miniBatchFraction Fraction of data to be used per iteration.
   * @param initialWeights Initial set of weights to be used. Array should be equal in size to
   *        the number of features in the data.
   *
   * @note Labels used in Logistic Regression should be {0, 1}
   */
  @Since("1.0.0")
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int,
      stepSize: Double,
      miniBatchFraction: Double,
      initialWeights: Vector): LogisticRegressionModel = {
    new GhandLogRSGDShuffleModel(stepSize, numIterations, 0.0, miniBatchFraction)
      .run(input, initialWeights)
  }

  /**
   * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
   * number of iterations of gradient descent using the specified step size. Each iteration uses
   * `miniBatchFraction` fraction of the data to calculate the gradient.
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @param stepSize Step size to be used for each iteration of gradient descent.
   * @param miniBatchFraction Fraction of data to be used per iteration.
   *
   * @note Labels used in Logistic Regression should be {0, 1}
   */
  @Since("1.0.0")
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int,
      stepSize: Double,
      miniBatchFraction: Double): LogisticRegressionModel = {
    new GhandLogRSGDShuffleModel(stepSize, numIterations, 0.0, miniBatchFraction)
      .run(input)
  }

  /**
   * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
   * number of iterations of gradient descent using the specified step size. We use the entire data
   * set to update the gradient in each iteration.
   *
   * @param input RDD of (label, array of features) pairs.
   * @param stepSize Step size to be used for each iteration of Gradient Descent.
   * @param numIterations Number of iterations of gradient descent to run.
   * @return a LogisticRegressionModel which has the weights and offset from training.
   *
   * @note Labels used in Logistic Regression should be {0, 1}
   */
  @Since("1.0.0")
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int,
      stepSize: Double): LogisticRegressionModel = {
    train(input, numIterations, stepSize, 1.0)
  }

  /**
   * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
   * number of iterations of gradient descent using a step size of 1.0. We use the entire data set
   * to update the gradient in each iteration.
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @return a LogisticRegressionModel which has the weights and offset from training.
   *
   * @note Labels used in Logistic Regression should be {0, 1}
   */
  @Since("1.0.0")
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int): LogisticRegressionModel = {
    train(input, numIterations, 1.0, 1.0)
  }
}

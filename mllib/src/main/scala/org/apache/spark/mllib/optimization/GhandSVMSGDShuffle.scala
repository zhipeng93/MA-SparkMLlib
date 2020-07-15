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

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import breeze.linalg.{norm => brzNorm}
import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.{HashPartitioner, TaskContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.SparkEnv
import scala.util.Random

/**
  * Class used to solve an optimization problem using Gradient Descent.
  *
  * @param gradient Gradient function to be used.
  * @param updater Updater to be used to update weights after every iteration.
  */

class GhandSVMSGDShuffle private[spark] (private var gradient: Gradient, private var updater: Updater)
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
    *  - If the norm of the new solution vector is greater than 1, the diff of solution vectors
    *    is compared to relative tolerance which means normalizing by the norm of
    *    the new solution vector.
    *  - If the norm of the new solution vector is less than or equal to 1, the diff of solution
    *    vectors is compared to absolute tolerance which is not normalizing.
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
    * @param data training data
    * @param initialWeights initial weights
    * @return solution vector
    */
  @DeveloperApi
  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = GhandSVMSGDShuffle.runMiniBatchSGD(
      data,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      miniBatchFraction,
      initialWeights,
      convergenceTol)
    weights
  }

}

/**
  * :: DeveloperApi ::
  * Top-level method to run gradient descent.
  */
@DeveloperApi
object GhandSVMSGDShuffle extends Logging {
//  private var last_model_norm: Double = Double.MaxValue
//  private var new_model_norm: Double = 0
  /**
    * Run stochastic gradient descent (SGD) in parallel using mini batches.
    * In each iteration, we sample a subset (fraction miniBatchFraction) of the total data
    * in order to compute a gradient estimate.
    * Sampling, and averaging the subgradients over this subset is performed using one standard
    * spark map-reduce in each iteration.
    *
    * @param data Input data for SGD. RDD of the set of data examples, each of
    *             the form (label, [feature values]).
    * @param gradient Gradient object (used to compute the gradient of the loss function of
    *                 one single data example)
    * @param updater Updater function to actually perform a gradient step in a given direction.
    * @param stepSize initial step size for the first step
    * @param numIterations number of iterations that SGD should be run.
    * @param regParam regularization parameter
    * @param miniBatchFraction fraction of the input data set that should be used for
    *                          one iteration of SGD. Default value 1.0.
    * @param convergenceTol Minibatch iteration will end before numIterations if the relative
    *                       difference between the current weight and the previous weight is less
    *                       than this value. In measuring convergence, L2 norm is calculated.
    *                       Default value 0.001. Must be between 0.0 and 1.0 inclusively.
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

    // convergenceTol should be set with non minibatch settings
    if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
      logWarning("Testing against a convergenceTol when using miniBatchFraction " +
        "< 1.0 can be unstable because of the stochasticity in sampling.")
    }
    if (numIterations * miniBatchFraction < 1.0) {
      logWarning("Not all examples will be used if numIterations * miniBatchFraction < 1.0: " +
        s"numIterations=$numIterations and miniBatchFraction=$miniBatchFraction")
    }
    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    // Record previous weight and current one to calculate solution vector difference
    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    if (numExamples * miniBatchFraction < 1) {
      logWarning("The miniBatchFraction is too small")
    }

    // Initialize weights as a column vector
    val initial_weights: DenseVector = Vectors.dense(initialWeights.toArray).toDense
    val num_features = initial_weights.size
    val bcWeights: Broadcast[DenseVector] = data.context.broadcast(initial_weights)
    // first run: broadcast the variable to the cluster.
    // the first model cannot be paralleize from a big Seq, since it maybe to big for the driver.
    val model: DenseVector = repeatMLStart(data, bcWeights, numIterations, stepSize, regParam, num_features, numExamples)

    bcWeights.destroy()

    (model, stochasticLossHistory.toArray)
  }

  /**
    *
    * @param initModelRDD The initial model RDD[U], where each partition has only one element, i.e.,
    *                     the parameter of the model, which is a breeze vector
    * @param numIter Number of iterations to perform the training
    * @return return model in the DenseVector
    */
  def repeatML(dataRDD: RDD[(Double, Vector)], initModelRDD: RDD[DenseVector],
               numIter: Int, stepSize: Double, regParam: Double, numFeatures: Int, numExamples: Long): DenseVector = {

    var time_cal_loss: Double = 0.0
    if(SparkEnv.get.conf.get("spark.ml.debug", "false").toBoolean){
      // do not take it back, calculate loss with data locality and return the loss.
      val calLoss = (itd: Iterator[(Double, Vector)], itm: Iterator[DenseVector]) => {
        val start_time = System.currentTimeMillis()
        val localModel: DenseVector = itm.next()

        val loss = itd.foldLeft((0.0))(
          (startLoss, datapoint) =>{
            val fea = datapoint._2
            val label = datapoint._1
            val tmp_loss = 1 - dot(fea, localModel) * (2 * label - 1)
            if (tmp_loss > 0)
              startLoss + tmp_loss
            else
              startLoss
          }
        )
        var weight_norm: Double = 0
        if(TaskContext.getPartitionId() == 0){
          weight_norm = brzNorm(localModel.asBreeze, 2)
        }
        val end_time = System.currentTimeMillis()
        Iterator((loss, weight_norm, (end_time - start_time) / 1000.0, 1))
      }

      // loss: sumOfLoss, weightNorm, calLossTime, numOfPartition
      val loss: (Double, Double, Double, Int) = dataRDD.zipPartitions(initModelRDD, preservesPartitioning = false)(calLoss).reduce(
        (a, b) => {
          (a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4)
        }
      )

      time_cal_loss = loss._3 / loss._4
      logInfo(s"ghand=Iteration:${numIter}=TimeCalLoss:${time_cal_loss}")
      logInfo(s"ghandTrainLoss=weightNorm:${loss._2}=" +
        s"trainLoss:${(loss._1) / numExamples + 0.5 * loss._2 * loss._2 * regParam}")
    }
    train_end_time = System.currentTimeMillis()
    logInfo(s"ghand=Iteration:${numIter}=TimeWithOutLoss:${(train_end_time - train_start_time) / 1000.0 - time_cal_loss}")
    train_start_time = System.currentTimeMillis()


    if (numIter > 0) {
      // zip the two RDDs
      val models: RDD[DenseVector] = updateModel(dataRDD, initModelRDD, stepSize, regParam)
      // models has numPartitions partitions, each partition with one Iterator,
      // and the Iterator has only one element, U, which is the local model.

      val averagedModels: RDD[DenseVector] = allReduce(models, numFeatures)
      // here exchange the model, do something like a all-reduce, to communicate the model,
      // the result is return a RDD with numPartitions elements, each element is the parameters
      // of the model
      // BUT: all the elements may not have exactly the same value, it depends on your all-reduce.
      repeatML(dataRDD, averagedModels, numIter - 1, stepSize, regParam, numFeatures, numExamples)
    }
    else {
      initModelRDD.take(1)(0)
    }

  }

  def repeatMLStart(dataRDD: RDD[(Double, Vector)], bcWeights: Broadcast[DenseVector],
                    numIter: Int, stepSize: Double, regParam: Double, numFeatures: Int, numExamples: Long): DenseVector = {
    if (numIter > 0) {
      train_start_time = System.currentTimeMillis()
      // first link the two RDD via the partition Index, then perform SeqOp on each data point
      // and the corresponding model
        val models: RDD[DenseVector] = updateModelStart(dataRDD, bcWeights, stepSize, regParam)
      // models has numPartitions partitions, each partition with one Iterator,
      // and the Iterator has only one element, U, which is the local model.
      val averagedModels: RDD[DenseVector] = allReduce(models, numFeatures)
      // here exchange the model, do something like a all-reduce, to communicate the model,
      // the result is return a RDD with numPartitions elements, each element is the parameters
      // of the model, and all the elements share exactly the same value.
      repeatML(dataRDD, averagedModels, numIter - 1, stepSize, regParam, numFeatures, numExamples)
    }
    else {
      bcWeights.value
    }

  }

  /**
    * @param dataRDD
    * @param initModelRDD modelRDD, each partition has exactly one element, which is the full model.
    * @return the newModelRDD
    */
  def updateModel(dataRDD: RDD[(Double, Vector)], initModelRDD: RDD[DenseVector], stepSize: Double, regParam: Double): RDD[DenseVector] = {
    //apply data from itt to the model itu. Note that itu only has one element, that is the model.
    val aggregateFunction =
      (itd: Iterator[(Double, Vector)], itm: Iterator[DenseVector]) => {
        val localModel: DenseVector = itm.next()

        if(!SparkEnv.get.conf.get("spark.ml.straggler", "false").toBoolean) {
          if (TaskContext.getPartitionId() == 1 || TaskContext.getPartitionId() == 0) {
            if (localModel.size < 10000000) {
              Thread.sleep(Random.nextInt(4) * 1000)
            }
            else if (localModel.size < 30000000) {
              Thread.sleep(Random.nextInt(20) * 1000)
            }
            else if (localModel.size < 60000000) {
              Thread.sleep(Random.nextInt(60) * 1000)
            }
          }
        }

        val startPoint: (DenseVector, Double) = (localModel, 1.0) // for L2 sparseUpdate
        // when the factor c is too small, will update the model in a dense way.
        val newModel = itd.foldLeft(startPoint)(
          (sparseL2Model, dataPoint) => {
            var c_t: Double = sparseL2Model._2
            val model: DenseVector = sparseL2Model._1
            if(c_t < 1e-5){ // magic number to avoid numerical issue
            val startDenseUpdateModel = System.currentTimeMillis()
              scal(c_t, model)
              c_t = 1.0
              logInfo(s"ghand=SS=L2UpdateDenseModel:${System.currentTimeMillis() - startDenseUpdateModel}")
            }

            val thisIterStepSize = stepSize
            val transStepSize = thisIterStepSize / (1 - thisIterStepSize * regParam) / c_t
            val dotProduct = dot(dataPoint._2, model) * c_t // because model is not the real model here.
            val labelScaled = 2 * dataPoint._1 - 1.0
            val local_loss = if (1.0 > labelScaled * dotProduct) {
              axpy((-labelScaled) * (-transStepSize), dataPoint._2, model)
              1.0 - labelScaled * dotProduct
            } else {
              0
            }

            (model, c_t * (1 - thisIterStepSize * regParam))
          }
        )
        val realModel: DenseVector = newModel._1
        scal(newModel._2, realModel)
        Iterator(realModel)
    }

    dataRDD.zipPartitions(initModelRDD, preservesPartitioning = false)(aggregateFunction)
  }

  def updateModelStart(dataRDD: RDD[(Double, Vector)], bcWeights: Broadcast[DenseVector],
                       stepSize: Double, regParam: Double): RDD[DenseVector] = {
    //apply data from itt to the model itu. Note that itu only has one element, that is the model.
    val mapPartitionsFunc =
      (itd: Iterator[(Double, Vector)]) => {
        // so many transformations, caused by implementation, i.e., polymorphic
        val localModel: DenseVector = bcWeights.value.toDense.copy // if you don't copy it, then it's hogWild!
        // have to specify it as copy when debugging, because when debugging, it will cause multiple iterations
        // over the data
        // conform the result to be a iterator of model
        if(!SparkEnv.get.conf.get("spark.ml.straggler", "false").toBoolean) {
          if (TaskContext.getPartitionId() == 1 || TaskContext.getPartitionId() == 0) {
            if (localModel.size < 10000000) {
              Thread.sleep(Random.nextInt(4) * 1000)
            }
            else if (localModel.size < 30000000) {
              Thread.sleep(Random.nextInt(20) * 1000)
            }
            else if (localModel.size < 60000000) {
              Thread.sleep(Random.nextInt(60) * 1000)
            }
          }
        }
        val startPoint: (DenseVector, Double) = (localModel, 1.0) // for L2 sparseUpdate
        // when the factor c is too small, will update the model in a dense way.
        val newModel = itd.foldLeft(startPoint)(
            (sparseL2Model, dataPoint) => {
              var c_t: Double = sparseL2Model._2
              val model: DenseVector = sparseL2Model._1
              if(c_t < 1e-5){ // magic number to avoid numerical issue
                val startDenseUpdateModel = System.currentTimeMillis()
                scal(c_t, model)
                c_t = 1.0
                logInfo(s"ghand=SS=L2UpdateDenseModel:${System.currentTimeMillis() - startDenseUpdateModel}")
              }

              val thisIterStepSize = stepSize
              val transStepSize = thisIterStepSize / (1 - thisIterStepSize * regParam) / c_t
              val dotProduct = dot(dataPoint._2, model) * c_t // because model is not the real model here.
              val labelScaled = 2 * dataPoint._1 - 1.0
              val local_loss = if (1.0 > labelScaled * dotProduct) {
                axpy((-labelScaled) * (-transStepSize), dataPoint._2, model)
                1.0 - labelScaled * dotProduct
              } else {
                0
              }
              (model, c_t * (1 - thisIterStepSize * regParam))
            }
          )
        val realModel: DenseVector = newModel._1
        scal(newModel._2, realModel)
        Iterator(realModel)
        // convert to one iterator with only one element
    }

    dataRDD.mapPartitions(it => mapPartitionsFunc(it), preservesPartitioning = false)
  }


  def allReduce(models: RDD[DenseVector], numFeatures: Int): RDD[DenseVector] = {
    // implement the real reduce, shuffle 1/numPartitions part of each model, and then shuffle back
    // the total number of communication is 2 * numPartitions * model_size.
    val numPartion = models.getNumPartitions
    // transform each dense vector into #numPartitions parts, with key from [0, numPartitions)
    val slicesModel: RDD[(Int, Array[Double])] = models.map {
      dv => {
        partitionDenseVector(dv.toArray, numPartion)
      }
    }.flatMap(array => array.iterator)

    // 2. shuffle by key, the number of partition should be the same with the initial modelRDD
    // concate the model, get the RDD[Double], each partition is a collection of double
    val reducedSlicesModel: RDD[(Int, Array[Double])] = slicesModel
      .reduceByKey(new HashPartitioner(numPartion), {
          (x, y) =>
            var i = 0
            while(i < x.length){
              x(i) += y(i)
              i += 1
            }
            x
        }
      )

    // (Int-for shuffleKey, (Int-sequenceId-in-the-model, Array[Double]))
    val forShuffleModel: RDD[(Int, (Int, Array[Double]))] = reducedSlicesModel.map {
      x => {
        // duplicate elements for shuffling
        //x: (sliceId, sliceModel)
        val x_average: (Int, Array[Double]) = (x._1, scalArray(x._2, numPartion.toDouble)) // model average
        val array = new Array[(Int, (Int, Array[Double]))](numPartion)
        (0 to numPartion - 1).map(i => array(i) = (i, x_average))
        array
      }
    }.flatMap(array => array.iterator)

    // return value of groupByKey is [Key, Iterable[V]]
    val slicesRealModel: RDD[(Int, Iterable[(Int, Array[Double])])] = forShuffleModel.groupByKey(new HashPartitioner(numPartion))

    val newModels: RDD[DenseVector] =  slicesRealModel.map {
      x =>
        constructDenseVector(x._2, numFeatures, numPartion)
    }

    // reduce RDDs that will not be useful in the future
    reducedSlicesModel.unpersist(blocking = false) // maybe blocking is not the one I thought

    newModels
  }

  def scalArray(array: Array[Double], numPartition: Double): Array[Double] = {
    var i = 0
    while(i < array.length){
      array(i) = array(i) / numPartition
      i += 1
    }
    array
  }

  def constructDenseVector(iterable: Iterable[(Int, Array[Double])], numFeatures: Int, numPartition: Int): DenseVector = {
    val array: Array[Double] = new Array[Double](numFeatures)
    val averge_len = numFeatures / numPartition
    var startId = 0
    var endId = 0

    val iter: Iterator[(Int, Array[Double])] = iterable.iterator

    while(iter.hasNext){
      val tmp = iter.next()
      val sliceId: Int = tmp._1
      val slice: Array[Double] = tmp._2
      // construct the new array
      startId = sliceId * averge_len
      endId = slice.length + startId

      (startId to endId - 1).foreach(x => array(x) = slice(x - startId))

    }
    val result = Vectors.dense(array).toDense

    result
  }

  def partitionDenseVector(array: Array[Double], numPartition: Int) : Array[(Int, Array[Double])] = {
    val len_total: Int = array.length
    val result: Array[(Int, Array[Double])] = new Array[(Int, Array[Double])](numPartition)
    var i: Int = 0
    val average_len = len_total / numPartition

    while(i < numPartition - 1){
      val x: Array[Double] = array.slice(i * average_len, i * average_len + average_len)
      result(i) = (i, x)
      i += 1
    }
    result(numPartition - 1) = (numPartition - 1, array.slice((numPartition - 1) * average_len, len_total))

    result
  }

  /**
    * Alias of `runMiniBatchSGD` with convergenceTol set to default value of 0.001.
    */
  def runMiniBatchSGD(
                       data: RDD[(Double, Vector)],
                       gradient: Gradient,
                       updater: Updater,
                       stepSize: Double,
                       numIterations: Int,
                       regParam: Double,
                       miniBatchFraction: Double,
                       initialWeights: Vector): (Vector, Array[Double]) =
    GhandSVMSGDShuffle.runMiniBatchSGD(data, gradient, updater, stepSize, numIterations,
      regParam, miniBatchFraction, initialWeights, 0.001)


  private def isConverged(
                           previousWeights: Vector,
                           currentWeights: Vector,
                           convergenceTol: Double): Boolean = {
    // To compare with convergence tolerance.
    val previousBDV = previousWeights.asBreeze.toDenseVector
    val currentBDV = currentWeights.asBreeze.toDenseVector

    // This represents the difference of updated weights in the iteration.
    val solutionVecDiff: Double = brzNorm(previousBDV - currentBDV)

    solutionVecDiff < convergenceTol * Math.max(brzNorm(currentBDV), 1.0)
  }

  var train_start_time: Long = 0
  var train_end_time: Long = 0

}

# MA-SparkMLlib
- An model averaging and all reduce implementation on Spark MLlib.
- Users can simply replace spark/mllib with the mllib code here and run `mvn clean package -Dmaven.test.skip=true`
- Note: Users need to write their own code to call methods, like `mllib/src/main/scala/org/apache/spark/mllib/classification/GhandSVMSGDShuffleModel.scala` 

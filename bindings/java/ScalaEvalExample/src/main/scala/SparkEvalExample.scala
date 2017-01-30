import java.awt.{Color, Image}
import java.awt.image.BufferedImage
import javax.imageio.ImageIO

import org.apache.spark.SparkFiles
import org.apache.spark.input.PortableDataStream
import org.apache.spark.sql.SparkSession

import Utils._

object SparkEvalExample extends App {
  System.loadLibrary("CNTKJava")
  val spark =
    SparkSession.builder().master("local[*]").appName("Spark SQL basic example").getOrCreate()
  val sc = spark.sparkContext
  sc.setLogLevel("WARN")

  val outputName = "Plus2060"
  val dataPath   = "/home/ratan/Downloads"
  sc.addFile(s"$dataPath/z.model")

  def eval(model: Function)(img: Seq[Float]): Seq[Float] = {
    val inputVar   = model.getArguments.get(0)
    val inputShape = inputVar.GetShape

    val floatVec    = img.foldLeft(new FloatVector()) { case (fv, f) => fv.add(f); fv }
    val floatVecVec = new FloatVectorVector()
    floatVecVec.add(floatVec)

    val inputVal =
      Value.CreateDenseFloat(inputShape, floatVecVec, DeviceDescriptor.GetCPUDevice)
    val inputDataMap = new UnorderedMapVariableValuePtr()
    inputDataMap.Add(inputVar, inputVal)

    val outputDataMap = new UnorderedMapVariableValuePtr()
    val outputVar     = model.getOutputs.get(0)
    outputDataMap.Add(outputVar, null)
    model.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.GetCPUDevice)

    val outputBuffer = new FloatVectorVector()
    outputDataMap.getitem(outputVar).CopyVariableValueToFloat(outputVar, outputBuffer)
    seqify(outputBuffer).head
  }

  sc.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive", "true")
  val images = sc.binaryFiles(s"$dataPath/grocery")

  def applyModel(modelName: String)(
      iter: Iterator[(String, PortableDataStream)]): Iterator[(String, Array[Float])] = {

    val model = Function.LoadModel(SparkFiles.get(modelName))

    val inputVar    = model.getArguments.get(0)
    val inputShape  = inputVar.GetShape
    val imageWidth  = inputShape.getDimensions.get(0).toInt
    val imageHeight = inputShape.getDimensions.get(1).toInt

    for (input <- iter) yield {
      val (filename, stream) = input

      val bmp     = ImageIO.read(stream.open)
      val resized = bmp.getScaledInstance(imageWidth, imageHeight, Image.SCALE_DEFAULT)
      val bImg = new BufferedImage(resized.getWidth(null),
                                   resized.getHeight(null),
                                   BufferedImage.TYPE_INT_RGB)
      bImg.getGraphics.drawImage(resized, 0, 0, null)

      val image = preprocessImage(bImg)

      (filename, eval(model)(image).toArray)
    }
  }

  val processed = images.mapPartitions(applyModel("z.model"))
  processed.collect.foreach {
    case (filename, data) =>
      println(filename)
      data.foreach(x => print(s"$x "))
      println(); println()
  }

  println(s"Evaluated ${processed.count} images")

  spark.stop()
}

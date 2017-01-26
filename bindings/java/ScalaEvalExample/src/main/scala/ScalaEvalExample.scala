/**
  * Created by ratan on 1/11/17.
  */
import javax.imageio.ImageIO
import java.awt.image._
import java.awt.Image
import java.awt.Color
import java.io.File
import scala.collection.JavaConversions._

object ScalaEvalExample {//extends App {
  System.loadLibrary("CNTKLib")

  val outputName = "Plus2060"
  val dataPath   = "/home/ratan/Downloads/"
  val modelFunc  = Function.LoadModel(dataPath + "z.model")
  val outputVar  = modelFunc.getOutputs.get(0)
  val inputVar   = modelFunc.getArguments.get(0)

  val inputShape    = inputVar.GetShape
  val imageWidth    = inputShape.getDimensions.get(0).toInt
  val imageHeight   = inputShape.getDimensions.get(1).toInt
  val imageChannels = inputShape.getDimensions.get(2).toInt
  val imageSize     = inputShape.GetTotalSize

  println("Evaluate single image")

  // Image preprocessing to match input requirements of the model.
  val bmp     = ImageIO.read(new File(dataPath + "00000.png"))
  val resized = bmp.getScaledInstance(imageWidth, imageHeight, Image.SCALE_DEFAULT)
  val bImg =
    new BufferedImage(resized.getWidth(null), resized.getHeight(null), BufferedImage.TYPE_INT_RGB)
  // or use any other fitting type
  bImg.getGraphics.drawImage(resized, 0, 0, null)

  val resizedCHW = for {
    c <- 0 until 3
    h <- 0 until bImg.getHeight()
    w <- 0 until bImg.getWidth()
    color = new Color(bImg.getRGB(w, h))
    intensity = if (c == 0) {
      color.getBlue
    } else if (c == 1) {
      color.getGreen
    } else {
      color.getRed
    }
  } yield intensity

  // TODO: write this comment
  val floatVec    = resizedCHW.foldLeft(new FloatVector()) { case (fv, f) => fv.add(f); fv }
  val floatVecVec = new FloatVectorVector()
  floatVecVec.add(floatVec)
  // Create input data map
  val inputVal     = Value.CreateDenseFloat(inputShape, floatVecVec, DeviceDescriptor.GetCPUDevice)
  val inputDataMap = new UnorderedMapVariableValuePtr()
  inputDataMap.Add(inputVar, inputVal)

  // Create output data map. Using null as Value to indicate using system allocated memory.
  // Alternatively, create a Value object and add it to the data map.
  val outputDataMap = new UnorderedMapVariableValuePtr()
  outputDataMap.Add(outputVar, null)

  // Start evaluation on the device
  modelFunc.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.GetCPUDevice)

  // Get evaluate result as dense output
  val outputBuffer = new FloatVectorVector()
  outputDataMap.getitem(outputVar).CopyVariableValueToFloat(outputVar, outputBuffer)
  for (i <- 0 until outputBuffer.get(0).size().toInt) {
    println(outputBuffer.get(0).get(i.toInt))
  }

}

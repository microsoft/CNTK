import com.microsoft.CNTK.*;
import org.junit.Assert;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import static org.junit.Assert.*;

public class JavaBindingsTest {

    @Test
    public void testEvalManagedEvaluateNoInput()    {
        String modelDefinition = "precision = \"float\"" +
                "traceLevel = 1" +
                " run=NDLNetworkBuilder" +
                "NDLNetworkBuilder=[" +
                "v1 = Constant(1)" +
                "v2 = Constant(2)" +
                " o1 = Plus(v1, v2, tag=\"output\")" +
                " FeatureNodes = (v1)" +
                "]";
        //model = new IEvaluateModelManagedF()
        /*
        {
            model.CreateNetwork(modelDefinition);

            var inDims = model.GetNodeDimensions(NodeGroup.Input);
            Assert.AreEqual(inDims.Count(), 0);

            var outDims = model.GetNodeDimensions(NodeGroup.Output);
            Assert.AreEqual(outDims.Count(), 1);
            Assert.AreEqual(outDims.First().Key, "o1");
            Assert.AreEqual(outDims.First().Value, 1);

            var outputVal = model.Evaluate(outDims.First().Key);

            var expected = new List<float>() {3};
            CollectionAssert.AreEqual(expected, outputVal);
        }*/
    }

    @Test
    public void testImageEvaluate() {
        System.loadLibrary("CNTKLibraryJavaBinding");

        String dataPath   = "data/";
        Function modelFunc  = Function.LoadModel("C:/repos/cntk/bindings/java/ScalaEvalExample/data/z.model");
        Variable outputVar  = modelFunc.getOutputs().get(0);
        Variable inputVar   = modelFunc.getArguments().get(0);
        NDShape inputShape    = inputVar.GetShape();

        int imageWidth    = inputShape.getDimensions().get(0).intValue();
        int imageHeight   = inputShape.getDimensions().get(1).intValue();
        int imageChannels = inputShape.getDimensions().get(2).intValue();
        int imageSize     = ((int) inputShape.GetTotalSize());

        // Image preprocessing to match input requirements of the model.
        BufferedImage bmp     = null;
        try {
            bmp = ImageIO.read(new File(dataPath + "00000.png"));
        } catch (IOException e) {
            e.printStackTrace();
            fail("Could not read image");
        }
        Image resized = bmp.getScaledInstance(imageWidth, imageHeight, Image.SCALE_DEFAULT);
        BufferedImage bImg =
                new BufferedImage(resized.getWidth(null), resized.getHeight(null), BufferedImage.TYPE_INT_RGB);
        bImg.getGraphics().drawImage(resized, 0, 0, null);


        int[] resizedCHW = new int[imageSize];

        int i = 0;
        for (int c = 0; c < imageChannels; c++) {
            for (int h = 0; h < bImg.getHeight(); h++) {
                for (int w = 0; w < bImg.getWidth(); w++) {
                    Color color = new Color(bImg.getRGB(w, h));
                    if (c == 0) {
                        resizedCHW[i]=color.getBlue();
                    } else if (c == 1) {
                        resizedCHW[i]=color.getGreen();
                    } else {
                        resizedCHW[i]=color.getRed();
                    }
                    i++;
                }
            }
        }

        FloatVector floatVec = new FloatVector();
        for (int intensity: resizedCHW) {
            floatVec.add(((float) intensity));
        }
        FloatVectorVector floatVecVec = new FloatVectorVector();
        floatVecVec.add(floatVec);

        // Create input data map
        Value inputVal     = Value.CreateDenseFloat(inputShape, floatVecVec, DeviceDescriptor.GetCPUDevice());
        UnorderedMapVariableValuePtr inputDataMap = new UnorderedMapVariableValuePtr();
        inputDataMap.Add(inputVar, inputVal);

        // Create output data map. Using null as Value to indicate using system allocated memory.
        // Alternatively, create a Value object and add it to the data map.
        UnorderedMapVariableValuePtr outputDataMap = new UnorderedMapVariableValuePtr();
        outputDataMap.Add(outputVar, null);

        // Start evaluation on the device
        modelFunc.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.GetCPUDevice());

        // Get evaluate result as dense output
        FloatVectorVector outputBuffer = new FloatVectorVector();
        outputDataMap.getitem(outputVar).CopyVariableValueToFloat(outputVar, outputBuffer);

        double[] truth = {-1.887479, -4.768533, 0.1516971, 6.805476,
                -0.3840595, 3.4106512, 1.3302777, -0.87714916, -2.18046, -4.1666183};
        double[] result = new double[((int) outputBuffer.get(0).size())];
        for (int j =0; j < outputBuffer.get(0).size(); j++) {
            result[j]=outputBuffer.get(0).get(j);
        }
        Assert.assertArrayEquals(result,truth,.0001);

    }

}
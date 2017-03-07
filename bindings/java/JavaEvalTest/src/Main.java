import com.microsoft.CNTK.*;
import javax.imageio.ImageIO;
import java.awt.image.*;
import java.awt.Image;
import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.reflect.Array;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Main {

    public static void main(String[] args) throws IOException {
        System.loadLibrary("CNTKLibraryJavaBinding");

        String outputName = "Plus2060";
        String dataPath   = "data/";

        Function modelFunc  = Function.LoadModel("C:/repos/cntk/bindings/java/ScalaEvalExample/data/z.model");

        Variable outputVar  = modelFunc.getOutputs().get(0);
        Variable inputVar   = modelFunc.getArguments().get(0);

        NDShape inputShape    = inputVar.GetShape();
        int imageWidth    = inputShape.getDimensions().get(0).intValue();
        int imageHeight   = inputShape.getDimensions().get(1).intValue();
        int imageChannels = inputShape.getDimensions().get(2).intValue();
        int imageSize     = ((int) inputShape.GetTotalSize());

        System.out.println("Evaluate single image");

        // Image preprocessing to match input requirements of the model.
        BufferedImage bmp     = ImageIO.read(new File(dataPath + "00000.png"));
        Image resized = bmp.getScaledInstance(imageWidth, imageHeight, Image.SCALE_DEFAULT);
        BufferedImage bImg =
                new BufferedImage(resized.getWidth(null), resized.getHeight(null), BufferedImage.TYPE_INT_RGB);
        // or use any other fitting type
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
        for (int j =0; j < outputBuffer.get(0).size(); j++) {
            System.out.println(outputBuffer.get(0).get(j));
        }

    }
}

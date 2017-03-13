// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

import com.microsoft.CNTK.*;
import javax.imageio.ImageIO;
import java.awt.image.*;
import java.awt.Image;
import java.awt.Color;
import java.io.File;
import java.io.IOException;

public class Main {

    public static Boolean equals(double d1,double d2){
        double sensitivity=.00001;
        return Math.abs(d1-d2)<sensitivity;
    }


    public static void main(String[] args) throws IOException {
        System.loadLibrary("CNTKLibraryJavaBinding");

        String dataPath = System.getenv("CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY") + "/Bindings/Java/";
        Function modelFunc = Function.LoadModel(dataPath + "z.model");
        Variable outputVar = modelFunc.getOutputs().get(0);
        Variable inputVar = modelFunc.getArguments().get(0);

        NDShape inputShape = inputVar.GetShape();
        int imageWidth = inputShape.getDimensions().get(0).intValue();
        int imageHeight = inputShape.getDimensions().get(1).intValue();
        int imageChannels = inputShape.getDimensions().get(2).intValue();
        int imageSize = ((int) inputShape.GetTotalSize());

        System.out.print("EvaluateSingleImage");

        // Image preprocessing to match input requirements of the model.
        BufferedImage bmp = ImageIO.read(new File(dataPath + "00000.png"));
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
                        resizedCHW[i] = color.getBlue();
                    } else if (c == 1) {
                        resizedCHW[i] = color.getGreen();
                    } else {
                        resizedCHW[i] = color.getRed();
                    }
                    i++;
                }
            }
        }

        FloatVector floatVec = new FloatVector();
        for (int intensity : resizedCHW) {
            floatVec.add(((float) intensity));
        }
        FloatVectorVector floatVecVec = new FloatVectorVector();
        floatVecVec.add(floatVec);
        // Create input data map
        Value inputVal = Value.CreateDenseFloat(inputShape, floatVecVec, DeviceDescriptor.GetCPUDevice());
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

        double[] trueResults = {
                -1.887479, -4.768533,
                0.1516971, 6.805476,
                -0.3840595, 3.4106512,
                1.3302777, -0.87714916,
                -2.18046, -4.1666183};


        for (int j = 0; j < outputBuffer.get(0).size(); j++) {
            System.out.println(outputBuffer.get(0).get(j) + " ");
            if (!equals(trueResults[j],outputBuffer.get(0).get(j))){
                throw new java.lang.RuntimeException("Test Failed on output "+j);
            }
        }
        System.out.println("Evaluation Complete");

    }
}

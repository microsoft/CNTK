// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

import com.microsoft.CNTK.*;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Main {

    private static Boolean equals(float d1, float d2) {
        double sensitivity = .00001;
        return Math.abs(d1 - d2) < sensitivity;
    }

    public static void main(String[] args) throws IOException {
        System.out.println("Working Directory = " + System.getProperty("user.dir"));

        DeviceDescriptor device = DeviceDescriptor.useDefaultDevice();
        File dataPath = new File(args[0]);


        // Load the model.
        // The model resnet20_cifar10_python.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/Models/TrainResNet_CIFAR10.py
        // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
        // Renaming the output model might be necessary
        Function modelFunc = Function.load(new File(dataPath, "resnet20_cifar10_python.dnn").getAbsolutePath(), device);
        Variable outputVar = modelFunc.getOutputs().get(0);
        Variable inputVar = modelFunc.getArguments().get(0);
        System.gc(); // This is not needed for normal usage. It is here just for testing that elements in getOutputs are not getting GC'd.

        NDShape inputShape = inputVar.getShape();
        int imageWidth = (int)inputShape.getDimensions()[0];
        int imageHeight = (int)inputShape.getDimensions()[1];
        int imageChannels = (int)inputShape.getDimensions()[2];
        int imageSize = ((int) inputShape.getTotalSize());

        System.out.println("EvaluateSingleImage");

        // Image preprocessing to match input requirements of the model.
        BufferedImage bmp = ImageIO.read(new File(dataPath, "00000.png"));
        Image resized = bmp.getScaledInstance(imageWidth, imageHeight, Image.SCALE_DEFAULT);
        BufferedImage bImg = new BufferedImage(
                resized.getWidth(null), resized.getHeight(null), BufferedImage.TYPE_INT_RGB);
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
        Value inputVal = Value.createDenseFloat(inputShape, floatVecVec, device);
        UnorderedMapVariableValuePtr inputDataMap = new UnorderedMapVariableValuePtr();
        inputDataMap.add(inputVar, inputVal);

        // Create output data map. Using null as Value to indicate using system allocated memory.
        // Alternatively, create a Value object and add it to the data map.
        UnorderedMapVariableValuePtr outputDataMap = new UnorderedMapVariableValuePtr();
        outputDataMap.add(outputVar, null);

        // Start evaluation on the device
        modelFunc.evaluate(inputDataMap, outputDataMap, device);

        // get evaluate result as dense output
        FloatVectorVector outputBuffer = new FloatVectorVector();
        outputDataMap.getitem(outputVar).copyVariableValueToFloat(outputVar, outputBuffer);


        float[] expectedResults = {
                -4.189664f,
                -3.1175408f,
                -1.7266451f,
                17.445856f,
                -3.7881997f,
                7.442085f,
                -3.8764064f,
                -6.151011f,
                3.721258f,
                -5.6161685f};

        FloatVector results = outputBuffer.get(0);
        for (int j = 0; j < results.size(); j++) {
            System.out.println(results.get(j) + " ");
            if (!equals(expectedResults[j], results.get(j))) {
                throw new RuntimeException("Test Failed on output " + j);
            }
        }
        System.out.println("Evaluation Complete");

    }
}

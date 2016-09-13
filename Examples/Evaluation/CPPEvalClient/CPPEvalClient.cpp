//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPPEvalClient.cpp : Sample application using the evaluation interface from C++
//

/// <summary>
/// Program for demonstrating how to run model evaluations using the native evaluation interface
/// </summary>
/// <description>
/// This program is a native C++ client using the native evaluation interface
/// located in the <see cref="eval.h"/> file.
/// The CNTK evaluation library (EvalDLL.dll on Windows, and LibEval.so on Linux), must be found through the system's path. 
/// The other requirement is that Eval.h be included
/// In order to run this program the model must already exist in the example. To create the model,
/// first run the example. Once the model file is created, you can run this client.
/// This program demonstrates the usage of the Evaluate method requiring the input and output layers as parameters.

#include "Eval.h"
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <math.h>
#include <sys/stat.h>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/variant.hpp>
#ifdef _WIN32
#include "Windows.h"
#endif

using namespace Microsoft::MSR::CNTK;

// Used for retrieving the model appropriate for the element type (float / double)
template<typename ElemType>
using GetEvalProc = void(*)(IEvaluateModel<ElemType>**);

typedef std::pair<std::wstring, std::vector<float>*> MapEntry;
typedef std::map<std::wstring, std::vector<float>*> Layer;

/// <summary>Saves the provided image to a file</summary>
/// <param name="mat">Source data</param>
/// <param name="name">Destination filename with extension</param>
void SaveImage(cv::Mat mat, std::string name)
{
    // Compression parameters for OpenCV to save image
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    try {
        cv::imwrite(name, mat, compression_params);
    }
    catch (std::runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    }
}

/// <summary>Reads floats from a string and saves them to a vector</summary>
/// <param name="s">A string to read from</param>
/// <param name="elems">A destination vector</param>
void SplitString(const std::string &s, std::vector<float> &elems)
{
    std::stringstream ss(s);
    float f;
    while (ss >> f) elems.push_back(f);
}

/// <summary>Substracts mean value from a corresponding element in an array</summary>
/// <param name="filePath">A path to XML file with the matrix of mean values</param>
/// <param name="array">An input array</param>
void SubstractMean(std::string filePath, std::vector<float>& array)
{
    // Read XML file with means 
    std::ifstream in(filePath);
    std::stringstream ss;
    ss << in.rdbuf();
    boost::property_tree::ptree pt;
    boost::property_tree::xml_parser::read_xml(ss, pt);

    // Get means matrix 
    std::vector<float> means;
    std::string dataString = pt.get_child("opencv_storage").get_child("MeanImg").get<std::string>("data");

    // Clean and read floats from the dataString
    dataString.erase(std::remove(dataString.begin(), dataString.end(), '\n'), dataString.end());
    SplitString(dataString, means);

    // Substract means
    for (int i = 0; i < array.size(); i++)
    {
        array[i] -= means[i];
    }
}

/// <summary>Reads an input for the MNIST example</summary>
/// <param name="input">An input string (expected length 28*28 with integer values in range [0,255]</param>
/// <param name="array">A destination array</param>
/// <param name="display">A boolean for displaying an image</param>
/// <param name="save">A boolean for saving an image</param>
void ReadMNISTInput(std::string input, std::vector<float>& array, bool display, bool save)
{
    int imgDimensionX = 28;
    int imgDimensionY = 28;
    int scaleFactor = 3;
    size_t step = sizeof(uchar)*imgDimensionX;

    uchar* matData = new uchar[imgDimensionX*imgDimensionY];
    std::istringstream iss(input);
    int pos = 0;
    for (std::string testImage; iss >> input;)
    {
        array.push_back(static_cast<float>(atof(input.c_str())));
        matData[pos] = static_cast<uchar> (atoi(input.c_str()));
        pos++;
    }

    cv::Mat img(cv::Size(imgDimensionX, imgDimensionY), CV_8UC1, matData, step);
    cv::resize(img, img, cv::Size(), scaleFactor, scaleFactor, cv::INTER_NEAREST);

    if (save)
    {
        SaveImage(img, "input.png");
    }
    if (display)
    {
        cv::imshow("input", img);
    }
}

/// <summary>Reads an RGB image from a file, crops to 224*224, and saves it to a vector in the correct order</summary>
/// <param name="filename">Path to file</param>
/// <param name="array">A destination array to load the data</param>
/// <param name="display">A boolean for an option to display an image</param>
void ReadRGBImage(std::string filename, std::vector<float>& array, bool display, bool save)
{
    auto testImage = cv::imread(filename, CV_LOAD_IMAGE_COLOR);

	// resize and center crop
	auto biggerSide = (testImage.rows > testImage.cols) ? testImage.rows : testImage.cols;
	auto smallerSide = (testImage.rows > testImage.cols) ? testImage.cols : testImage.rows;
	if (smallerSide != 224 || biggerSide != 224)
	{
		auto newCols = 224 * biggerSide / smallerSide;
		auto newSize = (testImage.rows > testImage.cols) ? cv::Size(224, newCols) : cv::Size(newCols, 224);
		cv::resize(testImage, testImage, newSize, 0, 0, cv::INTER_NEAREST);
		auto cropX = (testImage.rows > testImage.cols) ? 0 : newCols / 2 - 112;
		auto cropY = (testImage.rows < testImage.cols) ? 0 : newCols / 2 - 112;
		testImage = testImage(cv::Rect(cropX, cropY, 224, 224));
	}

    if (display)
    {
        cv::imshow("input", testImage);
    }
	if (save)
	{
		SaveImage(testImage, "input.png");
	}
    // Split image into 3 channels and save values in the following order: height * width * channel
    cv::Mat channel[3];
    cv::split(testImage, channel);
    for (auto ch : channel)
    {
        for (int i = 0; i < ch.rows; ++i)
        {
            array.insert(array.end(), ch.ptr<uchar>(i), ch.ptr<uchar>(i)+ch.cols);
        }
    }

    // Substract mean from each pixel value
    SubstractMean("Models/ImageNet1K_mean.xml", array);
}

/// <summary>Maps a set of given numbers to [0,1]</summary>
/// <param name="elems">A set of numbers</param>
void ScaleTo01(std::vector<float>& elems)
{
    auto minElement = *std::min_element(std::begin(elems), std::end(elems));
    auto maxElement = *std::max_element(std::begin(elems), std::end(elems));
    for (int i = 0; i < elems.size(); i++) elems[i] = (elems[i] - minElement) / (maxElement - minElement);
}

/// <summary>Converts a set of floats in [0,1] to unsinged chars in [0,255].</summary>
/// <param name="elems">A set of numbers</param>
std::vector<uchar> ConvertToUchar(std::vector<float> elems)
{
    std::vector<uchar> elemsChar;
    for (int i = 0; i < elems.size(); i++) elemsChar.push_back(static_cast<uchar> (round(255 * elems[i])));
    return elemsChar;
}

/// <summary>Transforms a layer output into cv::Mat</summary>
/// <param name="layerOutput">Output of the network layer</param>
/// <param name="params">{layerPosition, depth, imgDimensionX, imgDimensionY, scaleFactor}</param>
/// <param name="imgs">A destination vector</param>
void CreateImages(std::vector<uchar> layerOutput, std::vector<boost::variant<std::string, int>> params, std::vector<cv::Mat>& imgs)
{
    auto depth = boost::get<int>(params[1]);
    auto imgDimensionX = boost::get<int>(params[2]);
    auto imgDimensionY = boost::get<int>(params[3]);
    auto scaleFactor = boost::get<int>(params[4]);

    uchar** outputArrays = new uchar*[depth]; // stores images for each batch
    // Put elements in the correct order
    int l = 0;
    for (int i = 0; i < depth; i++)
    {
        outputArrays[i] = new uchar[imgDimensionX*imgDimensionY];
        for (int j = 0; j < imgDimensionX*imgDimensionY; j++)
        {
            outputArrays[i][j] = layerOutput[l];
            l++;
        }
    }

    auto step = sizeof(uchar)*imgDimensionX;
    for (int i = 0; i < depth; i++)
    {
        imgs.push_back(cv::Mat(cv::Size(imgDimensionX, imgDimensionY), CV_8UC1, outputArrays[i], step));
        if (imgDimensionY!=1) cv::resize(imgs[i], imgs[i], cv::Size(), scaleFactor, scaleFactor, cv::INTER_NEAREST);
    }
}

/// <summary>Displays images collected from layer activations</summary>
/// <param name="name">Layer name</param>
/// <param name="imgs">Images to display</param>
/// <param name="params">{layerPrefix, depth, imgDimensionX, imgDimensionY, scaleFactor}</param>
void VisualizeLayer(std::string name, std::vector<cv::Mat> imgs, std::vector<boost::variant<std::string, int>> params)
{
    auto layerPrefix = boost::get<std::string>(params[0]);
    auto depth = boost::get<int>(params[1]);
    auto imgDimensionX = boost::get<int>(params[2]);
    auto imgDimensionY = boost::get<int>(params[3]);
    auto scaleFactor = boost::get<int>(params[4]);
    float gap = 1.05; // interrow/column gap width

    // Create an empty pane for all images
    cv::Mat pane;
    cv::Size paneSize;
    if (imgs.size()==1 && imgDimensionY==1) // classes propabilities vector
    {
        // Remap to square
        auto side = static_cast<int> (ceil(sqrt(imgs[0].rows*imgs[0].cols)));
        paneSize = cv::Size(side, side);
        auto step = sizeof(uchar)*side;
        pane = cv::Mat(paneSize, CV_8SC1, imgs[0].data, step);
        cv::resize(pane, pane, cv::Size(), scaleFactor, scaleFactor, cv::INTER_NEAREST);
    }
    else
    {
        // Calculate the size of the pane based on params
        paneSize = cv::Size(static_cast<int> (gap*imgDimensionX*ceil(sqrt(depth))*scaleFactor), static_cast<int> (gap*imgDimensionY*scaleFactor*ceil(sqrt(depth))));
        pane = cv::Mat(paneSize, CV_8SC1, cv::Scalar(255));
        float column = 0;
        float row = 0;
        for (int i = 0; i < depth; i++)
        {
            if (imgs[i].cols*column > pane.cols - imgs[i].cols)
            {
                // Spacing between consecutive rows
                row = row + gap;
                // Start new row
                column = 0;
            }
            // Insert image in the correct position on the pane
            imgs[i].copyTo(pane(cv::Rect((int) round(imgs[i].cols*column), (int) round(imgs[i].rows*row), imgs[i].cols, imgs[i].rows)));
            // Spacing between consecutive columns
            column += gap;
        }
    }

    // Save the image
    SaveImage(pane, layerPrefix + "_" + name + ".png");

    // Display image
    cv::imshow(layerPrefix + "_" + name, pane);
}

/// <summary>Sets the dimensionality of network layers</summary>
/// <param name="model">Model name</param>
/// <param name="params">A destination dictionary of parameters</param>
void SetNetworkParams(std::string model, std::map<std::wstring, std::vector<boost::variant<std::string, int>>>& params)
{
    // LayerName : {layerPrefix, depth, imgDimensionX, imgDimensionY, scaleFactor}
    // <param name="layerPrefix">A string that defines the order of network layers</param>
    // <param name="depth">Equals the number of kernels on the layer</param>
    // <param name="imgDimensionX">Equals the layer output's x-dimensionality</param>
    // <param name="imgDimensionY">Equals the layer output's y-dimensionality</param>
    // <param name="scaleFactor">Scales an output image when visualizing layer's ativations (defined by user)</param>
    // One can get above parameters from the CNTK validation output
    // For instance, the interpretation of the following lines is 
    // Validating --> conv5.y = RectifiedLinear (conv5.z) : [13 x 13 x 256 x *] -> [13 x 13 x 256 x *]
    // Validating--> pool3 = MaxPooling(conv5.y) : [13 x 13 x 256 x *] ->[6 x 6 x 256 x *]
    // "pool3" follows "conv5" in the network stucture
    // depth[conv5.y] = 256; imgDimensionX[conv5.y] = 13; imgDimensionY[conv5.y] = 13
    // depth[pool3] = 256; imgDimensionX[pool3] = 6; imgDimensionY[pool3] = 6 

    if (model.find("AlexNet") != std::string::npos)
    {
        params[L"conv1.y"] = { "00", 64, 56, 56, 2 };
        params[L"pool1"] = { "01", 64, 27, 27, 3 };
        params[L"conv2.y"] = { "02", 192, 27, 27, 2 };
        params[L"pool2.p"] = { "03", 192, 13, 13, 3 };
        params[L"conv3.y"] = { "04", 384, 13, 13, 3 };
        params[L"conv4.y"] = { "05", 256, 13, 13, 3 };
        params[L"conv5.y"] = { "06", 256, 13, 13, 3 };
        params[L"pool3"] = { "07", 256, 6, 6, 6 };
        params[L"h1.b"] = { "08", 1, 4096, 1, 8 };
        params[L"h2.b"] = { "09", 1, 4096, 1, 8 };
        params[L"OutputNodes.z"] = { "10", 1, 1000, 1, 20 };
    }
    else if(model.find("MNIST") != std::string::npos && model.find("02_Convolution") != std::string::npos)
    {
        params[L"conv1.out"] = { "00", 16, 28, 28, 3 };
        params[L"pool1"] = { "01", 16, 14, 14, 6 };
        params[L"conv2.out"] = { "02", 32, 14, 14, 6 };
        params[L"pool2"] = { "03", 32, 7, 7, 12 };
        params[L"h1.y"] = { "04", 1, 125, 1, 84 };
        params[L"ol.z"] = { "05", 1, 10, 1, 84 };
    }
    else
    {
        fprintf(stderr, "Error: Parameters for the model %s are not specified yet", model.c_str());
    }
    
}

/// <summary>Collects visualisation of specified network layers and displays it</summary>
/// <param name="modelFilePath">Path to the network model file</param>
/// <param name="inputImage">An input image</param>
void VisualizeNetwork(std::string modelFilePath, std::string inputImage)
{
    IEvaluateModel<float> *model;

    struct stat statBuf;
    if (stat(modelFilePath.c_str(), &statBuf) != 0)
    {
        fprintf(stderr, "Error: The model %s does not exist. Please create a model first.\n", modelFilePath.c_str());
    }

    GetEvalF(&model);

    std::map<std::wstring, std::vector<boost::variant<std::string, int>>> params;
    SetNetworkParams(modelFilePath, params);

    // String with layer names
    std::string layers;
    for (auto layer : params)
    {
        layers += std::string(layer.first.begin(), layer.first.end()) + ", ";
    }

    // Load model with desired outputs
    std::string networkConfiguration;
    // When specifying outputNodeNames in the configuration, it will REPLACE the list of output nodes 
    // with the ones specified.
    networkConfiguration += "outputNodeNames=\"" + layers + "\"\n";
    networkConfiguration += "modelPath=\"" + modelFilePath + "\"";
    model->CreateNetwork(networkConfiguration);

    // get the model's layers dimensions
    std::map<std::wstring, size_t> inDims;
    std::map<std::wstring, size_t> outDims;
    model->GetNodeDimensions(inDims, NodeGroup::nodeInput);
    model->GetNodeDimensions(outDims, NodeGroup::nodeOutput);

    auto inputLayerName = inDims.begin()->first;
    std::vector<float> inputs;
    
    if (modelFilePath.find("AlexNet") != std::string::npos)
    {
        ReadRGBImage(inputImage, inputs, 1,1);
    }
    else if (modelFilePath.find("MNIST") != std::string::npos)
    {
        ReadMNISTInput(inputImage, inputs, 1, 1);
    }

    // Allocate the output values layer
    std::vector<std::vector<float>> outputs(outDims.size());

    // Setup the maps for inputs and output
    Layer inputLayer;
    inputLayer.insert(MapEntry(inputLayerName, &inputs));
    Layer outputLayers;
    int i = 0;
    for (auto &layer : outDims)
    {
        auto outputLayerName = layer.first;
        outputLayers.insert(MapEntry(outputLayerName, &outputs[i]));
        i++;
    }

    // We can call the evaluate method and get back the results 
    model->Evaluate(inputLayer, outputLayers);

    for (auto outputLayer : outputLayers)
    {
        auto layerName = outputLayer.first;
        auto output = *outputLayer.second;
        ScaleTo01(output);
        std::vector<uchar> outputChar = ConvertToUchar(output);
        std::vector<cv::Mat> imgs;
        CreateImages(outputChar, params[layerName], imgs);
        VisualizeLayer(std::string(layerName.begin(), layerName.end()), imgs, params[layerName]);
    }
}

int main(int argc, char* argv[])
{
    // Get the binary path (current working directory)
    argc = 0;
    std::string app = argv[0];
    std::string path;
    size_t pos;

    pos = app.rfind("\\");
    path = (pos == std::string::npos) ? "." : app.substr(0, pos);

    // This relative path assumes launching from CNTK's binary folder, e.g. x64\Release
    std::string modelWorkingDirectory = path + "/../../";
    std::string modelFilePath;
    std::string customPath;
    std::string inputImage;
	std::string userPath;
    char defaultPath;
    std::cout << "Choose example to run: \n \
                 1. MNIST \n \
                 2. AlexNet \n ";

    int option;
    std::cin >> option;
    switch (option)
    {
        case 1:
			userPath = "Examples/Image/MNIST/Data/";
			std::cout << "Use default model path? (y/n) (working directory \"" << userPath << "\")\n";
			std::cin >> defaultPath;
            if (defaultPath == 'y')
            {
                modelWorkingDirectory += userPath;
                modelFilePath = modelWorkingDirectory + "../Output/Models/02_Convolution";
            }
            else
            {
                std::cout << "Specify model file path relative to CNTK root folder \n";
                std::cin >> userPath;
                modelFilePath = modelWorkingDirectory + userPath;
            }
            // MNIST sample input taken from MNIST/Data/Train-28x28_cntk_text.txt
            inputImage = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 63 197 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 20 254 230 24 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 20 254 254 48 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 20 254 255 48 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 20 254 254 57 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 20 254 254 108 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 239 254 143 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 178 254 143 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 178 254 143 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 178 254 162 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 178 254 240 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 113 254 240 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 83 254 245 31 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 79 254 246 38 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 214 254 150 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 144 241 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 144 240 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 144 254 82 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 230 247 40 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 168 209 31 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0";
			break;
        case 2:
			userPath = "Examples/Evaluation/CPPEvalClient/";
            std::cout << "Use default model path? (y/n) (working directory \"" << userPath << "\")\n";
            std::cin >> defaultPath;
            if (defaultPath == 'y')
            {
                modelWorkingDirectory += userPath;
                modelFilePath = modelWorkingDirectory + "Models/AlexNet.89";
            }
            else
            {
                std::cout << "Specify model file path relative to CNTK root folder \n";
                std::cin >> userPath;
				modelFilePath = modelWorkingDirectory + userPath;
            }
			std::cout << "Provide test file path relative to  \"Examples/Evaluation/CPPEvalClient/\" \n";
			std::cin >> inputImage;
            break;
        default:
            fprintf(stderr, "Cannot match the choice\n");
			return 0;
    }

    VisualizeNetwork(modelFilePath, inputImage);
    
    cv::waitKey(0);
    return 0;
}
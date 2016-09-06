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

/// <summary>Reads an RGB image from file and saves it to vector in correct order</summary>
/// <param name="filename">Path to file</param>
/// <param name="array">Destination array to load the data</param>
void ReadRGBImage(std::string filename, std::vector<float>& array)
{
	auto testImage = cv::imread(filename, CV_LOAD_IMAGE_COLOR);

	// Display input image
	//cv::imshow("input", testImage);
	//cv::waitKey(0);

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
}

/// <summary>Saves image to file</summary>
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

/// <summary>Reads floats from string and saves them to vector</summary>
/// <param name="s">String to read from</param>
/// <param name="elems">Destination vaector</param>
void SplitString(const std::string &s, std::vector<float> &elems) 
{
	std::stringstream ss(s);
	float f;
	while (ss >> f) elems.push_back(f);
}

/// <summary>Substracts mean value from a corresponding element in array</summary>
/// <param name="filePath">Path to XML file with the matrix of mean values</param>
/// <param name="array">Input array</param>
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

	// Clean and read the dataString
	dataString.erase(std::remove(dataString.begin(), dataString.end(), '\n'), dataString.end());
	SplitString(dataString, means);

	for (int i = 0; i < array.size(); i++) array[i] -= means[i];
}

/// <summary>Maps a set of given numbers to [0,1] interval</summary>
/// <param name="elems">Set of numbers</param>
void ScaleTo01(std::vector<float>& elems)
{
	auto minElement = *std::min_element(std::begin(elems), std::end(elems));
	auto maxElement = *std::max_element(std::begin(elems), std::end(elems));
	for (int i = 0; i < elems.size(); i++) elems[i] = (elems[i] - minElement) / (maxElement - minElement);
}

/// <summary>Converts a set of floats in [0,1] to unsinged chars in [0,255].</summary>
/// <param name="elems">Set of numbers</param>
std::vector<uchar> ConvertToUchar(std::vector<float> elems)
{
	std::vector<uchar> elemsChar;
	for (int i = 0; i < elems.size(); i++) elemsChar.push_back(round(255 * elems[i]));
	return elemsChar;
}

/// <summary>Transforms layer output into cv::Mat</summary>
/// <param name="layerOutput">Output of the network layer</param>
/// <param name="params">{layerPosition, numberOfBatches, imgDimensionX, imgDimensionY, scaleFactor}</param>
/// <param name="imgs">Destination vector</param>
void CreateImages(std::vector<uchar> layerOutput, std::vector<boost::variant<std::string, int>> params, std::vector<cv::Mat>& imgs)
{
	auto numberOfBatches = boost::get<int>(params[1]);
	auto imgDimensionX = boost::get<int>(params[2]);
	auto imgDimensionY = boost::get<int>(params[3]);
	auto scaleFactor = boost::get<int>(params[4]);

	uchar** outputArrays = new uchar*[numberOfBatches]; // stores images for each batch
	// Put elements in the correct order
	int l = 0;
	for (int i = 0; i < numberOfBatches; i++)
	{
		outputArrays[i] = new uchar[imgDimensionX*imgDimensionY];
		for (int j = 0; j < imgDimensionX*imgDimensionY; j++)
		{
			outputArrays[i][j] = layerOutput[l];
			l++;
		}
	}

	auto step = sizeof(uchar)*imgDimensionX;
	for (int i = 0; i < numberOfBatches; i++)
	{
		imgs.push_back(cv::Mat(cv::Size(imgDimensionX, imgDimensionY), CV_8UC1, outputArrays[i], step));
		if (imgDimensionY!=1) cv::resize(imgs[i], imgs[i], cv::Size(), scaleFactor, scaleFactor, cv::INTER_NEAREST);
	}
}

/// <summary>Displays images collected from layer activations</summary>
/// <param name="name">Layer name</param>
/// <param name="imgs">Images to display</param>
/// <param name="params">{layerPosition, numberOfBatches, imgDimensionX, imgDimensionY, scaleFactor}</param>
void VisualizeLayer(std::string name, std::vector<cv::Mat> imgs, std::vector<boost::variant<std::string, int>> params)
{
	auto layerPosition = boost::get<std::string>(params[0]);
	auto numberOfBatches = boost::get<int>(params[1]);
	auto imgDimensionX = boost::get<int>(params[2]);
	auto imgDimensionY = boost::get<int>(params[3]);
	auto scaleFactor = boost::get<int>(params[4]);

	// Create an empty pane for all images
	cv::Mat pane;
	cv::Size paneSize;
	if (imgs.size()==1 && imgDimensionY==1) // classes propabilities vector
	{
		// Remap to square
		auto side = ceil(sqrt(imgs[0].rows*imgs[0].cols));
		paneSize = cv::Size(side, side);
		auto step = sizeof(uchar)*side;
		pane = cv::Mat(paneSize, CV_8SC1, imgs[0].data, step);
		cv::resize(pane, pane, cv::Size(), scaleFactor, scaleFactor, cv::INTER_NEAREST);
	}
	else
	{
		// Calculate the size of the pane based on params
		paneSize = cv::Size(1.05*imgDimensionX*ceil(sqrt(numberOfBatches))*scaleFactor, 1.05*imgDimensionY*scaleFactor*ceil(sqrt(numberOfBatches)));
		pane = cv::Mat(paneSize, CV_8SC1);
		float column = 0;
		float row = 0;
		for (int i = 0; i < numberOfBatches; i++)
		{
			if (imgs[i].cols*column > pane.cols - imgs[i].cols)
			{
				// Spacing between consequent rows
				row = row + 1.05;
				// Start new row
				column = 0;
			}
			// Insert image in the correct position on the pane
			imgs[i].copyTo(pane(cv::Rect(imgs[i].cols*column, imgs[i].rows*row, imgs[i].cols, imgs[i].rows)));
			// Spacing between consequent columns
			column += 1.05;
		}
	}

	// Save the image
	SaveImage(pane, layerPosition + "_" + name + ".png");

	// Display image
	cv::imshow(layerPosition + "_" + name, pane);
}

int main(int argc, char* argv[])
{
	// Get the binary path (current working directory)
	argc = 0;
	std::string app = argv[0];
	std::string path;
	IEvaluateModel<float> *model;
	size_t pos;

#ifdef _WIN32
	pos = app.rfind("\\");
	path = (pos == std::string::npos) ? "." : app.substr(0, pos);

	// This relative path assumes launching from CNTK's binary folder, e.g. x64\Release
	const std::string modelWorkingDirectory = path + "/../../Examples/Evaluation/CPPEvalClient/";
#else // on Linux
	pos = app.rfind("/");
	path = (pos == std::string::npos) ? "." : app.substr(0, pos);

	// This relative path assumes launching from CNTK's binary folder, e.g. build/release/bin/
	const std::string modelWorkingDirectory = path + "/../../Examples/Evaluation/CPPEvalClient/";
#endif
	const std::string modelFilePath = modelWorkingDirectory + "Models/AlexNet.89";

	struct stat statBuf;
	if (stat(modelFilePath.c_str(), &statBuf) != 0)
	{
		fprintf(stderr, "Error: The model %s does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/MNIST to create the model.\n", modelFilePath.c_str());
		return(1);
	}

	GetEvalF(&model);

	// Define needed layer names and their dimensionality
	// LayerName : {layerPosition, numberOfBatches, imgDimensionX, imgDimensionY, scaleFactor}
	// TODO: read values from network configuration?
	std::map<std::wstring, std::vector<boost::variant<std::string, int>>> dimensions;
	dimensions[L"conv1.y"] = { "00", 64, 56, 56, 2 };
	dimensions[L"pool1"] = { "01", 64, 27, 27, 3 };
	dimensions[L"conv2.y"] = { "02", 192, 27, 27, 2 };
	dimensions[L"pool2"] = { "03", 192, 13, 13, 3 };
	dimensions[L"conv3.y"] = { "04", 384, 13, 13, 3 };
	dimensions[L"conv4.y"] = { "05", 256, 13, 13, 3 };
	dimensions[L"conv5.y"] = { "06", 256, 13, 13, 3 };
	dimensions[L"pool3"] = { "07", 256, 6, 6, 6 };
	dimensions[L"h1.b"] = { "08", 1, 4096, 1, 8 };
	dimensions[L"h2.b"] = { "09", 1, 4096, 1, 8 };
	dimensions[L"OutputNodes.z"] = { "10", 1, 1000, 1, 20 };

	// String with layer names
	std::string layers;
	for (auto dim : dimensions)
	{
		layers += std::string(dim.first.begin(), dim.first.end())+", ";
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

	// Read input image
	auto inputImage = "val100/test2.JPG";
	auto inputLayerName = inDims.begin()->first;
	std::vector<float> inputs;
	ReadRGBImage(inputImage, inputs);

	// Substract mean from each pixel value
	SubstractMean("Models/ImageNet1K_mean.xml", inputs);

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
		CreateImages(outputChar, dimensions[layerName], imgs);
		VisualizeLayer(std::string(layerName.begin(), layerName.end()), imgs, dimensions[layerName]);
	}
	cv::waitKey(0);

	return 0;
}


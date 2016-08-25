//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPPEvalClient.cpp : Sample application using the evaluation interface from C++
//
#include "Eval.h"
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <math.h>
#ifdef _WIN32
#include "Windows.h"
#endif

using namespace Microsoft::MSR::CNTK;

// Used for retrieving the model appropriate for the element type (float / double)
template<typename ElemType>
using GetEvalProc = void(*)(IEvaluateModel<ElemType>**);

typedef std::pair<std::wstring, std::vector<float>*> MapEntry;
typedef std::map<std::wstring, std::vector<float>*> Layer;

/// <summary>
/// Program for demonstrating how to run model evaluations using the native evaluation interface
/// </summary>
/// <description>
/// This program is a native C++ client using the native evaluation interface
/// located in the <see cref="eval.h"/> file.
/// The CNTK evaluation library (EvalDLL.dll on Windows, and LibEval.so on Linux), must be found through the system's path. 
/// The other requirement is that Eval.h be included
/// In order to run this program the model must already exist in the example. To create the model,
/// first run the example in <CNTK>/Examples/Image/MNIST. Once the model file 01_OneHidden is created,
/// you can run this client.
/// This program demonstrates the usage of the Evaluate method requiring the input and output layers as parameters.

/**
* @brief makeCanvas Makes composite image from the given images
* @param vecMat Vector of Images.
* @param windowHeight The height of the new composite image to be formed.
* @param nRows Number of rows of images. (Number of columns will be calculated
*              depending on the value of total number of images).
* @return new composite image.
*/

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
	const std::string modelWorkingDirectory = path + "/../../Examples/Image/MNIST/Data/";
#else // on Linux
	pos = app.rfind("/");
	path = (pos == std::string::npos) ? "." : app.substr(0, pos);

	// This relative path assumes launching from CNTK's binary folder, e.g. build/release/bin/
	const std::string modelWorkingDirectory = path + "/../../../Examples/Image/MNIST/Data/";
#endif

	// params for openCV
	int numberOfChannels = 1;
	int imgDimensionX = 28;
	int imgDimensionY = 28;
	int scaleFactor = 3.0;
	auto cvType = CV_8UC1;
	size_t step = sizeof(uchar)*imgDimensionX;

	// compression parameters for saving an image
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	GetEvalF(&model);

	const std::string modelFilePath = modelWorkingDirectory + "../Output/Models/02_Convolution";

	// load model with desired outputs
	std::string networkConfiguration;

	networkConfiguration += "outputNodeNames=\"(conv1.out,pool1,conv2.out,pool2.p,ol.z)\"\n";
	networkConfiguration += "modelPath=\"" + modelFilePath + "\"";
	model->CreateNetwork(networkConfiguration);

	// get the model's layers dimensions
	std::map<std::wstring, size_t> inDims;
	std::map<std::wstring, size_t> outDims;
	model->GetNodeDimensions(inDims, NodeGroup::nodeInput);
	model->GetNodeDimensions(outDims, NodeGroup::nodeOutput);

	// Read input
	auto inputLayerName = inDims.begin()->first;
	std::string testImage = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 73 253 227 73 21 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 73 251 251 251 174 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 166 228 251 251 251 122 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 62 220 253 251 251 251 251 79 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 79 231 253 251 251 251 251 232 77 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 145 253 253 253 255 253 253 253 253 255 108 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 144 251 251 251 253 168 107 169 251 253 189 20 0 0 0 0 0 0 0 0 0 0 0 0 0 0 27 89 236 251 235 215 164 15 6 129 251 253 251 35 0 0 0 0 0 0 0 0 0 0 0 0 0 47 211 253 251 251 142 0 0 0 37 251 251 253 251 35 0 0 0 0 0 0 0 0 0 0 0 0 0 109 251 253 251 251 142 0 0 0 11 148 251 253 251 164 0 0 0 0 0 0 0 0 0 0 0 0 11 150 253 255 211 25 0 0 0 0 11 150 253 255 211 25 0 0 0 0 0 0 0 0 0 0 0 0 140 251 251 253 107 0 0 0 0 0 37 251 251 211 46 0 0 0 0 0 0 0 0 0 0 0 0 0 190 251 251 253 128 5 0 0 0 0 37 251 251 51 0 0 0 0 0 0 0 0 0 0 0 0 0 0 115 251 251 253 188 20 0 0 32 109 129 251 173 103 0 0 0 0 0 0 0 0 0 0 0 0 0 0 217 251 251 201 30 0 0 0 73 251 251 251 71 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 166 253 253 255 149 73 150 253 255 253 253 143 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 140 251 251 253 251 251 251 251 253 251 230 61 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 190 251 251 253 251 251 251 251 242 215 55 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 21 189 251 253 251 251 251 173 103 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 31 200 253 251 96 71 20 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0";

	std::vector<float> inputs;
	uchar* inputArray = new uchar[imgDimensionX*imgDimensionY];

	std::istringstream iss(testImage);
	pos = 0;
	for (std::string testImage; iss >> testImage;)
	{
		inputs.push_back((atoi(testImage.c_str())));
		inputArray[pos] = (atoi(testImage.c_str()));
		pos++;
	}

	cv::Mat img0(cv::Size(imgDimensionX, imgDimensionY), cvType, inputArray, step);
	cv::resize(img0, img0, cv::Size(), scaleFactor, scaleFactor, cv::INTER_NEAREST);
	// save the input image
	try {
		cv::imwrite("input.png", img0, compression_params);
	}
	catch (std::runtime_error& ex) {
		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
		return 1;
	}
	/*cv::imshow("input", img0);
	cv::waitKey(0);*/

	std::vector<std::vector<float>> outputs(outDims.size());

	// setup the maps for inputs and output
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

	// evaluate the model
	model->Evaluate(inputLayer, outputLayers);

	for (auto outputLayer : outputLayers)
	{

		// set number of channels, image dimensions, and scale factor depending on the layer
		if (outputLayer.first == std::wstring{ L"conv1.out" })
		{
			numberOfChannels = 16;
			imgDimensionX = 28;
			imgDimensionY = 28;
			scaleFactor = 3.0;
		}
		else if (outputLayer.first == std::wstring{ L"conv2.out" })
		{
			numberOfChannels = 32;
			imgDimensionX = 14;
			imgDimensionY = 14;
			scaleFactor = 6.0;
		}
		else if (outputLayer.first == std::wstring{ L"pool1" })
		{
			numberOfChannels = 16;
			imgDimensionX = 14;
			imgDimensionY = 14;
			scaleFactor = 6.0;
		}
		else if (outputLayer.first == std::wstring{ L"pool2.p" })
		{
			numberOfChannels = 32;
			imgDimensionX = 7;
			imgDimensionY = 7;
			scaleFactor = 12.0;
		}
		else if (outputLayer.first == std::wstring{ L"ol.z" })
		{
			numberOfChannels = 1;
			imgDimensionX = 10;
			imgDimensionY = 1;
			scaleFactor = 84.0;
		}
		else
		{
			std::cout << "Could not match layer name. Setting to defaults";
			numberOfChannels = 1;
			imgDimensionX = 28;
			imgDimensionY = 28;
			scaleFactor = 6.0;
		}

		auto output = *outputLayer.second;

		// normalize and scale to [0,255]
		std::vector<uchar> outputsChar(output.size());
		auto minMaxOutput = std::minmax_element(output.begin(), output.end());
		for (int i = 0; i < output.size(); i++)
		{
			outputsChar[i] = (uchar)round(255 * (output[i] - output[minMaxOutput.first - output.begin()])
				/ (output[minMaxOutput.second - output.begin()] - output[(minMaxOutput.first - output.begin())]));
		}

		uchar** outputArrays = new uchar*[numberOfChannels];
		int l = 0;
		for (int i = 0; i < numberOfChannels; i++)
		{
			outputArrays[i] = new uchar[imgDimensionX*imgDimensionY];
			for (int j = 0; j < imgDimensionX*imgDimensionY; j++)
			{
				outputArrays[i][j] = outputsChar[l];
				l++;
			}
		}
		step = sizeof(uchar)*imgDimensionX;
		std::vector<cv::Mat> imgs(numberOfChannels);
		for (int i = 0; i < numberOfChannels; i++)
		{
			imgs[i] = cv::Mat(cv::Size(imgDimensionX, imgDimensionY), cvType, outputArrays[i], step);
			cv::resize(imgs[i], imgs[i], cv::Size(), scaleFactor, scaleFactor, cv::INTER_NEAREST);
		}

		// copy image of each channel to layer pane
		auto pane = cv::Mat(cv::Size(1.05*imgDimensionX*ceil(sqrt(numberOfChannels))*scaleFactor, 1.05*imgDimensionY*scaleFactor*ceil(sqrt(numberOfChannels))), cvType);
		float column = 0;
		float row = 0;
		for (int i = 0; i < numberOfChannels; i++)
		{
			if (imgs[i].cols*column > pane.cols - imgs[i].cols)
			{
				row = row + 1.05;
				column = 0;
			}
			imgs[i].copyTo(pane(cv::Rect(imgs[i].cols*column, imgs[i].rows*row, imgs[i].cols, imgs[i].rows)));
			column += 1.05;
		}

		// save the image
		try {
			cv::imwrite(std::string(outputLayer.first.begin(), outputLayer.first.end()) + ".png", pane, compression_params);
		}
		catch (std::runtime_error& ex) {
			fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
			return 1;
		}
		//cv::imshow(std::string(outputLayer.first.begin(), outputLayer.first.end()), pane);
	}

	//cv::waitKey(0);
	return 0;
}
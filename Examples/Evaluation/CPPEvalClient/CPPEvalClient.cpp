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
	
	const int numberOfChannels = 16;
	const int imgDimensionX = 28;
	const int imgDimensionY = 28;
	int scaleFactor = 3.0;
	auto cvType = CV_8UC1;

	GetEvalF(&model);

	const std::string modelFilePath = modelWorkingDirectory + "../Output/Models/02_Convolution";

	// Load model with desired outputs
	std::string networkConfiguration;
	// Uncomment the following line to re-define the outputs (include h1.z AND the output ol.z)
	// When specifying outputNodeNames in the configuration, it will REPLACE the list of output nodes 
	// with the ones specified.
	networkConfiguration += "outputNodeNames=\"(features,conv1.out,pool1,conv2.out,pool2.p)\"\n";
	networkConfiguration += "modelPath=\"" + modelFilePath + "\"";
	model->CreateNetwork(networkConfiguration);

	// get the model's layers dimensions
	std::map<std::wstring, size_t> inDims;
	std::map<std::wstring, size_t> outDims;
	model->GetNodeDimensions(inDims, NodeGroup::nodeInput);
	model->GetNodeDimensions(outDims, NodeGroup::nodeOutput);


	// Read input
	auto inputLayerName = inDims.begin()->first;
	std::string testImage = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 38 43 105 255 253 253 253 253 253 174 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 43 139 224 226 252 253 252 252 252 252 252 252 158 14 0 0 0 0 0 0 0 0 0 0 0 0 0 0 178 252 252 252 252 253 252 252 252 252 252 252 252 59 0 0 0 0 0 0 0 0 0 0 0 0 0 0 109 252 252 230 132 133 132 132 189 252 252 252 252 59 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 29 29 24 0 0 0 0 14 226 252 252 172 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 85 243 252 252 144 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 88 189 252 252 252 14 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 91 212 247 252 252 252 204 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 32 125 193 193 193 253 252 252 252 238 102 28 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 45 222 252 252 252 252 253 252 252 252 177 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 45 223 253 253 253 253 255 253 253 253 253 74 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 31 123 52 44 44 44 44 143 252 252 74 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 15 252 252 74 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 86 252 252 74 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 75 9 0 0 0 0 0 0 98 242 252 252 74 0 0 0 0 0 0 0 0 0 0 0 0 0 61 183 252 29 0 0 0 0 18 92 239 252 252 243 65 0 0 0 0 0 0 0 0 0 0 0 0 0 208 252 252 147 134 134 134 134 203 253 252 252 188 83 0 0 0 0 0 0 0 0 0 0 0 0 0 0 208 252 252 252 252 252 252 252 252 253 230 153 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 49 157 252 252 252 252 252 217 207 146 45 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7 103 235 252 172 103 24 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0";

	std::vector<float> inputs;
	uchar inputArray[imgDimensionX][imgDimensionX];

	std::istringstream iss(testImage);
	for (std::string testImage; iss >> testImage;)
		inputs.push_back((atoi(testImage.c_str())));

	int k = 0;
	for (int i = 0; i < imgDimensionX; i++)
	{
		for (int j = 0; j < imgDimensionX; j++)
		{
			inputArray[i][j] = (uchar) inputs[k];
			k++;
		}
	}
	
	
	/*cv::Mat img0(cv::Size(imgDimensionX, imgDimensionY), cvType, &inputArray);
	cv::resize(img0, img0, cv::Size(), scaleFactor, scaleFactor, cv::INTER_NEAREST);
	cv::imshow("input", img0);
	cv::waitKey(0);*/


	// Allocate the output values layer
	std::vector<float> outputs;

	// Setup the maps for inputs and output
	Layer inputLayer;
	inputLayer.insert(MapEntry(inputLayerName, &inputs));
	Layer outputLayer;
	auto outputLayerName = outDims.begin()->first;
	outputLayer.insert(MapEntry(outputLayerName, &outputs));

	// We can call the evaluate method and get back the results (single layer)...
	model->Evaluate(inputLayer, outputLayer);

	std::vector<uchar> outputsChar(outputs.size());
	auto minMaxOutput = std::minmax_element(outputs.begin(), outputs.end());
	for (int i = 0; i < outputs.size(); i++)
	{
		outputsChar[i] = (uchar) round(255 * (outputs[i] - outputs[minMaxOutput.first - outputs.begin()])
			/ (outputs[minMaxOutput.second - outputs.begin()] - outputs[minMaxOutput.first - outputs.begin()]));
	}

	uchar outputArrays[numberOfChannels][imgDimensionX][imgDimensionY];

	int l = 0;
	for (int i = 0; i < numberOfChannels; i++)
	{
		for (int j = 0; j < imgDimensionX; j++)
		{
			for (int m = 0; m < imgDimensionY; m++)
			{
				outputArrays[i][j][m] = outputsChar[l];
				l++;
			}
		}
	}

	std::vector<cv::Mat> imgs(numberOfChannels);
	for (int i = 0; i < numberOfChannels; i++)
	{	
		imgs[i] = cv::Mat(cv::Size(imgDimensionX, imgDimensionY), cvType, &outputArrays[i]);
		cv::resize(imgs[i], imgs[i], cv::Size(), scaleFactor, scaleFactor, cv::INTER_NEAREST);
		//cv::imshow("Conv1 | channel# " + std::to_string(i+1), imgs[i]);
	}

	auto pane = cv::Mat(cv::Size(1344, imgDimensionY*scaleFactor), cvType);
	int column = 0;
	int row = 0;
	for (int i = 0; i < numberOfChannels; i++)
	{
		if (imgs[i].cols*column > pane.cols - imgs[i].cols)
		{
			row++;
			column = 0;
		}
		imgs[i].copyTo(pane(cv::Rect(imgs[i].cols*column, imgs[i].rows*row, imgs[i].cols, imgs[i].rows)));
		column++;
	}
	cv::imshow("Conv1.out", pane);

	// Save the image
	/*std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	try {
		cv::imwrite("conv1.png", pane, compression_params);
	}
	catch (std::runtime_error& ex) {
		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
		return 1;
	}*/

	cv::waitKey(0);
    return 0;
}


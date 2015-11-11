//
// <copyright file="DataReaderTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "stdafx.h"
#include "Matrix.h"
#include "commandArgUtil.h"
#include "DataReader.h"
#include <fstream>

using namespace Microsoft::MSR::CNTK;

namespace Microsoft
{
	namespace MSR
	{
		namespace CNTK
		{
			namespace Test
			{
				const int deviceId = 0;
				const float epsilon = 0.0001f;

				BOOST_AUTO_TEST_SUITE(ReaderTestSuite)

				// StandardLoopTest
				// Test of the DataReader loop 
				BOOST_AUTO_TEST_CASE(UCIFastReaderSimpleData)
				{
					size_t epochSize = 500;
					size_t mbSize = 250;
					size_t epochs = 2;

					std::wstring filePath{ L"Simple.config" };
					std::string controlDataFilePath{ "controldata.txt" };
					std::string testDataFilePath{ "testdata.txt" };

					ConfigParameters config;
					std::array<wchar_t*, 2> argv = { L"CNTK", L"configFile=Simple.config" };
					std::string rawConfigString = ConfigParameters::ParseCommandLine(2, argv.data(), config);
					config.ResolveVariables(rawConfigString);
					ConfigParameters simpleDemoConfig = config("Simple_Demo");
					ConfigParameters readerConfig = simpleDemoConfig("reader");

					DataReader<float> dataReader(readerConfig);

					std::map<std::wstring, Matrix<float>*> map{};
					Matrix<float> features;
					Matrix<float> labels;
					map.insert(std::pair<wstring, Matrix<float>*>(L"features", &features));
					map.insert(std::pair<wstring, Matrix<float>*>(L"labels", &labels));

					// Setup output file
					std::remove(testDataFilePath.c_str());
					ofstream outputFile(testDataFilePath, ios::out);

					for (int epoch = 0; epoch < epochs; epoch++)
					{
						dataReader.StartMinibatchLoop(mbSize, epoch, epochSize);

						for (int cnt = 0; dataReader.GetMinibatch(map); cnt++)
						{
							size_t numLabelsRows = labels.GetNumRows();
							size_t numLabelsCols = labels.GetNumCols();
							size_t numFeaturesRows = features.GetNumRows();
							size_t numFeaturesCols = features.GetNumCols();

							BOOST_CHECK_EQUAL(2, numLabelsRows);
							BOOST_CHECK_EQUAL(mbSize, numLabelsCols);
							BOOST_CHECK_EQUAL(2, numFeaturesRows);
							BOOST_CHECK_EQUAL(mbSize, numFeaturesCols);

							if (outputFile.is_open())
							{
								float* arrFeatures = features.CopyToArray();
								float* arrLabels = labels.CopyToArray();

								float* pFeature = arrFeatures;
								float* pLabel = arrLabels;

								size_t numFeatures = numFeaturesRows * numFeaturesCols;
								size_t numLabels = numLabelsRows * numLabelsCols;

								for (int i = 0; i < numFeatures; i++)
								{
									outputFile << *pFeature++;
									outputFile << (i % numFeaturesRows ? "\n" : " ");
								}

								for (int i = 0; i < numLabels; i++)
								{
									outputFile << *pLabel++;
									outputFile << (i % numLabelsRows ? "\n" : " ");
								}

								delete[] arrFeatures;
								delete[] arrLabels;
							}
						}
					}

					outputFile.close();

					std::ifstream ifs1(controlDataFilePath);
					std::ifstream ifs2(testDataFilePath);

					std::istream_iterator<char> b1(ifs1);
					std::istream_iterator<char> e1;
					std::istream_iterator<char> b2(ifs2);
					std::istream_iterator<char> e2;

					BOOST_CHECK_EQUAL_COLLECTIONS(b1, e1, b2, e2);

					ifs1.close();
					ifs2.close();

					std::remove(testDataFilePath.c_str());
				};

				BOOST_AUTO_TEST_SUITE_END()
			}
		}
	}
}
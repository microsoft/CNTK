//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "Common/ReaderTestHelper.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

struct ImageReaderFixture : ReaderFixture
{
    ImageReaderFixture()
        : ReaderFixture("/Data")
    {
    }
};

BOOST_FIXTURE_TEST_SUITE(ReaderTestSuite, ImageReaderFixture)

BOOST_AUTO_TEST_CASE(ImageReaderSimple)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/ImageReaderSimple_Config.cntk",
        testDataPath() + "/Control/ImageReaderSimple_Control.txt",
        testDataPath() + "/Control/ImageReaderSimple_Output.txt",
        "Simple_Test",
        "reader",
        4,
        4,
        1,
        1,
        0,
        0,
        1);
}

BOOST_AUTO_TEST_CASE(ImageAndTextReaderSimple)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/ImageAndTextReaderSimple_Config.cntk",
        testDataPath() + "/Control/ImageAndTextReaderSimple_Control.txt",
        testDataPath() + "/Control/ImageAndTextReaderSimple_Output.txt",
        "Simple_Test",
        "reader",
        4,
        4,
        1,
        1,
        0,
        0,
        1);
}

BOOST_AUTO_TEST_CASE(ImageSimpleCompositeAndBase64)
{
    auto test = [this](std::vector<std::wstring> additionalParameters)
    {
        HelperRunReaderTest<float>(
            testDataPath() + "/Config/ImageReaderSimple_Config.cntk",
            testDataPath() + "/Control/ImageSimpleCompositeAndBase64_Control.txt",
            testDataPath() + "/Control/ImageSimpleCompositeAndBase64_Output.txt",
            "Composite_Test",
            "reader",
            4,
            4,
            1,
            1,
            1,
            0,
            1,
            false,
            true,
            true,
            additionalParameters);
    };

    // Image deserializer.
    test({});
    // Base64 deserializer.
    test(
    {
        L"MapFile=\"$RootDir$/Base64ImageReaderSimple_map.txt\"",
        L"DeserializerType=\"Base64ImageDeserializer\"",
        L"useNumericSequenceKeys=true"
    });
};

BOOST_AUTO_TEST_CASE(InvalidImageSimpleCompositeAndBase64)
{
    auto test = [this](std::vector<std::wstring> additionalParameters)
    {
        HelperRunReaderTest<float>(
            testDataPath() + "/Config/ImageReaderSimple_Config.cntk",
            testDataPath() + "/Control/InvalidImageSimpleCompositeAndBase64_Control.txt",
            testDataPath() + "/Control/InvalidImageSimpleCompositeAndBase64_Output.txt",
            "Composite_Test",
            "reader",
            4,
            4,
            1,
            1,
            1,
            0,
            1,
            false,
            false,
            true,
            additionalParameters);
    };

    test(
    {
        L"MapFile=\"$RootDir$/InvalidBase64ImageReaderSimple_map.txt\"",
        L"DeserializerType=\"Base64ImageDeserializer\"",
        L"maxErrors=4",
        L"useNumericSequenceKeys=true"
    });
};

BOOST_AUTO_TEST_CASE(Base64WithWriteIds)
{
    auto test = [this](std::vector<std::wstring> additionalParameters, multiset<string>& actualKeys)
    {
        shared_ptr<StreamMinibatchInputs> inputs = CreateStreamMinibatchInputs<float>(1, 1, false, true, true);
        shared_ptr<DataReader> reader = GetDataReader(testDataPath() + "/Config/ImageReaderSimple_Config.cntk",
            "Composite_Test", "reader", additionalParameters);

        reader->StartMinibatchLoop(2 /*mbSize*/, 0, inputs->GetStreamDescriptions(), 8/*epochSize*/);
        for (auto index = 0; reader->GetMinibatch(*inputs); index++)
        {
            for (const auto s : inputs->GetInput(L"features").pMBLayout->GetAllSequences())
            {
                auto key = inputs->m_getKeyById(s.seqId);
                actualKeys.insert(key);
            }
        }
    };

    {
        multiset<string> expectedKeys
        {
            "image0", "image0",
            "image1", "image1",
            "image2", "image2",
            "image3", "image3"
        };

        std::vector<std::wstring> additionalParameters
        {
            L"MapFile=\"$RootDir$/Base64WithStringIds_map.txt\"",
            L"DeserializerType=\"Base64ImageDeserializer\""
        };

        multiset<string> actualKeys;
        test(additionalParameters, actualKeys);
        BOOST_REQUIRE_EQUAL_COLLECTIONS(actualKeys.begin(), actualKeys.end(),
            expectedKeys.begin(), expectedKeys.end());

        additionalParameters.push_back(L"truncationLength=1");
        actualKeys.clear();
        test(additionalParameters, actualKeys);
        BOOST_REQUIRE_EQUAL_COLLECTIONS(actualKeys.begin(), actualKeys.end(),
            expectedKeys.begin(), expectedKeys.end());
    }

    {
        multiset<string> expectedKeys
        {
            "0", "0",
            "1", "1",
            "2", "2",
            "3", "3"
        };

        std::vector<std::wstring> additionalParameters
        {
            L"MapFile=\"$RootDir$/Base64ImageReaderSimple_map.txt\"",
            L"DeserializerType=\"Base64ImageDeserializer\"",
            L"useNumericSequenceKeys=true"
        };

        multiset<string> actualKeys;
        test(additionalParameters, actualKeys);
        BOOST_REQUIRE_EQUAL_COLLECTIONS(actualKeys.begin(), actualKeys.end(),
            expectedKeys.begin(), expectedKeys.end());

        additionalParameters.push_back(L"truncationLength=1");
        actualKeys.clear();
        test(additionalParameters, actualKeys);
        BOOST_REQUIRE_EQUAL_COLLECTIONS(actualKeys.begin(), actualKeys.end(),
            expectedKeys.begin(), expectedKeys.end());
    }
};

BOOST_AUTO_TEST_CASE(ImageAndImageReaderSimple)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/ImageDeserializers.cntk",
        testDataPath() + "/Control/ImageAndImageReaderSimple_Control.txt",
        testDataPath() + "/Control/ImageAndImageReaderSimple_Output.txt",
        "ImageAndImageReaderSimple_Test",
        "reader",
        4,
        4,
        1,
        2,
        2,
        0,
        1,
        false,
        false,
        true,
        { L"MapFile=\"$RootDir$/ImageReaderSimple_map.txt\"" });
}

BOOST_AUTO_TEST_CASE(ImageReaderBadMap)
{
    BOOST_REQUIRE_EXCEPTION(
        HelperRunReaderTest<float>(
            testDataPath() + "/Config/ImageReaderBadMap_Config.cntk",
            testDataPath() + "/Control/ImageReaderSimple_Control.txt",
            testDataPath() + "/Control/ImageReaderSimple_Output.txt",
            "Simple_Test",
            "reader",
            4,
            4,
            1,
            1,
            0,
            0,
            1),
            std::runtime_error,
            [](std::runtime_error const& ex) { return string("Invalid map file format, must contain 2 or 3 tab-delimited columns, line 2 in file ./ImageReaderBadMap_map.txt.") == ex.what(); });
}

BOOST_AUTO_TEST_CASE(ImageReaderBadLabel)
{
    BOOST_REQUIRE_EXCEPTION(
        HelperRunReaderTest<float>(
            testDataPath() + "/Config/ImageReaderBadLabel_Config.cntk",
            testDataPath() + "/Control/ImageReaderSimple_Control.txt",
            testDataPath() + "/Control/ImageReaderSimple_Output.txt",
            "Simple_Test",
            "reader",
            4,
            4,
            1,
            1,
            0,
            0,
            1),
            std::runtime_error,
            [](std::runtime_error const& ex) { return string("Cannot parse label value on line 1, second column, in file ./ImageReaderBadLabel_map.txt.") == ex.what(); });
}

BOOST_AUTO_TEST_CASE(ImageReaderLabelOutOfRange)
{
    BOOST_REQUIRE_EXCEPTION(
        HelperRunReaderTest<float>(
            testDataPath() + "/Config/ImageReaderLabelOutOfRange_Config.cntk",
            testDataPath() + "/Control/ImageReaderSimple_Control.txt",
            testDataPath() + "/Control/ImageReaderSimple_Output.txt",
            "Simple_Test",
            "reader",
            4,
            4,
            1,
            1,
            0,
            0,
            1),
            std::runtime_error,
            [](std::runtime_error const& ex) { return string("Image 'images/red.jpg' has invalid class id '10'. It is exceeding the label dimension of '4'. Line 3 in file ./ImageReaderLabelOutOfRange_map.txt.") == ex.what(); });
}

BOOST_AUTO_TEST_CASE(ImageReaderZip)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/ImageReaderZip_Config.cntk",
        testDataPath() + "/Control/ImageReaderZip_Control.txt",
        testDataPath() + "/Control/ImageReaderZip_Output.txt",
        "Zip_Test",
        "reader",
        4,
        4,
        1,
        1,
        0,
        0,
        1);
}

BOOST_AUTO_TEST_CASE(ImageReaderZipDuplicate)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/ImageDeserializers.cntk",
        testDataPath() + "/Control/ImageReaderZipDuplicate_Control.txt",
        testDataPath() + "/Control/ImageReaderZipDuplicate_Output.txt",
        "SimpleZip",
        "reader",
        4,
        4,
        1,
        1,
        0,
        0,
        1,
        false,
        false,
        true,
        { L"MapFile=\"$RootDir$/ImageReaderZipDuplicate_map.txt\"" });
}


BOOST_AUTO_TEST_CASE(ImageReaderZipMissingFile)
{
    BOOST_REQUIRE_EXCEPTION(
        HelperRunReaderTest<float>(
            testDataPath() + "/Config/ImageReaderZipMissing_Config.cntk",
            testDataPath() + "/Control/ImageReaderZip_Control.txt",
            testDataPath() + "/Control/ImageReaderZip_Output.txt",
            "ZipMissing_Test",
            "reader",
            4,
            4,
            1,
            1,
            0,
            0,
            1),
            std::runtime_error,
            [](std::runtime_error const& ex) { return string("Cannot retrieve image data for some sequences. For more detail, please see the log file.") == ex.what(); });
}

BOOST_AUTO_TEST_CASE(ImageReaderMultiView)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/ImageReaderMultiView_Config.cntk",
        testDataPath() + "/Control/ImageReaderMultiView_Control.txt",
        testDataPath() + "/Control/ImageReaderMultiView_Output.txt",
        "MultiView_Test",
        "reader",
        10,
        10,
        1,
        1,
        0,
        0,
        1);
}

BOOST_AUTO_TEST_CASE(ImageReaderIntensityTransform)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/ImageReaderIntensityTransform_Config.cntk",
        testDataPath() + "/Control/ImageReaderIntensityTransform_Control.txt",
        testDataPath() + "/Control/ImageReaderIntensityTransform_Output.txt",
        "IntensityTransform_Test",
        "reader",
        1,
        1,
        2,
        1,
        0,
        0,
        1);
}


BOOST_AUTO_TEST_CASE(ImageReaderColorTransform)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/ImageReaderColorTransform_Config.cntk",
        testDataPath() + "/Control/ImageReaderColorTransform_Control.txt",
        testDataPath() + "/Control/ImageReaderColorTransform_Output.txt",
        "ColorTransform_Test",
        "reader",
        1,
        1,
        2,
        1,
        0,
        0,
        1);
}

BOOST_AUTO_TEST_CASE(ImageReaderGrayscale)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/ImageReaderGrayscale_Config.cntk",
        testDataPath() + "/Control/ImageReaderGrayscale_Control.txt",
        testDataPath() + "/Control/ImageReaderGrayscale_Output.txt",
        "Grayscale_Test",
        "reader",
        1,
        1,
        1,
        1,
        0,
        0,
        1);
}

BOOST_AUTO_TEST_CASE(ImageReaderMissingImage)
{
    BOOST_REQUIRE_EXCEPTION(
        HelperRunReaderTest<float>(
        testDataPath() + "/Config/ImageReaderMissingImage_Config.cntk",
        testDataPath() + "/Control/ImageReaderSimple_Control.txt",
        testDataPath() + "/Control/ImageReaderSimple_Output.txt",
        "MissingImage_Test",
        "reader",
        4,
        4,
        1,
        1,
        0,
        0,
        1),
        std::runtime_error,
        [](const std::runtime_error& ex) { return string("Cannot open file 'imageDoesNotExists/black.jpg'") == ex.what(); });
}

BOOST_AUTO_TEST_CASE(ImageReaderEmptyTransforms)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/ImageTransforms_Config.cntk",
        testDataPath() + "/Control/ImageTransforms_Control.txt",
        testDataPath() + "/Control/ImageTransforms_Output.txt",
        "SameShapeEmptyTransforms_Test",
        "reader",
        2,
        1,
        1,
        1,
        0,
        0,
        1);
}

BOOST_AUTO_TEST_CASE(ImageReaderInvalidEmptyTransforms)
{
    BOOST_REQUIRE_EXCEPTION(
        HelperRunReaderTest<float>(
        testDataPath() + "/Config/ImageTransforms_Config.cntk",
        testDataPath() + "/Control/ImageReaderInvalidEmptyTransforms.txt",
        testDataPath() + "/Control/ImageReaderInvalidEmptyTransforms_Output.txt",
        "DifferentShapeEmptyTransforms_Test",
        "reader",
        2,
        2,
        1,
        1,
        0,
        0,
        1),
        std::runtime_error,
        [](const std::runtime_error& ex)
        {
            return string("Packer currently does not support samples with varying shapes."
                "Please make sure there is a transform that unifies the shape of samples for input stream 'features' "
                "or the deserializer provides samples with the same shape.") == ex.what();
        });
}

BOOST_AUTO_TEST_CASE(ImageReaderMissingScaleTransforms)
{
    BOOST_REQUIRE_EXCEPTION(
        HelperRunReaderTest<float>(
        testDataPath() + "/Config/ImageTransforms_Config.cntk",
        testDataPath() + "/Control/ImageReaderMissingScaleTransforms.txt",
        testDataPath() + "/Control/ImageReaderMissingScaleTransforms_Output.txt",
        "NoScaleTransforms_Test",
        "reader",
        2,
        2,
        1,
        1,
        0,
        0,
        1),
        std::runtime_error,
        [](const std::runtime_error& ex)
        {
            return string("Packer currently does not support samples with varying shapes."
                "Please make sure there is a transform that unifies the shape of samples"
                " for input stream 'features' or the deserializer provides samples with the same shape.") == ex.what();
        });
}

BOOST_AUTO_TEST_SUITE_END()

namespace
{
    // Test with not set data directory.
    struct EmptyDataDirFixture : ReaderFixture
    {
        EmptyDataDirFixture() : ReaderFixture("/.") { }
    };

    BOOST_FIXTURE_TEST_SUITE(ReaderTestSuite, EmptyDataDirFixture)

    BOOST_AUTO_TEST_CASE(ImageReader3DotsSyntaxInMapFile)
    {
        auto testDir = testDataPath();
        std::wstring mapFileLocaton = std::wstring(testDir.begin(), testDir.end()) + L"/Data/ImageReader3Dots_map.txt";
        HelperRunReaderTest<float>(
            testDataPath() + "/Config/ImageDeserializers.cntk",
            testDataPath() + "/Control/ImageReader3DotsSyntaxInMapFile_Control.txt",
            testDataPath() + "/Control/ImageReader3DotsSyntaxInMapFile_Output.txt",
            "3DotsExpansionTest",
            "reader",
            1,
            2,
            1,
            1,
            1,
            0,
            1,
            false,
            true,
            true,
            { L"MapFile=\"" + mapFileLocaton + L"\"" });
    }

    BOOST_AUTO_TEST_SUITE_END()
}

}}}}

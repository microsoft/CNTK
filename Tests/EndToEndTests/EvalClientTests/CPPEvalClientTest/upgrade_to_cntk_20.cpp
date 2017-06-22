#include "stdafx.h"

#include <Eval.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace Microsoft::MSR::CNTK;

struct Mat {
    vector<float> data;
    int cols;
    int rows;
};

Mat ImreadPfm(const std::string& file_name) {
    ifstream in(file_name.c_str(), ios_base::binary);
    string format;
    in >> format;
    if (format != "Pf") {
        throw std::logic_error("Wrong image format");
    }
    int rows, cols;
    float scale;
    in >> cols >> rows >> scale;
    if (scale > 0) {
        throw std::logic_error("Wrong image format");
    }

    while (in) {
        char c;
        in.read(&c, sizeof(c));
        if (c == '\n')
            break;
    }

    Mat img;
    img.cols = cols;
    img.rows = rows;
    img.data.resize(cols * rows);
    for (int y = img.rows - 1; y >= 0; --y) {
        float * row = (float*)(&img.data[y * cols]);
        in.read((char*)row, sizeof(*row) * img.cols);
    }
    if (abs(scale) != 1.f) {
        for (float & p : img.data) {
            p *= abs(scale);
        }
    }
    return img;
}



class Model {
public:
    Model()
        : eval_(0) {
    }

    ~Model() {
        Release();
    }

    void Load(const std::string& model_file_name) {
        GetEvalF(&eval_);
        if (eval_ == 0)
            throw std::runtime_error("Cannot create CNTK evaluator.");

        eval_->Init("numCPUThreads=1");

        stringstream init_string;
        init_string << "modelPath=\"" << model_file_name << "\"\n";
        eval_->CreateNetwork(init_string.str());
    }

    vector<float> Predict(Mat& image) {
        vector<float> output;

        std::map<std::wstring, std::vector<float>*> input_map;
        std::map<std::wstring, std::vector<float>*> output_map;

        // get the model's layers dimensions
        std::map<std::wstring, size_t> in_dims;
        std::map<std::wstring, size_t> out_dims;
        eval_->GetNodeDimensions(in_dims, NodeGroup::nodeInput);
        eval_->GetNodeDimensions(out_dims, NodeGroup::nodeOutput);

        // assume we have only one input and output.
        if (in_dims.empty() || out_dims.empty())
            throw invalid_argument("Empty input or output network layers");

        input_map[in_dims.begin()->first] = &image.data;
        output_map[out_dims.begin()->first] = &output;

        eval_->Evaluate(input_map, output_map);

        printf("Prediction:\n");
        for (auto & o : output)
            printf("%.12f, ", o );
        printf("\n");

        return output;
    }

    int Classify(Mat& image) {
        vector<float> z = Predict(image);
        int best_label = -1;
        float best_value = -1e9f;
        for (size_t i = 0; i < z.size(); ++i) {
            if (z[i] > best_value) {
                best_value = z[i];
                best_label = (int)i;
            }
        }

        return int(best_label);
    }

    int Classify(const std::string& file_name) {
        Mat image = ImreadPfm(file_name);
        return Classify(image);
    }

    void Release() {
        if (eval_) {
            eval_->Destroy();
            eval_ = 0;
        }
    }

    IEvaluateModel<float> * eval_;
};

void CheckClassification(Model& model, const std::string& sample_file_name, int expected_label) {
    int label = model.Classify(sample_file_name);
    cout << sample_file_name << ", expected label: " << expected_label << ", actual label " << label << (expected_label == label ? "    OK" : "    ERROR") << endl;
}

void TestModel1() {
    // This model was converted from an old CNTK released in Feb. 2016. 
    // I had trouble with this conversion and needed to patch CNTK 1.7, as described here https://github.com/Microsoft/CNTK/issues/865
    // The patched CNTK 1.7 works with this model as I expect.
    // Original CNTK 1.7 and CNTK 2.0 predict the same, although not what I expect.

    string model_dir = "./models/Audi_C7_DL_SL_SmallPressurePlate1/";
    Model model;
    model.Load(model_dir + "Network.dat");

    CheckClassification(model, model_dir + "samples/sample1.pfm", 1);
    CheckClassification(model, model_dir + "samples/sample2.pfm", 0);
}

void TestModel2() {
    // This model was created with CNTK 1.7 and works fine in all versions.

    string model_dir = "./models/BMW_iCond_ImmersionTubeTop/";
    Model model;
    model.Load(model_dir + "Network.dat");

    CheckClassification(model, model_dir + "samples/sample1.pfm", 0);
    CheckClassification(model, model_dir + "samples/sample2.pfm", 0);
}

void TestModel3() {
    // This model was created with CNTK 1.7 and predict different results in CNTK 1.7 and CNTK 2.0.

    string model_dir = "./models/BMW_iCond_ImmersionTubeTop2/";
    Model model;
    model.Load(model_dir + "Network.dat");

   /* CheckClassification(model, model_dir + "samples/sample1.pfm", 0);
    CheckClassification(model, model_dir + "samples/sample2.pfm", 0);*/
    CheckClassification(model, model_dir + "samples/sample3.pfm", 0); // This one differs.
}

#if 0
int _tmain(int argc, _TCHAR* argv[])
{
    /*TestModel1();
    TestModel2();*/
    TestModel3();
	return 0;
}
#endif


#include <thread>
#include "CNTKLibrary.h"

using namespace CNTK;


void RunKevinEvaluation(int thId, FunctionPtr model, const DeviceDescriptor& device)
{
    auto cosine_distance = model->FindByName(L"query_positive_document_cosine_distance");
    auto evalFunc = AsComposite(cosine_distance);
    auto queryInput = evalFunc->Arguments()[0];
    auto documentInput = evalFunc->Arguments()[1];

    auto queryInputShape = queryInput.Shape();
    auto documentInputShape = documentInput.Shape();

    auto sampleSize = 49293;
    auto querySeqSize = sampleSize * 5; //246465
    std::vector<int> queryNonZeroIndices { 1855, 3314, 49360, 49512, 50410, 51064, 51302, 98632, 98633, 147934, 148430, 197234, 197359, 197360, 197754 }; 
    std::vector<float> querySequence(querySeqSize, 0);
    if (queryNonZeroIndices.size() != 15)
        printf("ERROR: wrong query non-zero index values\n");
    for (int i = 0; i < queryNonZeroIndices.size(); i++)
        querySequence[queryNonZeroIndices[i]] = 1;

    auto documentSeqSize = sampleSize * 15; //739395
    std::vector<int> documentNonZeroIndices { 23, 105, 106, 235, 49729, 49767, 49986, 50003, 50716, 50994, 98586, 98587, 98694, 148140, 148159, 148160, 197218, 197219, 246692, 246710, 246711, 246712, 246713, 246945, 247224, 247302, 247303, 253437, 296860, 345390, 345391, 345899, 346531, 347515, 348586, 351420, 394543, 394830, 394897, 395243, 395666, 396338, 443747, 443752, 443753, 443754, 443963, 443964, 493072, 493480, 542271, 591748, 592564, 592603, 592652, 641306, 641307, 690120, 690178, 690179, 690180, 690451, 690585, 690638, 691139, 692107, 693674, 697594, 698253 };
    std::vector<float> documentSequence(documentSeqSize, 0);
    if (documentNonZeroIndices.size() != 69)
        printf("ERROR: wrong document non-zero index values\n");
    for (int i = 0; i < documentNonZeroIndices.size(); i++)
        documentSequence[documentNonZeroIndices[i]] = 1;


    auto queryInputValue = Value::CreateSequence<float>(queryInputShape, querySequence, device);
    auto documentInputValue = Value::CreateSequence<float>(documentInputShape, documentSequence, device);

    std::unordered_map<Variable, ValuePtr> inputs {{queryInput, queryInputValue}, {documentInput, documentInputValue}};

    std::unordered_map<Variable, ValuePtr> outputs { {evalFunc->Output(), nullptr}};

    fprintf(stderr, "%d evaluate starts.\n", thId);
    evalFunc->Evaluate(inputs, outputs, device);

    fprintf(stderr,"%d evaluate complete.\n", thId);

}

void KevinPan()
{
    auto device = DeviceDescriptor::CPUDevice();
    auto model = Function::Load(L"C:\\CNTKMisc\\KevinPan-Memory\\trained_model\\cntk_2_0_6_layer_self_attention_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_01_model_batch_600000_38951002.dnn", device);

    int threadCount = 20;
    std::vector<std::thread> threadList(threadCount);
    std::vector<FunctionPtr> models(threadCount);
    for (int th = 0; th < threadCount; ++th)
    {
        models[th] = model->Clone(ParameterCloningMethod::Share);
    }

    for (int th = 0; th < threadCount; ++th)
    {
        auto evalFunc = models[th];
        threadList[th] = std::thread(RunKevinEvaluation, th, evalFunc, device);
    }

    for (int th = 0; th < threadCount; ++th)
    {
        threadList[th].join();
        fprintf(stderr, "thread %d joined.\n", th);
        fflush(stderr);
    }

}


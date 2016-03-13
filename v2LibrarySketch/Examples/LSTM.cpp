#include "Model.h"
#include "SGDOptimizer.h"
#include <assert.h>

using namespace CNTK;

std::pair<Function, Function> LSTMPComponentWithSelfStab(Variable input,
                                                 Variable prevOutput,
                                                 Variable prevCell,
                                                 size_t outputDim,
                                                 size_t cellDim)
{
    size_t inputDim = input.Shape().back();
    Variable Wxo = Variable({ cellDim, inputDim }, L"WxoParam");
    Variable Wxi = Variable({ cellDim, inputDim }, L"WxiParam");
    Variable Wxf = Variable({ cellDim, inputDim }, L"WxfParam");
    Variable Wxc = Variable({ cellDim, inputDim }, L"WxcParam");

    Variable Bo = Variable({ cellDim }, L"BoParam");
    Variable Bc = Variable({ cellDim }, L"BcParam");
    Variable Bi = Variable({ cellDim }, L"BiParam");
    Variable Bf = Variable({ cellDim }, L"BfParam");

    Variable Whi = Variable({ cellDim, outputDim }, L"WhiParam");
    Variable Wci = Variable({ cellDim }, L"WciParam");

    Variable Whf = Variable({ cellDim, outputDim }, L"WhfParam");
    Variable Wcf = Variable({ cellDim }, L"WcfParam");

    Variable Who = Variable({ cellDim, outputDim }, L"WhoParam");
    Variable Wco = Variable({ cellDim }, L"WcoParam");

    Variable Whc = Variable({ cellDim, outputDim }, L"WhcParam");

    Variable Wmr = Variable({ outputDim, cellDim }, L"WmrParam");

    Variable sWxo = Variable({ 1, 1 }, L"sWxoParam");
    Variable sWxi = Variable({ 1, 1 }, L"sWxiParam");
    Variable sWxf = Variable({ 1, 1 }, L"sWxfParam");
    Variable sWxc = Variable({ 1, 1 }, L"sWxcParam");

    Variable sWhi = Variable({ 1, 1 }, L"sWhiParam");
    Variable sWci = Variable({ 1, 1 }, L"sWciParam");

    Variable sWhf = Variable({ 1, 1 }, L"sWhfParam");
    Variable sWcf = Variable({ 1, 1 }, L"sWcfParam");
    Variable sWho = Variable({ 1, 1 }, L"sWhoParam");
    Variable sWco = Variable({ 1, 1 }, L"sWcoParam");
    Variable sWhc = Variable({ 1, 1 }, L"sWhcParam");

    Variable sWmr = Variable({ 1, 1 }, L"sWmrParam");

    Function expsWxo = Exp(sWxo);
    Function expsWxi = Exp(sWxi);
    Function expsWxf = Exp(sWxf);
    Function expsWxc = Exp(sWxc);

    Function expsWhi = Exp(sWhi);
    Function expsWci = Exp(sWci);

    Function expsWhf = Exp(sWhf);
    Function expsWcf = Exp(sWcf);
    Function expsWho = Exp(sWho);
    Function expsWco = Exp(sWco);
    Function expsWhc = Exp(sWhc);

    Function expsWmr = Exp(sWmr);

    Variable outputPlaceholder = Variable({outputDim}, L"outputPlaceHolder");
    Function dh = PastValue(prevOutput, outputPlaceholder, L"OutputPastValue");
    Variable ctPlaceholder = Variable({ cellDim }, L"ctPlaceHolder");
    Function dc = PastValue(prevCell, ctPlaceholder, L"CellPastValue");

    Function Wxix = Times(Wxi, Scale(expsWxi.Output(), input).Output());
    Function Whidh = Times(Whi, Scale(expsWhi.Output(), dh.Output()).Output());
    Function Wcidc = DiagTimes(Wci, Scale(expsWci.Output(), dc.Output()).Output());

    Function it = Sigmoid(Plus(Plus(Plus(Wxix.Output(), Bi).Output(), Whidh.Output()).Output(), Wcidc.Output()).Output());

    Function Wxcx = Times(Wxc, Scale(expsWxc.Output(), input).Output());
    Function Whcdh = Times(Whc, Scale(expsWhc.Output(), dh.Output()).Output());
    Function bit = ElementTimes(it.Output(), Tanh(Plus(Wxcx.Output(), Plus(Whcdh.Output(), Bc).Output()).Output()).Output());

    Function Wxfx = Times(Wxf, Scale(expsWxf.Output(), input).Output());
    Function Whfdh = Times(Whf, Scale(expsWhf.Output(), dh.Output()).Output());
    Function Wcfdc = DiagTimes(Wcf, Scale(expsWcf.Output(), dc.Output()).Output());

    Function ft = Sigmoid(Plus(Plus(Plus(Wxfx.Output(), Bf).Output(), Whfdh.Output()).Output(), Wcfdc.Output()).Output());

    Function bft = ElementTimes(ft.Output(), dc.Output());

    Function ct = Plus(bft.Output(), bit.Output());

    Function Wxox = Times(Wxo, Scale(expsWxo.Output(), input).Output());
    Function Whodh = Times(Who, Scale(expsWho.Output(), dh.Output()).Output());
    Function Wcoct = DiagTimes(Wco, Scale(expsWco.Output(), ctPlaceholder).Output());

    Function ot = Sigmoid(Plus(Plus(Plus(Wxox.Output(), Bo).Output(), Whodh.Output()).Output(), Wcoct.Output()).Output());

    Function mt = ElementTimes(ot.Output(), Tanh(ct.Output()).Output());

    Function output = Times(Wmr, Scale(expsWmr.Output(), mt.Output()).Output());
    output.SetInput(outputPlaceholder, output.Output());
    output.SetInput(ctPlaceholder, ct.Output());

    return{ output, ct };
}

Function LSTMNet(size_t inputDim, size_t cellDim, size_t hiddenDim, size_t numOutputClasses, size_t numLSTMLayers)
{
    Variable features({ inputDim }, L"features");

    Variable layer1PrevOutput({hiddenDim}, L"layer1PrevOutput");
    Variable layer1PrevCell({ cellDim }, L"layer1PrevCell");
    auto LSTMoutputs1 = LSTMPComponentWithSelfStab(features, layer1PrevOutput, layer1PrevCell, hiddenDim, cellDim);

    Variable layer2PrevOutput({ hiddenDim }, L"layer2PrevOutput");
    Variable layer2PrevCell({ cellDim }, L"layer2PrevCell");
    auto LSTMoutputs2 = LSTMPComponentWithSelfStab(LSTMoutputs1.first.Output(), layer2PrevOutput, layer2PrevCell, hiddenDim, cellDim);

    Variable layer3PrevOutput({ hiddenDim }, L"layer3PrevOutput");
    Variable layer3PrevCell({ cellDim }, L"layer3PrevCell");
    auto LSTMoutputs3 = LSTMPComponentWithSelfStab(LSTMoutputs2.first.Output(), layer3PrevOutput, layer3PrevCell, hiddenDim, cellDim);

    Variable W({ numOutputClasses, hiddenDim }, L"OutputWParam");
    Variable b({ numOutputClasses }, L"OutputBParam");

    Variable sW({ 1, 1 }, L"sWParam");
    Function expsW = Exp(sW);

    Function LSTMoutputW = Plus(Times(W, Scale(expsW.Output(), LSTMoutputs3.first.Output()).Output()).Output(), b);

    return LSTMoutputW;
}

void TrainFeedForwardClassifier(Reader trainingDataReader)
{
    const size_t inputDim = 87;
    const size_t numOutputClasses = 9404;
    const size_t numLSTMLayers = 3;
    const size_t cellDim = 1024;
    const size_t hiddenDim = 512;

    Function netOutputNode = LSTMNet(inputDim, cellDim, hiddenDim, numOutputClasses, numLSTMLayers);

    Variable labels = Variable({ numOutputClasses }, L"labels");
    Function trainingLossNode = CNTK::CrossEntropyWithSoftmax(netOutputNode.Output(), labels, L"lossFunction");
    Function predictionNode = CNTK::PredictionError(netOutputNode.Output(), labels, L"predictionError");

    // Initialize parameters
    auto timesParameters = predictionNode.Inputs(L"W*Param");
    auto biasParameters = predictionNode.Inputs(L"B*Param");
    std::unordered_map<Variable, Value> initialParameters;
    for each (auto param in timesParameters)
        initialParameters.insert({ param, CNTK::RandomNormal(param.Shape(), 0, 1) });

    for each (auto param in biasParameters)
        initialParameters.insert({ param, CNTK::Const(param.Shape(), 0) });

    Model LSTMClassifier({ trainingLossNode, predictionNode }, initialParameters);

    size_t momentumTimeConstant = 1024;
    // Train with 100000 samples; checkpiint every 10000 samples
    TrainingControl driver = CNTK::BasicTrainingControl(100000, 10000, L"LSTM.net");
    SGDOptimizer LSTMClassifierOptimizer(LSTMClassifier, { CNTK::MomentumLearner(predictionNode.Inputs(L"Param"), momentumTimeConstant), CNTK::PastValueLearner(predictionNode.Children(L"PastValue")) });

    Reader textReader = CNTK::TextReader(/*reader config*/);
    LSTMClassifierOptimizer.TrainCorpus(textReader, trainingLossNode, driver);
}


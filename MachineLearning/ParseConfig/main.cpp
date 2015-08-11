// main.cpp -- main function for testing config parsing

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "ConfigEvaluator.h"

using namespace Microsoft::MSR::CNTK;

#ifndef let
#define let const auto
#endif

wstring standardFunctions =
L"Format(value,format) = new StringFunction [ what = 'format' ; arg = value ; how = format ] \n"
L"Print(value) = new PrintAction [ what = value ] \n"
L""
L""
L""
L""
L""
L""
L""
;

wstring computationNodes =
L"Mean(z) = new ComputationNode [ class = 'MeanNode' ; arg = z ] \n"
L"InvStdDev(z) = new ComputationNode [ class = 'InvStdDevNode' ; arg = z ] \n"
L"PerDimMeanVarNormalization(feat,mean,invStdDev) = new ComputationNode [ class = 'PerDimMeanVarNormalizationNode' ; arg1 = feat ; arg2 = mean ; arg3 = invStdDev ] \n"
L"Parameter(outD, inD) = new ComputationNode [ class = 'LearnableParameterNode' ; outDim = outD ; inDim = inD ] \n"
L"Input(dim) = Parameter(dim,1)   // TODO: for now \n"
L"RowSlice(firstRow, rows, features) = new ComputationNode [ class = 'RowSliceNode' ; arg = features ; first = firstRow ; num = rows ] \n"
L"Sigmoid(z) = new ComputationNode [ class = 'SigmoidNode' ; arg = z ] \n"
L"Log(z) = new ComputationNode [ class = 'LogNode' ; arg = z ] \n"
L"CrossEntropyWithSoftmax(labels, outZ) = new ComputationNode [ class = 'CrossEntropyWithSoftmaxNode' ; left = labels ; right = outZ ] \n"
L"ErrorPrediction(labels, outZ) = new ComputationNode [ class = 'ErrorPredictionNode' ; left = labels ; right = outZ ] \n"
L" \n"
L" \n"
L" \n"
L" \n"
L" \n"
L" \n"
L" \n"
;

wstring commonMacros =  // TODO: rename rows and cols to inDim and outDim or vice versa, whichever it is
L"BFF(in, rows, cols) = [ B = Parameter(rows, 1/*init = fixedvalue, value = 0*/) ; W = Parameter(rows, cols) ; z = W*in+B ] \n"
L"SBFF(in, rows, cols) = [ Eh = Sigmoid(BFF(in, rows, cols).z) ] \n "
L"MeanVarNorm(feat) = PerDimMeanVarNormalization(feat, Mean(feat), InvStdDev(feat)) \n"
L"LogPrior(labels) = Log(Mean(labels)) \n"
L""
L""
;



int wmain(int /*argc*/, wchar_t* /*argv*/[])
{
    // there is record of parameters
    // user wants to get a parameter
    // double x = config->GetParam("name", 0.0);
    try
    {
        //let parserTest = L"a=1\na1_=13;b=2 // cmt\ndo = new PrintAction [message='hello'];do1=(print\n:train:eval) ; x = array[1..13] (i=>1+i*print.message==13*42) ; print = new PrintAction [ message = 'Hello World' ]";
        let parserTest1 = L"do3 = new LearnableParameter [ inDim=13; outDim=42 ] * new InputValue [ ] + new LearnableParameter [ outDim=42 ]\n"
            L"do2 = array [1..10] (i=>i*i) ;"
            L"do = new PrintAction [ what = 'abc' ] ;"
            L"do5 = new PrintAction [ what = new StringFunction [ x = 13 ; y = 42 ; what = 'format' ; how = '.2' ; arg = x*y ] ] ;"
            L"do4 = new PrintAction [ what = \"new StringFunction [ what = 'format' ; how = '.2' ; arg = '13 > 42' ]\" ] ;"
            L"do1 = new PrintAction [ what = if 13 > 42 || 12 > 1 then 'Hello World' + \"!\" else 'Oops?']";
        let parserTest2 = L"i2s(i) = new StringFunction [ what = 'format' ; arg = i ; how = '.2' ] ; print(s) = new PrintAction [ what = s ] ; do = print('result=' + i2s((( [ v = (i => i + delta) ].v(5)))+13)) ; delta = 42 ";
        let parserTest3 = L"do = new PrintAction [ what = val ] ; val=1+2*3; text = 'hello'+' world' ";
        let parserTest4 = L"do = new PrintAction [ what = new StringFunction [ what = 'format' ; arg = (13:(fortytwo:1):100) ; how = '' ] ];fortytwo=42 ";
        let parserTest5 = L"do = new PrintAction [ what = val ] ; val=if !false then 42 else -+-++-13:[a='a';b=42]:+14; arr = array [1..10] (i => 2*i) ";
        let parserTest6 = L"do = new PrintAction [ what = arg ] ; N = 5 ; arr = array [1..N] (i => if i < N then arr[i+1]*i else N) ; arg = arr ";
        let parserTest7 = L"do = new PrintAction [ what = val ] ; val = [ v = (i => i + offset) ].v(42) ; offset = 13 ";
        let parserTest8 = L" \n"
                          L"do = Print(val) \n"
                          L"val = new NDLNetwork [\n"
                          L"  featDim=40*31 ; labelDim=9000 ; hiddenDim=2048 ; numHiddenLayers = 7 \n"
                          L"  myFeatures = Input(featDim) ; myLabels = Input(labelDim) \n"
                          L"  featNorm = MeanVarNorm(myFeatures) \n"
                          L"  HiddenStack(layer) = if layer > 1 then SBFF(HiddenStack(layer - 1).Eh, hiddenDim, hiddenDim) else SBFF(featNorm, hiddenDim, featDim) \n"
                          L"  outLayer = BFF(HiddenStack(numHiddenLayers).Eh, labelDim, hiddenDim) \n"
                          L"  outZ = outLayer.z \n"
                          L"  CE = CrossEntropyWithSoftmax(myLabels, outZ) \n"
                          L"  Err = ErrorPrediction(myLabels, outZ) \n"
                          L"  logPrior = LogPrior(myLabels) \n"
                          L"  ScaledLogLikelihood = outZ - logPrior \n"
                          L"]\n";
        let parserTest9 = L"do = new PrintAction [ what = val ] ; fac(i) = if i > 1 then fac(i-1)*i else i ; val = fac(5) ";
        let parserTest10 = L"do = new PrintAction [ what = val ] ; fib(n) = [ vals = array[1..n] (i => if i < 3 then 1 else vals[i-1]+vals[i-2]) ].vals ; val = fib(10) ";
        parserTest1; parserTest2; parserTest3; parserTest4; parserTest5; parserTest6; parserTest7; parserTest8; parserTest9; parserTest10;
        let parserTest = parserTest8;
        let expr = ParseConfigString(standardFunctions + computationNodes + commonMacros + parserTest);
        //expr->Dump();
        Do(expr);
        //ParseConfigFile(L"c:/me/test.txt")->Dump();
    }
    catch (const ConfigError & err)
    {
        err.PrintError();
    }
    return EXIT_SUCCESS;
}

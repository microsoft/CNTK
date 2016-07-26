// BrainScriptTest.cpp -- some tests

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "BrainScriptEvaluator.h"
#include "BrainScriptParser.h"

#ifndef let
#define let const auto
#endif

namespace Microsoft { namespace MSR { namespace BS {

using namespace std;
using namespace msra::strfun;

// Note: currently this seems to be the master copy; got to check whether the other one was also changed

//extern wstring standardFunctions, computationNodes, commonMacros;

#if 1 // TODO: these may be newer, merge into Experimentalthingy

static wstring standardFunctions =
    L"Print(value, format='') = new PrintAction [ what = value /*; how = format*/ ] \n"
    L"Fail(msg) = new FailAction [ what = msg ] \n"
    L"RequiredParameter(message) = Fail('RequiredParameter: ' + message) \n"
    L"Format(value, format) = new StringFunction [ what = 'Format' ; arg = value ; how = format ] \n"
    L"Replace(s, from, to) = new StringFunction [ what = 'Replace' ; arg = s ; replacewhat = from ; withwhat = to ] \n"
    L"Substr(s, begin, num) = new StringFunction [ what = 'Substr' ; arg = s ; pos = begin ; chars = num ] \n"
    L"Chr(c) = new StringFunction [ what = 'Chr' ;  arg = c ] \n"
    L"Floor(x)  = new NumericFunction [ what = 'Floor' ;  arg = x ] \n"
    L"Length(x) = new NumericFunction [ what = 'Length' ; arg = x ] \n"
    L"Ceil(x) = -Floor(-x) \n"
    L"Round(x) = Floor(x+0.5) \n"
    L"Abs(x) = if x >= 0 then x else -x \n"
    L"Sign(x) = if x > 0 then 1 else if x < 0 then -1 else 0 \n"
    L"Min(a,b) = if a < b then a else b \n"
    L"Max(a,b) = if a > b then a else b \n"
    L"Fac(n) = if n > 1 then Fac(n-1) * n else 1 \n";

static wstring computationNodes = // BUGBUG: optional args not working yet, some scope problem causing a circular reference
    L"Mean(z, tag='') = new ComputationNode [ class = 'MeanNode' ; inputs = z /* ; tag = tag */ ]\n"
    L"InvStdDev(z, tag='') = new ComputationNode [ class = 'InvStdDevNode' ; inputs = z /* ; tag = tag */ ]\n"
    L"PerDimMeanVarNormalization(feat,mean,invStdDev, tag='') = new ComputationNode [ class = 'PerDimMeanVarNormalizationNode' ; inputs = feat:mean:invStdDev /* ; tag = tag */ ]\n"
    L"Parameter(outD, inD, tag='parameter') = new ComputationNode [ class = 'LearnableParameterNode' ; outDim = outD ; inDim = inD /*; tag = tag*/ ]\n"
    L"Input(dim,tag='features') = Parameter(dim,1,tag=tag)   // TODO: for now \n"
    L"RowSlice(firstRow, rows, features, tag='') = new ComputationNode [ class = 'RowSliceNode' ; inputs = features ; first = firstRow ; num = rows /* ; tag = tag */ ]\n"
    L"Delay(in, delay, tag='') = new RecurrentComputationNode [ class = 'DelayNode' ; inputs = in ; deltaT = -delay /* ; tag = tag */ ]\n"
    L"Sigmoid(z, tag='') = new ComputationNode [ class = 'SigmoidNode' ; inputs = z /* ; tag = tag */ ]\n"
    L"Log(z, tag='') = new ComputationNode [ class = 'LogNode' ; inputs = z /* ; tag = tag */ ]\n"
    L"CrossEntropyWithSoftmax(labels, outZ, tag='') = new ComputationNode [ class = 'CrossEntropyWithSoftmaxNode' ; inputs = labels:outZ /* ; tag = tag */ ]\n"
    L"ErrorPrediction(labels, outZ, tag='') = new ComputationNode [ class = 'ErrorPredictionNode' ; inputs = labels:outZ /* ; tag = tag */ ]\n";

static wstring commonMacros = // TODO: rename rows and cols to inDim and outDim or vice versa, whichever it is
    L"BFF(in, rows, cols) = [ B = Parameter(rows, 1/*init = fixedvalue, value = 0*/) ; W = Parameter(rows, cols) ; z = W*in+B ] \n"
    L"SBFF(in, rows, cols) = [ Eh = Sigmoid(BFF(in, rows, cols).z) ] \n "
    L"MeanVarNorm(feat) = PerDimMeanVarNormalization(feat, Mean(feat), InvStdDev(feat)) \n"
    L"LogPrior(labels) = Log(Mean(labels)) \n";

#endif

void SomeTests()
{
    try
    {
        // collecting all sorts of test cases here
        const wchar_t* parserTests[] =
            {
                L"do = Parameter(13,42) * Input(42) + Parameter(13,1)",
                L"do = Print(array [1..10] (i=>i*i))",
                L"do = new PrintAction [ what = 'abc' ]",
                L"do = Print(new StringFunction [ x = 13 ; y = 42 ; what = 'Format' ; how = '.2' ; arg = x*y ])",
                L"do = Print(\"new StringFunction [ what = 'Format' ; how = '.2' ; arg = '13 > 42' ]\")",
                L"do = new PrintAction [ what = if 13 > 42 || 12 > 1 then 'Hello World' + \"!\" else 'Oops?']",
                L"i2s(i) = new StringFunction [ what = 'Format' ; arg = i ; how = '.2' ] ; do = Print('result=' + i2s((( [ v = (i => i + delta) ].v(5)))+13)) ; delta = 42 ",
                L"do = Print(1+2*3) : Print('hello'+' world')",
                L"do = Print(Format( (13:(fortytwo:1):100), '')) ; fortytwo=42 ",
                L"do = Print(val) ; val=if !false then 42 else -+-++-13:[a='a';b=42]:+14; arr = array [1..10] (i => 2*i)",
                L"do = Print(arg) ; N = 5 ; arr = array [1..N] (i => if i < N then arr[i+1]*i else N) ; arg = arr ",
                L"do = Print(val) ; val = [ v = (i => i + offset) ].v(42) ; offset = 13 ",
                // #12: DNN with recursion
                L"do = Print(val) \n"
                L"val = new NDLComputationNetwork [\n"
                L"  featDim=40*31 ; labelDim=9000 ; hiddenDim=2048 ; numHiddenLayers = 3 \n"
                L"  myFeatures = Input(featDim) ; myLabels = Input(labelDim) \n"
                L"  featNorm = MeanVarNorm(myFeatures) \n"
                L"  HiddenStack(layer) = if layer > 1 then SBFF(HiddenStack(layer - 1).Eh, hiddenDim, hiddenDim) else SBFF(featNorm, hiddenDim, featDim) \n"
                L"  outLayer = BFF(HiddenStack(numHiddenLayers).Eh, labelDim, hiddenDim) \n"
                L"  outZ = outLayer.z \n"
                L"  CE = CrossEntropyWithSoftmax(myLabels, outZ) \n"
                L"  Err = ErrorPrediction(myLabels, outZ) \n"
                L"  logPrior = LogPrior(myLabels) \n"
                L"  ScaledLogLikelihood = outZ - logPrior \n"
                L"]\n",
                // #13: factorial
                L"do = Print(fac(5)) ; fac(i) = if i > 1 then fac(i-1)*i else 1 ",
                // #14: Fibonacci sequence with memoization
                L"do = Print(fibs(10)) ; fibs(n) = [ vals = array[1..n] (i => if i < 3 then i-1 else vals[i-1]+vals[i-2]) ].vals[n] ",
                // #15: DNN with array
                L"do = Print(val) \n"
                L"val = new NDLComputationNetwork [\n"
                L"  featDim=40*31 ; labelDim=9000 ; hiddenDim=2048 ; numHiddenLayers = 3 \n"
                L"  myFeatures = Input(featDim, tag='features') ; myLabels = Input(labelDim, tag='labels') \n"
                L"  featNorm = MeanVarNorm(myFeatures) \n"
                L"  layers[layer:1..numHiddenLayers] = if layer > 1 then SBFF(layers[layer-1].Eh, hiddenDim, hiddenDim) else SBFF(featNorm, hiddenDim, featDim) \n"
                L"  outLayer = BFF(layers[numHiddenLayers].Eh, labelDim, hiddenDim) \n"
                L"  outZ = outLayer.z + Delay(outZ, 1) \n"
                L"  CE = CrossEntropyWithSoftmax(myLabels, outZ) \n"
                L"  Err = ErrorPrediction(myLabels, outZ) \n"
                L"  logPrior = LogPrior(myLabels) \n"
                L"  ScaledLogLikelihood = outZ - logPrior \n"
                L"]\n",
                // #16: windowed RNN
                L"do = Print(val)                                                                                                           \n"
                L"val = new NDLComputationNetwork [                                                                                         \n"
                L"   hiddenDim = 512                                                                                                        \n"
                L"   numHiddenLayers = 2                                                                                                    \n"
                L"   T = 3                                  // total context window                                                         \n"
                L"                                                                                                                          \n"
                L"   // data sources                                                                                                        \n"
                L"   featDim = 40 ; labelDim = 9000                                                                                         \n"
                L"   myFeatures = Input(featDim) ; myLabels = Input(labelDim)                                                               \n"
                L"                                                                                                                          \n"
                L"   // split the augmented input vector into individual frame vectors                                                      \n"
                L"   subframes[t:0..T - 1] = RowSlice(t * featDim, featDim, myFeatures)                                                     \n"
                L"                                                                                                                          \n"
                L"   // hidden layers                                                                                                       \n"
                L"   layers[layer:1..numHiddenLayers] = [     // each layer stores a dict that stores its hidden fwd and bwd state vectors  \n"
                L"       // model parameters                                                                                                \n"
                L"       W_fwd = Parameter(hiddenDim, featDim)                                              // Parameter(outdim, indim)     \n"
                L"       W_bwd = if layer > 1 then Parameter(hiddenDim, hiddenDim) else Fail('no W_bwd')    // input-to-hidden              \n"
                L"       H_fwd = Parameter(hiddenDim, hiddenDim)                                            // hidden-to-hidden             \n"
                L"       H_bwd = Parameter(hiddenDim, hiddenDim)                                                                            \n"
                L"       b = Parameter(hiddenDim, 1)                                                        // bias                         \n"
                L"       // shared part of activations (input connections and bias)                                                         \n"
                L"       z_shared[t:0..T-1] = (if layer > 1                                                                                 \n"
                L"                             then W_fwd * layers[layer - 1].h_fwd[t] + W_bwd * layers[layer - 1].h_bwd[t]                 \n"
                L"                             else W_fwd * subframes[t]                                                                    \n"
                L"                            ) + b                                                                                         \n"
                L"       // recurrent part and non-linearity                                                                                \n"
                L"       step(H, h, dt, t) = Sigmoid(if (t + dt >= 0 && t + dt < T)                                                         \n"
                L"                                   then z_shared[t] + H * h[t + dt]                                                       \n"
                L"                                   else z_shared[t])                                                                      \n"
                L"       h_fwd[t:0..T-1] = step(H_fwd, h_fwd, -1, t)                                                                        \n"
                L"       h_bwd[t:0..T-1] = step(H_bwd, h_bwd,  1, t)                                                                        \n"
                L"   ]                                                                                                                      \n"
                L"   // output layer --linear only at this point; Softmax is applied later                                                  \n"
                L"   outLayer = [                                                                                                           \n"
                L"       // model parameters                                                                                                \n"
                L"       W_fwd = Parameter(labelDim, hiddenDim)                                                                             \n"
                L"       W_bwd = Parameter(labelDim, hiddenDim)                                                                             \n"
                L"       b = Parameter(labelDim, 1)                                                                                         \n"
                L"       //  output                                                                                                         \n"
                L"       topHiddenLayer = layers[numHiddenLayers]                                                                           \n"
                L"       centerT = Floor(T/2)                                                                                               \n"
                L"       z = W_fwd * topHiddenLayer.h_fwd[centerT] + W_bwd * topHiddenLayer.h_bwd[centerT] + b                              \n"
                L"   ]                                                                                                                      \n"
                L"   outZ = outLayer.z     // we only want this one & don't care about the rest of this dictionary                          \n"
                L"                                                                                                                          \n"
                L"   // define criterion nodes                                                                                              \n"
                L"   CE = CrossEntropyWithSoftmax(myLabels, outZ)                                                                           \n"
                L"   Err = ErrorPrediction(myLabels, outZ)                                                                                  \n"
                L"                                                                                                                          \n"
                L"   // define output node for decoding                                                                                     \n"
                L"   logPrior = LogPrior(myLabels)                                                                                          \n"
                L"   ScaledLogLikelihood = outZ - logPrior   // before: Minus(CE.BFF.FF.P,logPrior,tag=Output)                              \n"
                L"]\n",
                L" \n" // this fails because dict is outside val; expression name is not local to it
                L"do = Print(val) \n"
                L"dict = [ outY = Input(13) ] ; val = new NDLComputationNetwork [ outZ = dict.outY \n"
                L"]\n",
                L"f(x,option='default') = Print(option); do = f(42,option='value')",
                NULL};
        let first = 0; // 0 for all
        bool oneOnly = first > 0;
        for (size_t i = first; parserTests[i]; i++)
        {
            fprintf(stderr, "\n### Test %d ###\n\n", (int) i), fflush(stderr);
            let parserTest = parserTests[i];
            let expr = ParseConfigDictFromString(standardFunctions + computationNodes + commonMacros + parserTest, L"Test", vector<wstring>());

            Do(expr);
            if (oneOnly)
                break;
        }
    }
    catch (const ConfigException& err)
    {
        err.PrintError(L"error");
    }
}

}}} // namespaces

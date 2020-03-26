#pragma once


// #encoder part
// features=Input(featDim, tag="feature")
// feashift=RowSlice(RowSliceStart, baseFeatDim, features)
// GlobalMean=Parameter(baseFeatDim, 1,  init='fromFile', initFromFilePath='/hdfs/ipgsp/huhu/RNNT/GlobalMean_80.txt',    learningRateMultiplier=0) 
// GlobalInvStd=Parameter(baseFeatDim, 1, init='fromFile', initFromFilePath='/hdfs/ipgsp/huhu/RNNT/GlobalInvStd_80.txt',  learningRateMultiplier=0 ) 
// featNorm=PerDimMeanVarNormalization(feashift, GlobalMean, GlobalInvStd)    
//
// #Encoder
// numENLSTMLayers = 6
// ENlstmLayers[k:1..numENLSTMLayers] = if k == 1
//               then  LSTMPLayer_PY_LN(baseFeatDim,  LSTMCellDim, LSTMProjDim, featNorm, initWScale, initBias).o
//               else LSTMPLayer_PY_LN(LSTMProjDim,  LSTMCellDim, LSTMProjDim, ENlstmLayers[k-1], initWScale, initBias).o
// EncoderOutput = ENlstmLayers[numENLSTMLayers]
//
// EncodeNet = BS.Network.CloneFunction (inModel.features, [HLast = inModel.EncoderOutput], parameters="learnable")
// EncoderOutput = EncodeNet(features).HLast
// EncoderOutputW = DNNLinearLayer(LSTMProjDim, LSTMProjDim, EncoderOutput, initWScale, initBias).o;
// EncoderOutputWLN = LayerNorm(LSTMProjDim, EncoderOutputW, initBias, initWScale).o 
class CEncoder_3 : public IEncoder
{
public:
    CEncoder_3(const CModelParams& params)
        : GlobalMean(params.GetVectorParams(L"EncoderOutput.GlobalMean")),
          GlobalInvStd(params.GetVectorParams(L"EncoderOutput.GlobalInvStd")),
          ENlstmLayers_1(params, L"EncoderOutput.ENlstmLayers[1]"),
          ENlstmLayers_2(params, L"EncoderOutput.ENlstmLayers[2]"),
          ENlstmLayers_3(params, L"EncoderOutput.ENlstmLayers[3]"),
          ENlstmLayers_4(params, L"EncoderOutput.ENlstmLayers[4]"),
          ENlstmLayers_5(params, L"EncoderOutput.ENlstmLayers[5]"),
          ENlstmLayers_6(params, L"EncoderOutput.ENlstmLayers[6]"),
          EncoderOutputW(params, L"EncoderOutputW"),
          EncoderOutputWLN(params, L"EncoderOutputWLN"),
          EncoderOutputWLN_o(params.GetVectorParams(L"EncoderOutputWLN.gain").M)
    {
    }

    virtual void Reset()
    {
        ENlstmLayers_1.Reset();
        ENlstmLayers_2.Reset();
        ENlstmLayers_3.Reset();
        ENlstmLayers_4.Reset();
        ENlstmLayers_5.Reset();
        ENlstmLayers_6.Reset();
    }

    virtual const CVector& Forward(const float* feats, size_t baseFeatDim)
    {
        rassert_eq(GlobalMean.M, baseFeatDim);
        rassert_eq(GlobalInvStd.M, baseFeatDim);

        {
            _stack_CVector(featNorm, GlobalMean.M);
            for (size_t i = 0; i < baseFeatDim; i++)
                featNorm[i] = (feats[i] - GlobalMean[i]) * GlobalInvStd[i];

            ENlstmLayers_1.Forward(featNorm);
        }
        ENlstmLayers_2.Forward(ENlstmLayers_1.o());
        ENlstmLayers_3.Forward(ENlstmLayers_2.o());
        ENlstmLayers_4.Forward(ENlstmLayers_3.o());
        ENlstmLayers_5.Forward(ENlstmLayers_4.o());
        ENlstmLayers_6.Forward(ENlstmLayers_5.o());
        EncoderOutputW.Forward(EncoderOutputWLN_o, ENlstmLayers_6.o());
        EncoderOutputWLN.Forward(EncoderOutputWLN_o, EncoderOutputWLN_o);

        return EncoderOutputWLN_o;
    }

private:
    const CVector& GlobalMean;
    const CVector& GlobalInvStd;

    LSTMPLayer_PY_LN ENlstmLayers_1;
    LSTMPLayer_PY_LN ENlstmLayers_2;
    LSTMPLayer_PY_LN ENlstmLayers_3;
    LSTMPLayer_PY_LN ENlstmLayers_4;
    LSTMPLayer_PY_LN ENlstmLayers_5;
    LSTMPLayer_PY_LN ENlstmLayers_6;

    DNNLinearLayer EncoderOutputW;
    LayerNorm EncoderOutputWLN;
    CVector EncoderOutputWLN_o;
};


// #decoder
// lminputaxis= DynamicAxis()
// lmin=Input(ctclabledim,dynamicAxis=lminputaxis)
// numLSTMLayers = 2
// embedding = DNNLinearLayer(ctclabledim, LSTMProjDim,  lmin, initWScale, initBias).o;
// embeddingLN = LayerNorm(LSTMProjDim, embedding, initBias, initWScale).o 
// embedding_s = Sigmoid(embeddingLN)
// lstmLayers[k:1..numLSTMLayers] = if k == 1
//               then  LSTMPLayer_PY_LN(LSTMProjDim,  LSTMCellDim, LSTMProjDim, embedding_s, initWScale, initBias).o
//               else LSTMPLayer_PY_LN(LSTMProjDim,  LSTMCellDim, LSTMProjDim, lstmLayers[k-1], initWScale, initBias).o
// DecodeOutput = DNNLinearLayer(LSTMProjDim, LSTMProjDim, lstmLayers[numLSTMLayers], initWScale, initBias).o;
// DecodeOutputLN = LayerNorm(LSTMProjDim, DecodeOutput, initBias, initWScale).o 
using CPredictor_3 = CPredictor_1;


// mysum = PlusBroadcast(EncoderOutputWLN,DecodeOutputLN)
// mysum_non = ReLU(mysum)
// Wm = ParamFromRand(LSTMProjDim, ctclabledim, initWScale)
// bm = ParamFromValue(ctclabledim, 1, initBias)
// totalcr = RNNT(lmgraph, EncoderOutputWLN, DecodeOutputLN,mysum_non, Wm,bm, 4000, delayConstraint=-1, tag="criterion")
using CJoint_3 = CJoint_2;

static std::set<std::wstring> v3ParamNames = {
        L"DecodeOutput.W",
        L"DecodeOutput.b",
        L"DecodeOutputLN.bias",
        L"DecodeOutputLN.gain",
        L"EncoderOutput.ENlstmLayers[1].Wmr",
        L"EncoderOutput.ENlstmLayers[1].b",
        L"EncoderOutput.ENlstmLayers[1].ctln.bias",
        L"EncoderOutput.ENlstmLayers[1].ctln.gain",
        L"EncoderOutput.ENlstmLayers[1].wh",
        L"EncoderOutput.ENlstmLayers[1].whhln.bias",
        L"EncoderOutput.ENlstmLayers[1].whhln.gain",
        L"EncoderOutput.ENlstmLayers[1].wx",
        L"EncoderOutput.ENlstmLayers[1].wxxln.bias",
        L"EncoderOutput.ENlstmLayers[1].wxxln.gain",
        L"EncoderOutput.ENlstmLayers[2].Wmr",
        L"EncoderOutput.ENlstmLayers[2].b",
        L"EncoderOutput.ENlstmLayers[2].ctln.bias",
        L"EncoderOutput.ENlstmLayers[2].ctln.gain",
        L"EncoderOutput.ENlstmLayers[2].wh",
        L"EncoderOutput.ENlstmLayers[2].whhln.bias",
        L"EncoderOutput.ENlstmLayers[2].whhln.gain",
        L"EncoderOutput.ENlstmLayers[2].wx",
        L"EncoderOutput.ENlstmLayers[2].wxxln.bias",
        L"EncoderOutput.ENlstmLayers[2].wxxln.gain",
        L"EncoderOutput.ENlstmLayers[3].Wmr",
        L"EncoderOutput.ENlstmLayers[3].b",
        L"EncoderOutput.ENlstmLayers[3].ctln.bias",
        L"EncoderOutput.ENlstmLayers[3].ctln.gain",
        L"EncoderOutput.ENlstmLayers[3].wh",
        L"EncoderOutput.ENlstmLayers[3].whhln.bias",
        L"EncoderOutput.ENlstmLayers[3].whhln.gain",
        L"EncoderOutput.ENlstmLayers[3].wx",
        L"EncoderOutput.ENlstmLayers[3].wxxln.bias",
        L"EncoderOutput.ENlstmLayers[3].wxxln.gain",
        L"EncoderOutput.ENlstmLayers[4].Wmr",
        L"EncoderOutput.ENlstmLayers[4].b",
        L"EncoderOutput.ENlstmLayers[4].ctln.bias",
        L"EncoderOutput.ENlstmLayers[4].ctln.gain",
        L"EncoderOutput.ENlstmLayers[4].wh",
        L"EncoderOutput.ENlstmLayers[4].whhln.bias",
        L"EncoderOutput.ENlstmLayers[4].whhln.gain",
        L"EncoderOutput.ENlstmLayers[4].wx",
        L"EncoderOutput.ENlstmLayers[4].wxxln.bias",
        L"EncoderOutput.ENlstmLayers[4].wxxln.gain",
        L"EncoderOutput.ENlstmLayers[5].Wmr",
        L"EncoderOutput.ENlstmLayers[5].b",
        L"EncoderOutput.ENlstmLayers[5].ctln.bias",
        L"EncoderOutput.ENlstmLayers[5].ctln.gain",
        L"EncoderOutput.ENlstmLayers[5].wh",
        L"EncoderOutput.ENlstmLayers[5].whhln.bias",
        L"EncoderOutput.ENlstmLayers[5].whhln.gain",
        L"EncoderOutput.ENlstmLayers[5].wx",
        L"EncoderOutput.ENlstmLayers[5].wxxln.bias",
        L"EncoderOutput.ENlstmLayers[5].wxxln.gain",
        L"EncoderOutput.ENlstmLayers[6].Wmr",
        L"EncoderOutput.ENlstmLayers[6].b",
        L"EncoderOutput.ENlstmLayers[6].ctln.bias",
        L"EncoderOutput.ENlstmLayers[6].ctln.gain",
        L"EncoderOutput.ENlstmLayers[6].wh",
        L"EncoderOutput.ENlstmLayers[6].whhln.bias",
        L"EncoderOutput.ENlstmLayers[6].whhln.gain",
        L"EncoderOutput.ENlstmLayers[6].wx",
        L"EncoderOutput.ENlstmLayers[6].wxxln.bias",
        L"EncoderOutput.ENlstmLayers[6].wxxln.gain",
        L"EncoderOutput.GlobalInvStd",
        L"EncoderOutput.GlobalMean",
        L"EncoderOutput.__ENlstmLayers[1].ctln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[1].whhln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[1].wxxln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[2].ctln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[2].whhln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[2].wxxln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[3].ctln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[3].whhln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[3].wxxln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[4].ctln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[4].whhln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[4].wxxln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[5].ctln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[5].whhln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[5].wxxln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[6].ctln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[6].whhln.xHat.y",
        L"EncoderOutput.__ENlstmLayers[6].wxxln.xHat.y",
        L"EncoderOutputW.W",
        L"EncoderOutputW.b",
        L"EncoderOutputWLN.bias",
        L"EncoderOutputWLN.gain",
        L"Wm",
        L"__DecodeOutputLN.xHat.y",
        L"__EncoderOutputWLN.xHat.y",
        L"__embeddingLN.xHat.y",
        L"__lstmLayers[1].ctln.xHat.y",
        L"__lstmLayers[1].whhln.xHat.y",
        L"__lstmLayers[1].wxxln.xHat.y",
        L"__lstmLayers[2].ctln.xHat.y",
        L"__lstmLayers[2].whhln.xHat.y",
        L"__lstmLayers[2].wxxln.xHat.y",
        L"bm",
        L"embedding.W",
        L"embedding.b",
        L"embeddingLN.bias",
        L"embeddingLN.gain",
        L"lstmLayers[1].Wmr",
        L"lstmLayers[1].b",
        L"lstmLayers[1].ctln.bias",
        L"lstmLayers[1].ctln.gain",
        L"lstmLayers[1].wh",
        L"lstmLayers[1].whhln.bias",
        L"lstmLayers[1].whhln.gain",
        L"lstmLayers[1].wx",
        L"lstmLayers[1].wxxln.bias",
        L"lstmLayers[1].wxxln.gain",
        L"lstmLayers[2].Wmr",
        L"lstmLayers[2].b",
        L"lstmLayers[2].ctln.bias",
        L"lstmLayers[2].ctln.gain",
        L"lstmLayers[2].wh",
        L"lstmLayers[2].whhln.bias",
        L"lstmLayers[2].whhln.gain",
        L"lstmLayers[2].wx",
        L"lstmLayers[2].wxxln.bias",
        L"lstmLayers[2].wxxln.gain",
};    

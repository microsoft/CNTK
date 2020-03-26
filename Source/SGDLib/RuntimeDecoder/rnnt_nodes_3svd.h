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
class CEncoder_3svd : public IEncoder
{
public:
    CEncoder_3svd(const CModelParams& params)
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

    LSTMPLayer_PY_LN_svd ENlstmLayers_1;
    LSTMPLayer_PY_LN_svd ENlstmLayers_2;
    LSTMPLayer_PY_LN_svd ENlstmLayers_3;
    LSTMPLayer_PY_LN_svd ENlstmLayers_4;
    LSTMPLayer_PY_LN_svd ENlstmLayers_5;
    LSTMPLayer_PY_LN_svd ENlstmLayers_6;

    DNNLinearLayer_svd EncoderOutputW;
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
class CPredictor_3svd : public IPredictor
{
public:
    CPredictor_3svd(const CModelParams& params)
        : embedding(params, L"embedding"),
          embeddingLN(params, L"embeddingLN"),
          lstmLayers_1(params, L"lstmLayers[1]"),
          lstmLayers_2(params, L"lstmLayers[2]"),
          DecodeOutput(params, L"DecodeOutput"),
          DecodeOutputLN(params, L"DecodeOutputLN"),
          DecodeOutputLN_o(params.GetVectorParams(L"DecodeOutputLN.gain").M)
    {
        Reset();
    }

    virtual void Reset()
    {
        lstmLayers_1.Reset();
        lstmLayers_2.Reset();

        m_s = std::make_unique<State>(
                      lstmLayers_1.DetachState(),
                      lstmLayers_2.DetachState());
    }

    virtual const CVector& Forward(unsigned int label)
    {
        return Forward(*m_s, *m_s, label);
    }

    struct State : public IState
    {
        State(LSTMState&& _s1, LSTMState&& _s2)
            : s1(std::move(_s1)),
              s2(std::move(_s2))
        {
        }

        LSTMState s1;
        LSTMState s2;
    };

    virtual std::unique_ptr<IState> NewState() const
    {
        return std::make_unique<State>(
                lstmLayers_1.NewState(),
                lstmLayers_2.NewState());
    }

    virtual std::unique_ptr<IState> DetachState()
    {
        return std::move(m_s);
    }

    virtual const CVector& Forward(IState& _s, const IState& _s0, unsigned int label)
    {
        auto& s = dynamic_cast<State&>(_s);
        const auto& s0 = dynamic_cast<const State&>(_s0);

        {
            _stack_CVector(embedding_s, embedding.M());
            CVector& embeddingLN_o(embedding_s);
            embedding.Forward_OneHot(embeddingLN_o, label);
            embeddingLN.Forward(embeddingLN_o, embeddingLN_o);
            embedding_s.SetSigmoid(embeddingLN_o);

            lstmLayers_1.Forward(s.s1, s0.s1, embedding_s);
        }
        lstmLayers_2.Forward(s.s2, s0.s2, s.s1.o);

        DecodeOutput.Forward(DecodeOutputLN_o, s.s2.o);
        DecodeOutputLN.Forward(DecodeOutputLN_o, DecodeOutputLN_o);

        return DecodeOutputLN_o;
    }

private:
    DNNLinearLayer_svd embedding;
    LayerNorm embeddingLN;

    LSTMPLayer_PY_LN_svd lstmLayers_1;
    LSTMPLayer_PY_LN_svd lstmLayers_2;

    DNNLinearLayer_svd DecodeOutput;
    LayerNorm DecodeOutputLN;
    CVector DecodeOutputLN_o;

    std::unique_ptr<State> m_s;
};


// mysum = PlusBroadcast(EncoderOutputWLN,DecodeOutputLN)
// mysum_non = ReLU(mysum)
// Wm = ParamFromRand(LSTMProjDim, ctclabledim, initWScale)
// bm = ParamFromValue(ctclabledim, 1, initBias)
// totalcr = RNNT(lmgraph, EncoderOutputWLN, DecodeOutputLN,mysum_non, Wm,bm, 4000, delayConstraint=-1, tag="criterion")
class CJoint_3svd : public IJoint
{
public:
    CJoint_3svd(const CModelParams& params)
        // Hack: Wm_U and Wm_V are transposed :(
        : Wm_U(params.GetMatrixParams(L"Wm_V")),
          Wm_V(params.GetMatrixParams(L"Wm_U")),
          bm(params.GetVectorParams(L"bm")),

          mysum(Wm_V->N),
          mysum_non(mysum),
          sum_linear(Wm_U->M),
          softmax(sum_linear.M)
    {
    }

    virtual const CVector& Forward(const CVector& EncodeOutputWLN, const CVector& DecodeOutputWLN)
    {
        mysum.SetPlus(EncodeOutputWLN, DecodeOutputWLN);
        mysum_non.SetReLU(mysum);
        {
            _stack_CVector(sum_linear1, Wm_U->N);
            sum_linear1.SetTimes(Wm_V, mysum_non);
            sum_linear.SetTimes(Wm_U, sum_linear1);
        }
        sum_linear.SetPlus(sum_linear, bm);

        return sum_linear;
    }

    virtual const CVector& Forward_SoftMax(const CVector& EncodeOutputWLN, const CVector& DecodeOutputWLN)
    {
        const auto& x = Forward(EncodeOutputWLN, DecodeOutputWLN);
        softmax.SetSoftMax(x);

        return softmax;
    }

private:
    const IMatrix* Wm_U;
    const IMatrix* Wm_V;
    const CVector& bm;

    CVector mysum;
    CVector& mysum_non;
    CVector sum_linear;
    CVector softmax;
};

static std::set<std::wstring> v3svdParamNames = {
        L"DecodeOutput.W_U",
        L"DecodeOutput.W_V",
        L"DecodeOutput.b",
        L"DecodeOutputLN.bias",
        L"DecodeOutputLN.gain",
        L"EncoderOutput.ENlstmLayers[1].Wmr_U",
        L"EncoderOutput.ENlstmLayers[1].Wmr_V",
        L"EncoderOutput.ENlstmLayers[1].b",
        L"EncoderOutput.ENlstmLayers[1].ctln.bias",
        L"EncoderOutput.ENlstmLayers[1].ctln.gain",
        L"EncoderOutput.ENlstmLayers[1].wh_U",
        L"EncoderOutput.ENlstmLayers[1].wh_V",
        L"EncoderOutput.ENlstmLayers[1].whhln.bias",
        L"EncoderOutput.ENlstmLayers[1].whhln.gain",
        L"EncoderOutput.ENlstmLayers[1].wx_U",
        L"EncoderOutput.ENlstmLayers[1].wx_V",
        L"EncoderOutput.ENlstmLayers[1].wxxln.bias",
        L"EncoderOutput.ENlstmLayers[1].wxxln.gain",
        L"EncoderOutput.ENlstmLayers[2].Wmr_U",
        L"EncoderOutput.ENlstmLayers[2].Wmr_V",
        L"EncoderOutput.ENlstmLayers[2].b",
        L"EncoderOutput.ENlstmLayers[2].ctln.bias",
        L"EncoderOutput.ENlstmLayers[2].ctln.gain",
        L"EncoderOutput.ENlstmLayers[2].wh_U",
        L"EncoderOutput.ENlstmLayers[2].wh_V",
        L"EncoderOutput.ENlstmLayers[2].whhln.bias",
        L"EncoderOutput.ENlstmLayers[2].whhln.gain",
        L"EncoderOutput.ENlstmLayers[2].wx_U",
        L"EncoderOutput.ENlstmLayers[2].wx_V",
        L"EncoderOutput.ENlstmLayers[2].wxxln.bias",
        L"EncoderOutput.ENlstmLayers[2].wxxln.gain",
        L"EncoderOutput.ENlstmLayers[3].Wmr_U",
        L"EncoderOutput.ENlstmLayers[3].Wmr_V",
        L"EncoderOutput.ENlstmLayers[3].b",
        L"EncoderOutput.ENlstmLayers[3].ctln.bias",
        L"EncoderOutput.ENlstmLayers[3].ctln.gain",
        L"EncoderOutput.ENlstmLayers[3].wh_U",
        L"EncoderOutput.ENlstmLayers[3].wh_V",
        L"EncoderOutput.ENlstmLayers[3].whhln.bias",
        L"EncoderOutput.ENlstmLayers[3].whhln.gain",
        L"EncoderOutput.ENlstmLayers[3].wx_U",
        L"EncoderOutput.ENlstmLayers[3].wx_V",
        L"EncoderOutput.ENlstmLayers[3].wxxln.bias",
        L"EncoderOutput.ENlstmLayers[3].wxxln.gain",
        L"EncoderOutput.ENlstmLayers[4].Wmr_U",
        L"EncoderOutput.ENlstmLayers[4].Wmr_V",
        L"EncoderOutput.ENlstmLayers[4].b",
        L"EncoderOutput.ENlstmLayers[4].ctln.bias",
        L"EncoderOutput.ENlstmLayers[4].ctln.gain",
        L"EncoderOutput.ENlstmLayers[4].wh_U",
        L"EncoderOutput.ENlstmLayers[4].wh_V",
        L"EncoderOutput.ENlstmLayers[4].whhln.bias",
        L"EncoderOutput.ENlstmLayers[4].whhln.gain",
        L"EncoderOutput.ENlstmLayers[4].wx_U",
        L"EncoderOutput.ENlstmLayers[4].wx_V",
        L"EncoderOutput.ENlstmLayers[4].wxxln.bias",
        L"EncoderOutput.ENlstmLayers[4].wxxln.gain",
        L"EncoderOutput.ENlstmLayers[5].Wmr_U",
        L"EncoderOutput.ENlstmLayers[5].Wmr_V",
        L"EncoderOutput.ENlstmLayers[5].b",
        L"EncoderOutput.ENlstmLayers[5].ctln.bias",
        L"EncoderOutput.ENlstmLayers[5].ctln.gain",
        L"EncoderOutput.ENlstmLayers[5].wh_U",
        L"EncoderOutput.ENlstmLayers[5].wh_V",
        L"EncoderOutput.ENlstmLayers[5].whhln.bias",
        L"EncoderOutput.ENlstmLayers[5].whhln.gain",
        L"EncoderOutput.ENlstmLayers[5].wx_U",
        L"EncoderOutput.ENlstmLayers[5].wx_V",
        L"EncoderOutput.ENlstmLayers[5].wxxln.bias",
        L"EncoderOutput.ENlstmLayers[5].wxxln.gain",
        L"EncoderOutput.ENlstmLayers[6].Wmr_U",
        L"EncoderOutput.ENlstmLayers[6].Wmr_V",
        L"EncoderOutput.ENlstmLayers[6].b",
        L"EncoderOutput.ENlstmLayers[6].ctln.bias",
        L"EncoderOutput.ENlstmLayers[6].ctln.gain",
        L"EncoderOutput.ENlstmLayers[6].wh_U",
        L"EncoderOutput.ENlstmLayers[6].wh_V",
        L"EncoderOutput.ENlstmLayers[6].whhln.bias",
        L"EncoderOutput.ENlstmLayers[6].whhln.gain",
        L"EncoderOutput.ENlstmLayers[6].wx_U",
        L"EncoderOutput.ENlstmLayers[6].wx_V",
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
        L"EncoderOutputW.W_U",
        L"EncoderOutputW.W_V",
        L"EncoderOutputW.b",
        L"EncoderOutputWLN.bias",
        L"EncoderOutputWLN.gain",
        L"Wm_U",
        L"Wm_V",
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
        L"embedding.W_U",
        L"embedding.W_V",
        L"embedding.b",
        L"embeddingLN.bias",
        L"embeddingLN.gain",
        L"lstmLayers[1].Wmr_U",
        L"lstmLayers[1].Wmr_V",
        L"lstmLayers[1].b",
        L"lstmLayers[1].ctln.bias",
        L"lstmLayers[1].ctln.gain",
        L"lstmLayers[1].wh_U",
        L"lstmLayers[1].wh_V",
        L"lstmLayers[1].whhln.bias",
        L"lstmLayers[1].whhln.gain",
        L"lstmLayers[1].wx_U",
        L"lstmLayers[1].wx_V",
        L"lstmLayers[1].wxxln.bias",
        L"lstmLayers[1].wxxln.gain",
        L"lstmLayers[2].Wmr_U",
        L"lstmLayers[2].Wmr_V",
        L"lstmLayers[2].b",
        L"lstmLayers[2].ctln.bias",
        L"lstmLayers[2].ctln.gain",
        L"lstmLayers[2].wh_U",
        L"lstmLayers[2].wh_V",
        L"lstmLayers[2].whhln.bias",
        L"lstmLayers[2].whhln.gain",
        L"lstmLayers[2].wx_U",
        L"lstmLayers[2].wx_V",
        L"lstmLayers[2].wxxln.bias",
        L"lstmLayers[2].wxxln.gain",
};

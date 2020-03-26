#pragma once


// #feature normalization
// features=Input(featDim, tag="feature")
// feashift=RowSlice(RowSliceStart, baseFeatDim, features)
// GlobalMean=Parameter(baseFeatDim, 1,  init='fromFile', initFromFilePath='/hdfs/ipgsp/ruizhao/RNNT/mixunit/GlobalMean_80.txt',    learningRateMultiplier=0) 
// GlobalInvStd=Parameter(baseFeatDim, 1, init='fromFile', initFromFilePath='/hdfs/ipgsp/ruizhao/RNNT/mixunit/GlobalInvStd_80.txt',  learningRateMultiplier=0 ) 
// featNorm=PerDimMeanVarNormalization(feashift, GlobalMean, GlobalInvStd)    
// 
// #Encoder
// numENLSTMLayers = 6
// ENlstmLayers[k:1..numENLSTMLayers] = if k == 1
//               then  LSTMPLayer_PY_LN(baseFeatDim,  LSTMCellDim, LSTMProjDim, featNorm, initWScale, initBias).o
//               else LSTMPLayer_PY_LN(LSTMProjDim,  LSTMCellDim, LSTMProjDim, ENlstmLayers[k-1], initWScale, initBias).o
// EncodeOutput = DNNLinearLayer(LSTMProjDim, LSTMProjDim, ENlstmLayers[numENLSTMLayers], initWScale, initBias).o;
// EncodeOutputLN = LayerNorm(LSTMProjDim, EncodeOutput, initBias, initWScale).o
class CEncoder_1 : public IEncoder
{
public:
    CEncoder_1(const CModelParams& params)
        : GlobalMean(params.GetVectorParams(L"GlobalMean")),
          GlobalInvStd(params.GetVectorParams(L"GlobalInvStd")),
          ENlstmLayers_1(params, L"ENlstmLayers[1]"),
          ENlstmLayers_2(params, L"ENlstmLayers[2]"),
          ENlstmLayers_3(params, L"ENlstmLayers[3]"),
          ENlstmLayers_4(params, L"ENlstmLayers[4]"),
          ENlstmLayers_5(params, L"ENlstmLayers[5]"),
          ENlstmLayers_6(params, L"ENlstmLayers[6]"),
          EncodeOutput(params, L"EncodeOutput"),
          EncodeOutputLN(params, L"EncodeOutputLN"),
          EncodeOutputLN_o(params.GetVectorParams(L"EncodeOutputLN.gain").M)
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
        EncodeOutput.Forward(EncodeOutputLN_o, ENlstmLayers_6.o());
        EncodeOutputLN.Forward(EncodeOutputLN_o, EncodeOutputLN_o);

        return EncodeOutputLN_o;
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

    DNNLinearLayer EncodeOutput;
    LayerNorm EncodeOutputLN;
    CVector EncodeOutputLN_o;
};


// #prediction
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
class CPredictor_1 : public IPredictor
{
public:
    CPredictor_1(const CModelParams& params)
        : embedding(params, L"embedding"),
          embeddingLN(params, L"embeddingLN"),
          lstmLayers_1(params, L"lstmLayers[1]"),
          lstmLayers_2(params, L"lstmLayers[2]"),
          DecodeOutput(params, L"DecodeOutput"),
          DecodeOutputLN(params, L"DecodeOutputLN"),
          DecodeOutputLN_o(params.GetVectorParams(L"DecodeOutputLN.gain").M),
          m_s(std::make_unique<State>(
                      lstmLayers_1.DetachState(),
                      lstmLayers_2.DetachState()))
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
    DNNLinearLayer embedding;
    LayerNorm embeddingLN;

    LSTMPLayer_PY_LN lstmLayers_1;
    LSTMPLayer_PY_LN lstmLayers_2;

    DNNLinearLayer DecodeOutput;
    LayerNorm DecodeOutputLN;
    CVector DecodeOutputLN_o;

    std::unique_ptr<State> m_s;
};


// mysum = PlusBroadcast(EncodeOutputLN,DecodeOutputLN)
// mysum_non = ReLU(mysum)
// 
// #linear
// sum_linear = DNNLinearLayer(LSTMProjDim,ctclabledim, mysum_non, initWScale, initBias).o;
class CJoint_1 : public IJoint
{
public:
    CJoint_1(const CModelParams& params)
        : mysum(params.GetMatrixParams(L"sum_linear.W")->N),
          mysum_non(mysum),
          sum_linear(params, L"sum_linear"),
          sum_linear_o(params.GetMatrixParams(L"sum_linear.W")->M),
          softmax(sum_linear_o.M)
    {
    }

    virtual const CVector& Forward(const CVector& EncodeOutputLN, const CVector& DecodeOutputLN)
    {
        mysum.SetPlus(EncodeOutputLN, DecodeOutputLN);
        mysum_non.SetReLU(mysum);
        sum_linear.Forward(sum_linear_o, mysum_non);

        return sum_linear_o;
    }

    virtual const CVector& Forward_SoftMax(const CVector& EncodeOutputLN, const CVector& DecodeOutputLN)
    {
        const auto& x = Forward(EncodeOutputLN, DecodeOutputLN);
        softmax.SetSoftMax(x);

        return softmax;
    }

private:
    CVector mysum;
    CVector& mysum_non;
    DNNLinearLayer sum_linear;
    CVector sum_linear_o;
    CVector softmax;
};

static std::set<std::wstring> v1ParamNames = {
    L"DecodeOutput.W",
    L"DecodeOutput.b",
    L"DecodeOutputLN.bias",
    L"DecodeOutputLN.gain",
    L"ENlstmLayers[1].Wmr",
    L"ENlstmLayers[1].b",
    L"ENlstmLayers[1].ctln.bias",
    L"ENlstmLayers[1].ctln.gain",
    L"ENlstmLayers[1].wh",
    L"ENlstmLayers[1].whhln.bias",
    L"ENlstmLayers[1].whhln.gain",
    L"ENlstmLayers[1].wx",
    L"ENlstmLayers[1].wxxln.bias",
    L"ENlstmLayers[1].wxxln.gain",
    L"ENlstmLayers[2].Wmr",
    L"ENlstmLayers[2].b",
    L"ENlstmLayers[2].ctln.bias",
    L"ENlstmLayers[2].ctln.gain",
    L"ENlstmLayers[2].wh",
    L"ENlstmLayers[2].whhln.bias",
    L"ENlstmLayers[2].whhln.gain",
    L"ENlstmLayers[2].wx",
    L"ENlstmLayers[2].wxxln.bias",
    L"ENlstmLayers[2].wxxln.gain",
    L"ENlstmLayers[3].Wmr",
    L"ENlstmLayers[3].b",
    L"ENlstmLayers[3].ctln.bias",
    L"ENlstmLayers[3].ctln.gain",
    L"ENlstmLayers[3].wh",
    L"ENlstmLayers[3].whhln.bias",
    L"ENlstmLayers[3].whhln.gain",
    L"ENlstmLayers[3].wx",
    L"ENlstmLayers[3].wxxln.bias",
    L"ENlstmLayers[3].wxxln.gain",
    L"ENlstmLayers[4].Wmr",
    L"ENlstmLayers[4].b",
    L"ENlstmLayers[4].ctln.bias",
    L"ENlstmLayers[4].ctln.gain",
    L"ENlstmLayers[4].wh",
    L"ENlstmLayers[4].whhln.bias",
    L"ENlstmLayers[4].whhln.gain",
    L"ENlstmLayers[4].wx",
    L"ENlstmLayers[4].wxxln.bias",
    L"ENlstmLayers[4].wxxln.gain",
    L"ENlstmLayers[5].Wmr",
    L"ENlstmLayers[5].b",
    L"ENlstmLayers[5].ctln.bias",
    L"ENlstmLayers[5].ctln.gain",
    L"ENlstmLayers[5].wh",
    L"ENlstmLayers[5].whhln.bias",
    L"ENlstmLayers[5].whhln.gain",
    L"ENlstmLayers[5].wx",
    L"ENlstmLayers[5].wxxln.bias",
    L"ENlstmLayers[5].wxxln.gain",
    L"ENlstmLayers[6].Wmr",
    L"ENlstmLayers[6].b",
    L"ENlstmLayers[6].ctln.bias",
    L"ENlstmLayers[6].ctln.gain",
    L"ENlstmLayers[6].wh",
    L"ENlstmLayers[6].whhln.bias",
    L"ENlstmLayers[6].whhln.gain",
    L"ENlstmLayers[6].wx",
    L"ENlstmLayers[6].wxxln.bias",
    L"ENlstmLayers[6].wxxln.gain",
    L"EncodeOutput.W",
    L"EncodeOutput.b",
    L"EncodeOutputLN.bias",
    L"EncodeOutputLN.gain",
    L"GlobalInvStd",
    L"GlobalMean",
    L"__DecodeOutputLN.xHat.y",
    L"__ENlstmLayers[1].ctln.xHat.y",
    L"__ENlstmLayers[1].whhln.xHat.y",
    L"__ENlstmLayers[1].wxxln.xHat.y",
    L"__ENlstmLayers[2].ctln.xHat.y",
    L"__ENlstmLayers[2].whhln.xHat.y",
    L"__ENlstmLayers[2].wxxln.xHat.y",
    L"__ENlstmLayers[3].ctln.xHat.y",
    L"__ENlstmLayers[3].whhln.xHat.y",
    L"__ENlstmLayers[3].wxxln.xHat.y",
    L"__ENlstmLayers[4].ctln.xHat.y",
    L"__ENlstmLayers[4].whhln.xHat.y",
    L"__ENlstmLayers[4].wxxln.xHat.y",
    L"__ENlstmLayers[5].ctln.xHat.y",
    L"__ENlstmLayers[5].whhln.xHat.y",
    L"__ENlstmLayers[5].wxxln.xHat.y",
    L"__ENlstmLayers[6].ctln.xHat.y",
    L"__ENlstmLayers[6].whhln.xHat.y",
    L"__ENlstmLayers[6].wxxln.xHat.y",
    L"__EncodeOutputLN.xHat.y",
    L"__embeddingLN.xHat.y",
    L"__lstmLayers[1].ctln.xHat.y",
    L"__lstmLayers[1].whhln.xHat.y",
    L"__lstmLayers[1].wxxln.xHat.y",
    L"__lstmLayers[2].ctln.xHat.y",
    L"__lstmLayers[2].whhln.xHat.y",
    L"__lstmLayers[2].wxxln.xHat.y",
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
    L"sum_linear.W",
    L"sum_linear.b",
};

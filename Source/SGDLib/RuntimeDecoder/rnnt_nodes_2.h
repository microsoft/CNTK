#pragma once


// features=Input(featDim, tag=feature)
// labels=Input(labelDim, tag=label)
// GlobalMean=Parameter(baseFeatDim,   init=fromFile, initFromFilePath=$GMean$,    computeGradient=false) 
// GlobalInvStd=Parameter(baseFeatDim, init=fromFile, initFromFilePath=$GInvStd$,  computeGradient=false ) 
// feashift=RowSlice(RowSliceStart, baseFeatDim, features)
// featNorm=PerDimMeanVarNormalization(feashift, GlobalMean, GlobalInvStd)    
//
// # layer 1
// LSTMoutput1 = LSTMPComponent(input_hidden_Dim1, hiddenDim, cellDim, cellDimT2, cellDimT3, cellDimT4, featNorm);
// # layer 2 
// LSTMoutput2 = LSTMPComponent(input_hidden_Dim2, hiddenDim, cellDim, cellDimT2, cellDimT3, cellDimT4, LSTMoutput1);
// # layer 3 
// LSTMoutput3 = LSTMPComponent(input_hidden_Dim2, hiddenDim, cellDim, cellDimT2, cellDimT3, cellDimT4,  LSTMoutput2);
// # layer 4
// LSTMoutput4 = LSTMPComponent(input_hidden_Dim2, hiddenDim, cellDim, cellDimT2, cellDimT3, cellDimT4,  LSTMoutput3);
// # layer 5
// LSTMoutput5 = LSTMPComponent(input_hidden_Dim2, hiddenDim, cellDim, cellDimT2, cellDimT3, cellDimT4, LSTMoutput4);
// # layer 6
// LSTMoutput6 = LSTMPComponent(input_hidden_Dim2, hiddenDim, cellDim, cellDimT2, cellDimT3, cellDimT4, LSTMoutput5);
//
// EncodeNet = BS.Network.CloneFunction (inCTCModel.features, [ HLast =inCTCModel.LSTMoutput6], parameters="learnable")
// EncodeOutput = EncodeNet(features).HLast
// EncodeOutputW = DNNLinearLayer(LSTMProjDim, LSTMProjDim, EncodeOutput, initWScale, initBias).o;
// EncodeOutputWLN = LayerNorm(LSTMProjDim, EncodeOutputW, initBias, initWScale).o 
class CEncoder_2 : public IEncoder
{
public:
    CEncoder_2(const CModelParams& params)
        : GlobalMean(params.GetVectorParams(L"EncodeOutput.GlobalMean")),
          GlobalInvStd(params.GetVectorParams(L"EncodeOutput.GlobalInvStd")),
          ENlstmLayers_1(params, L"EncodeOutput.LSTMoutput1"),
          ENlstmLayers_2(params, L"EncodeOutput.LSTMoutput2"),
          ENlstmLayers_3(params, L"EncodeOutput.LSTMoutput3"),
          ENlstmLayers_4(params, L"EncodeOutput.LSTMoutput4"),
          ENlstmLayers_5(params, L"EncodeOutput.LSTMoutput5"),
          ENlstmLayers_6(params, L"EncodeOutput.LSTMoutput6"),
          EncodeOutputW(params, L"EncodeOutputW"),
          EncodeOutputWLN(params, L"EncodeOutputWLN"),
          EncodeOutputWLN_o(params.GetVectorParams(L"EncodeOutputWLN.gain").M)
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
        EncodeOutputW.Forward(EncodeOutputWLN_o, ENlstmLayers_6.o());
        EncodeOutputWLN.Forward(EncodeOutputWLN_o, EncodeOutputWLN_o);

        return EncodeOutputWLN_o;
    }

private:
    const CVector& GlobalMean;
    const CVector& GlobalInvStd;

    LSTMPComponent ENlstmLayers_1;
    LSTMPComponent ENlstmLayers_2;
    LSTMPComponent ENlstmLayers_3;
    LSTMPComponent ENlstmLayers_4;
    LSTMPComponent ENlstmLayers_5;
    LSTMPComponent ENlstmLayers_6;

    DNNLinearLayer EncodeOutputW;
    LayerNorm EncodeOutputWLN;
    CVector EncodeOutputWLN_o;
};



// #LM part
// lminputaxis= DynamicAxis()
// lmin=Input(ctclabledim,dynamicAxis=lminputaxis)
// lmout=Input(ctclabledim,dynamicAxis=lminputaxis)
// numLSTMLayers = 2
// embedding = DNNLinearLayer(ctclabledim, LSTMProjDim,  lmin, initWScale, initBias).o;
// embedding_s = Sigmoid(embedding)
// lstmLayers[k:1..numLSTMLayers] = if k == 1
//               then  LSTMPLayer_PY(LSTMProjDim,  LSTMCellDim, LSTMProjDim, embedding_s, initWScale, initBias).o
//               else LSTMPLayer_PY(LSTMProjDim,  LSTMCellDim, LSTMProjDim, lstmLayers[k-1], initWScale, initBias).o
//
// DecodeNet = BS.Network.CloneFunction (inLMModel.lmin, [ HLast =inLMModel.DELSTMOutput], parameters="learnable")
// DecodeOutput = DecodeNet(lmin).HLast
// DecodeOutputW = DNNLinearLayer(LSTMProjDim, LSTMProjDim, DecodeOutput, initWScale, initBias).o;
// DecodeOutputWLN = LayerNorm(LSTMProjDim, DecodeOutputW, initBias, initWScale).o 
class CPredictor_2 : public IPredictor
{
public:
    CPredictor_2(const CModelParams& params)
        : embedding(params, L"DecodeOutput.embedding"),
          lstmLayers_1(params, L"DecodeOutput.lstmLayers[1]"),
          lstmLayers_2(params, L"DecodeOutput.lstmLayers[2]"),
          DecodeOutputW(params, L"DecodeOutputW"),
          DecodeOutputWLN(params, L"DecodeOutputWLN"),
          DecodeOutputWLN_o(params.GetVectorParams(L"DecodeOutputWLN.gain").M)
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
            embedding.Forward_OneHot(embedding_s, label);
            embedding_s.SetSigmoid(embedding_s);

            lstmLayers_1.Forward(s.s1, s0.s1, embedding_s);
        }
        lstmLayers_2.Forward(s.s2, s0.s2, s.s1.o);

        DecodeOutputW.Forward(DecodeOutputWLN_o, s.s2.o);
        DecodeOutputWLN.Forward(DecodeOutputWLN_o, DecodeOutputWLN_o);

        return DecodeOutputWLN_o;
    }

private:
    DNNLinearLayer embedding;

    LSTMPLayer_PY lstmLayers_1;
    LSTMPLayer_PY lstmLayers_2;

    DNNLinearLayer DecodeOutputW;
    LayerNorm DecodeOutputWLN;
    CVector DecodeOutputWLN_o;

    std::unique_ptr<State> m_s;
};


// mysum = PlusBroadcast(EncodeOutputWLN,DecodeOutputWLN)
// mysum_non = ReLU(mysum)
// Wm = ParamFromRand(LSTMProjDim, ctclabledim, initWScale)
// bm = ParamFromValue(ctclabledim, 1, initBias)
// RNNTcr = RNNT(lmgraph,EncodeOutputW,DecodeOutputW,mysum_non, Wm,bm, 3996,delayConstraint=-1, tag="criterion")
class CJoint_2 : public IJoint
{
public:
    CJoint_2(const CModelParams& params)
        : Wm(params.GetMatrixParams(L"Wm")),
          bm(params.GetVectorParams(L"bm")),

          mysum(Wm->N),
          mysum_non(mysum),
          sum_linear(Wm->M),
          softmax(sum_linear.M)
    {
    }

    virtual const CVector& Forward(const CVector& EncodeOutputWLN, const CVector& DecodeOutputWLN)
    {
        mysum.SetPlus(EncodeOutputWLN, DecodeOutputWLN);
        mysum_non.SetReLU(mysum);
        sum_linear.SetTimes(Wm, mysum_non);
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
    const IMatrix* Wm;
    const CVector& bm;

    CVector mysum;
    CVector& mysum_non;
    CVector sum_linear;
    CVector softmax;
};


static std::set<std::wstring> v2ParamNames = {
    L"DecodeOutput.embedding.W",
    L"DecodeOutput.embedding.b",
    L"DecodeOutput.lstmLayers[1].Wmr",
    L"DecodeOutput.lstmLayers[1].b",
    L"DecodeOutput.lstmLayers[1].wh",
    L"DecodeOutput.lstmLayers[1].wx",
    L"DecodeOutput.lstmLayers[2].Wmr",
    L"DecodeOutput.lstmLayers[2].b",
    L"DecodeOutput.lstmLayers[2].wh",
    L"DecodeOutput.lstmLayers[2].wx",
    L"DecodeOutputW.W",
    L"DecodeOutputW.b",
    L"DecodeOutputWLN.bias",
    L"DecodeOutputWLN.gain",
    L"EncodeOutput.GlobalInvStd",
    L"EncodeOutput.GlobalMean",
    L"EncodeOutput.LSTMoutput1.W_ifgo",
    L"EncodeOutput.LSTMoutput1.W_r_m",
    L"EncodeOutput.LSTMoutput1.bc",
    L"EncodeOutput.LSTMoutput1.bf",
    L"EncodeOutput.LSTMoutput1.bi",
    L"EncodeOutput.LSTMoutput1.bo",
    L"EncodeOutput.LSTMoutput1.phole_f_c",
    L"EncodeOutput.LSTMoutput1.phole_i_c",
    L"EncodeOutput.LSTMoutput1.phole_o_c",
    L"EncodeOutput.LSTMoutput2.W_ifgo",
    L"EncodeOutput.LSTMoutput2.W_r_m",
    L"EncodeOutput.LSTMoutput2.bc",
    L"EncodeOutput.LSTMoutput2.bf",
    L"EncodeOutput.LSTMoutput2.bi",
    L"EncodeOutput.LSTMoutput2.bo",
    L"EncodeOutput.LSTMoutput2.phole_f_c",
    L"EncodeOutput.LSTMoutput2.phole_i_c",
    L"EncodeOutput.LSTMoutput2.phole_o_c",
    L"EncodeOutput.LSTMoutput3.W_ifgo",
    L"EncodeOutput.LSTMoutput3.W_r_m",
    L"EncodeOutput.LSTMoutput3.bc",
    L"EncodeOutput.LSTMoutput3.bf",
    L"EncodeOutput.LSTMoutput3.bi",
    L"EncodeOutput.LSTMoutput3.bo",
    L"EncodeOutput.LSTMoutput3.phole_f_c",
    L"EncodeOutput.LSTMoutput3.phole_i_c",
    L"EncodeOutput.LSTMoutput3.phole_o_c",
    L"EncodeOutput.LSTMoutput4.W_ifgo",
    L"EncodeOutput.LSTMoutput4.W_r_m",
    L"EncodeOutput.LSTMoutput4.bc",
    L"EncodeOutput.LSTMoutput4.bf",
    L"EncodeOutput.LSTMoutput4.bi",
    L"EncodeOutput.LSTMoutput4.bo",
    L"EncodeOutput.LSTMoutput4.phole_f_c",
    L"EncodeOutput.LSTMoutput4.phole_i_c",
    L"EncodeOutput.LSTMoutput4.phole_o_c",
    L"EncodeOutput.LSTMoutput5.W_ifgo",
    L"EncodeOutput.LSTMoutput5.W_r_m",
    L"EncodeOutput.LSTMoutput5.bc",
    L"EncodeOutput.LSTMoutput5.bf",
    L"EncodeOutput.LSTMoutput5.bi",
    L"EncodeOutput.LSTMoutput5.bo",
    L"EncodeOutput.LSTMoutput5.phole_f_c",
    L"EncodeOutput.LSTMoutput5.phole_i_c",
    L"EncodeOutput.LSTMoutput5.phole_o_c",
    L"EncodeOutput.LSTMoutput6.W_ifgo",
    L"EncodeOutput.LSTMoutput6.W_r_m",
    L"EncodeOutput.LSTMoutput6.bc",
    L"EncodeOutput.LSTMoutput6.bf",
    L"EncodeOutput.LSTMoutput6.bi",
    L"EncodeOutput.LSTMoutput6.bo",
    L"EncodeOutput.LSTMoutput6.phole_f_c",
    L"EncodeOutput.LSTMoutput6.phole_i_c",
    L"EncodeOutput.LSTMoutput6.phole_o_c",
    L"EncodeOutputW.W",
    L"EncodeOutputW.b",
    L"EncodeOutputWLN.bias",
    L"EncodeOutputWLN.gain",
    L"Wm",
    L"__DecodeOutputWLN.xHat.y",
    L"__EncodeOutputWLN.xHat.y",
    L"bm",
    L"DecodeOutput.embedding.W",
    L"DecodeOutput.embedding.b",
    L"DecodeOutput.lstmLayers[1].Wmr",
    L"DecodeOutput.lstmLayers[1].b",
    L"DecodeOutput.lstmLayers[1].wh",
    L"DecodeOutput.lstmLayers[1].wx",
    L"DecodeOutput.lstmLayers[2].Wmr",
    L"DecodeOutput.lstmLayers[2].b",
    L"DecodeOutput.lstmLayers[2].wh",
    L"DecodeOutput.lstmLayers[2].wx",
    L"DecodeOutputW.W",
    L"DecodeOutputW.b",
    L"DecodeOutputWLN.bias",
    L"DecodeOutputWLN.gain",
    L"EncodeOutput.GlobalInvStd",
    L"EncodeOutput.GlobalMean",
    L"EncodeOutput.LSTMoutput1.W_ifgo",
    L"EncodeOutput.LSTMoutput1.W_r_m",
    L"EncodeOutput.LSTMoutput1.bc",
    L"EncodeOutput.LSTMoutput1.bf",
    L"EncodeOutput.LSTMoutput1.bi",
    L"EncodeOutput.LSTMoutput1.bo",
    L"EncodeOutput.LSTMoutput1.phole_f_c",
    L"EncodeOutput.LSTMoutput1.phole_i_c",
    L"EncodeOutput.LSTMoutput1.phole_o_c",
    L"EncodeOutput.LSTMoutput2.W_ifgo",
    L"EncodeOutput.LSTMoutput2.W_r_m",
    L"EncodeOutput.LSTMoutput2.bc",
    L"EncodeOutput.LSTMoutput2.bf",
    L"EncodeOutput.LSTMoutput2.bi",
    L"EncodeOutput.LSTMoutput2.bo",
    L"EncodeOutput.LSTMoutput2.phole_f_c",
    L"EncodeOutput.LSTMoutput2.phole_i_c",
    L"EncodeOutput.LSTMoutput2.phole_o_c",
    L"EncodeOutput.LSTMoutput3.W_ifgo",
    L"EncodeOutput.LSTMoutput3.W_r_m",
    L"EncodeOutput.LSTMoutput3.bc",
    L"EncodeOutput.LSTMoutput3.bf",
    L"EncodeOutput.LSTMoutput3.bi",
    L"EncodeOutput.LSTMoutput3.bo",
    L"EncodeOutput.LSTMoutput3.phole_f_c",
    L"EncodeOutput.LSTMoutput3.phole_i_c",
    L"EncodeOutput.LSTMoutput3.phole_o_c",
    L"EncodeOutput.LSTMoutput4.W_ifgo",
    L"EncodeOutput.LSTMoutput4.W_r_m",
    L"EncodeOutput.LSTMoutput4.bc",
    L"EncodeOutput.LSTMoutput4.bf",
    L"EncodeOutput.LSTMoutput4.bi",
    L"EncodeOutput.LSTMoutput4.bo",
    L"EncodeOutput.LSTMoutput4.phole_f_c",
    L"EncodeOutput.LSTMoutput4.phole_i_c",
    L"EncodeOutput.LSTMoutput4.phole_o_c",
    L"EncodeOutput.LSTMoutput5.W_ifgo",
    L"EncodeOutput.LSTMoutput5.W_r_m",
    L"EncodeOutput.LSTMoutput5.bc",
    L"EncodeOutput.LSTMoutput5.bf",
    L"EncodeOutput.LSTMoutput5.bi",
    L"EncodeOutput.LSTMoutput5.bo",
    L"EncodeOutput.LSTMoutput5.phole_f_c",
    L"EncodeOutput.LSTMoutput5.phole_i_c",
    L"EncodeOutput.LSTMoutput5.phole_o_c",
    L"EncodeOutput.LSTMoutput6.W_ifgo",
    L"EncodeOutput.LSTMoutput6.W_r_m",
    L"EncodeOutput.LSTMoutput6.bc",
    L"EncodeOutput.LSTMoutput6.bf",
    L"EncodeOutput.LSTMoutput6.bi",
    L"EncodeOutput.LSTMoutput6.bo",
    L"EncodeOutput.LSTMoutput6.phole_f_c",
    L"EncodeOutput.LSTMoutput6.phole_i_c",
    L"EncodeOutput.LSTMoutput6.phole_o_c",
    L"EncodeOutputW.W",
    L"EncodeOutputW.b",
    L"EncodeOutputWLN.bias",
    L"EncodeOutputWLN.gain",
    L"Wm",
    L"__DecodeOutputWLN.xHat.y",
    L"__EncodeOutputWLN.xHat.y",
    L"bm",
    L"DecodeOutput.embedding.W",
    L"DecodeOutput.embedding.b",
    L"DecodeOutput.lstmLayers[1].Wmr",
    L"DecodeOutput.lstmLayers[1].b",
    L"DecodeOutput.lstmLayers[1].wh",
    L"DecodeOutput.lstmLayers[1].wx",
    L"DecodeOutput.lstmLayers[2].Wmr",
    L"DecodeOutput.lstmLayers[2].b",
    L"DecodeOutput.lstmLayers[2].wh",
    L"DecodeOutput.lstmLayers[2].wx",
    L"DecodeOutputW.W",
    L"DecodeOutputW.b",
    L"DecodeOutputWLN.bias",
    L"DecodeOutputWLN.gain",
    L"EncodeOutput.GlobalInvStd",
    L"EncodeOutput.GlobalMean",
    L"EncodeOutput.LSTMoutput1.W_ifgo",
    L"EncodeOutput.LSTMoutput1.W_r_m",
    L"EncodeOutput.LSTMoutput1.bc",
    L"EncodeOutput.LSTMoutput1.bf",
    L"EncodeOutput.LSTMoutput1.bi",
    L"EncodeOutput.LSTMoutput1.bo",
    L"EncodeOutput.LSTMoutput1.phole_f_c",
    L"EncodeOutput.LSTMoutput1.phole_i_c",
    L"EncodeOutput.LSTMoutput1.phole_o_c",
    L"EncodeOutput.LSTMoutput2.W_ifgo",
    L"EncodeOutput.LSTMoutput2.W_r_m",
    L"EncodeOutput.LSTMoutput2.bc",
    L"EncodeOutput.LSTMoutput2.bf",
    L"EncodeOutput.LSTMoutput2.bi",
    L"EncodeOutput.LSTMoutput2.bo",
    L"EncodeOutput.LSTMoutput2.phole_f_c",
    L"EncodeOutput.LSTMoutput2.phole_i_c",
    L"EncodeOutput.LSTMoutput2.phole_o_c",
    L"EncodeOutput.LSTMoutput3.W_ifgo",
    L"EncodeOutput.LSTMoutput3.W_r_m",
    L"EncodeOutput.LSTMoutput3.bc",
    L"EncodeOutput.LSTMoutput3.bf",
    L"EncodeOutput.LSTMoutput3.bi",
    L"EncodeOutput.LSTMoutput3.bo",
    L"EncodeOutput.LSTMoutput3.phole_f_c",
    L"EncodeOutput.LSTMoutput3.phole_i_c",
    L"EncodeOutput.LSTMoutput3.phole_o_c",
    L"EncodeOutput.LSTMoutput4.W_ifgo",
    L"EncodeOutput.LSTMoutput4.W_r_m",
    L"EncodeOutput.LSTMoutput4.bc",
    L"EncodeOutput.LSTMoutput4.bf",
    L"EncodeOutput.LSTMoutput4.bi",
    L"EncodeOutput.LSTMoutput4.bo",
    L"EncodeOutput.LSTMoutput4.phole_f_c",
    L"EncodeOutput.LSTMoutput4.phole_i_c",
    L"EncodeOutput.LSTMoutput4.phole_o_c",
    L"EncodeOutput.LSTMoutput5.W_ifgo",
    L"EncodeOutput.LSTMoutput5.W_r_m",
    L"EncodeOutput.LSTMoutput5.bc",
    L"EncodeOutput.LSTMoutput5.bf",
    L"EncodeOutput.LSTMoutput5.bi",
    L"EncodeOutput.LSTMoutput5.bo",
    L"EncodeOutput.LSTMoutput5.phole_f_c",
    L"EncodeOutput.LSTMoutput5.phole_i_c",
    L"EncodeOutput.LSTMoutput5.phole_o_c",
    L"EncodeOutput.LSTMoutput6.W_ifgo",
    L"EncodeOutput.LSTMoutput6.W_r_m",
    L"EncodeOutput.LSTMoutput6.bc",
    L"EncodeOutput.LSTMoutput6.bf",
    L"EncodeOutput.LSTMoutput6.bi",
    L"EncodeOutput.LSTMoutput6.bo",
    L"EncodeOutput.LSTMoutput6.phole_f_c",
    L"EncodeOutput.LSTMoutput6.phole_i_c",
    L"EncodeOutput.LSTMoutput6.phole_o_c",
    L"EncodeOutputW.W",
    L"EncodeOutputW.b",
    L"EncodeOutputWLN.bias",
    L"EncodeOutputWLN.gain",
    L"Wm",
    L"__DecodeOutputWLN.xHat.y",
    L"__EncodeOutputWLN.xHat.y",
    L"bm",
};

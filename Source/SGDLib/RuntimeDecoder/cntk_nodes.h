#pragma once

// LayerNorm(dim, inputx, initBias, initScale) = 
// [
//     gain = ParamFromValue(dim, 1, initScale);
//     #        f = ConstantTensor (4, (1))
//     #        fInv = Reciprocal (f)
//     #        gain = fInv .* Log (BS.Constants.One + Exp (f .* ParameterTensor ((1), initValue=0.99537863/* 1/f*ln (e^f-1) */))) # init value is 1
//     bias = ParamFromValue(dim, 1, initBias);
// 
//     mean = ReduceMean (inputx)
//     x0 = inputx - mean;
//     std = Sqrt (ReduceMean (x0 .* x0))
//     xHat = ElementDivide (x0, std+0.00001)
// 
//     # denormalize with learned parameters
//     o = xHat .* gain + bias
// ]
class LayerNorm
{
public:
    LayerNorm(const CModelParams& params, const std::wstring& prefix)
        : gain(params.GetVectorParams(prefix + L".gain")),
          bias(params.GetVectorParams(prefix + L".bias"))
    {
    }

    void Forward(CVector& o, const CVector& inputx)
    {
        CVector& x0(o);
        CVector& xHat(o);

        auto mean = ReduceMean(inputx);
        x0.SetMinus(inputx, mean);
        auto std = SqrMoment(x0);
        xHat.SetElementDivide(x0, std+0.00001f);
        o.SetElementTimes(xHat, gain);
        o.SetPlus(o, bias);
    }

private:
    float ReduceMean(const CVector& x)
    {
        return x.Sum() / x.M;
    }

    float SqrMoment(const CVector& x)
    {
        return sqrt(x.SqrSum() / x.M);
    }

private:
    const CVector& gain;
    const CVector& bias;
};

#define DEFAULT_HIDDEN_ACTIVATION 0.1f


struct LSTMState
{
    LSTMState()
    {
    }

    LSTMState(uint32_t oM, uint32_t ctM)
        : o(oM),
          ct(ctM)
    {
    }

    LSTMState(LSTMState&& that)
        : o(std::move(that.o)),
          ct(std::move(that.ct))
    {
    }

    void operator=(LSTMState&& that)
    {
        o = std::move(that.o);
        ct = std::move(that.ct);
    }

    CVector o;
    CVector ct;
};


// LSTMPLayer_PY(inputDim, cellDim, outputDim, inputx, initWScale, initBias) = 
// [
//     cellDimX2 = cellDim * 2;
//     cellDimX3 = cellDim * 3;
//     cellDimX4 = cellDim * 4;
// 
//     wx = ParamFromRand(cellDimX4, inputDim, initWScale); 
//     b = ParamFromValue(cellDimX4, 1, initBias);
//     wh = ParamFromRand(cellDimX4, outputDim, initWScale);
// 
//    
//     dh = PastValue(outputDim, o, timeStep=1);
//     dc = PastValue(cellDim, ct, timeStep=1);
//
//
//     wxx = wx * inputx;
//     wxxpb = wxx + b;
//     
//     whh = wh * dh;
//     
//     wxxpbpwhh = wxxpb + whh;
//             
//     G1 = RowSlice(0, cellDim, wxxpbpwhh);
//     G2 = RowSlice(cellDim, cellDim, wxxpbpwhh);
//     G3 = RowSlice(cellDimX2, cellDim, wxxpbpwhh);
//     G4 = RowSlice(cellDimX3, cellDim, wxxpbpwhh);
//     
//     it = Sigmoid (G1 );
// 
//     bit = it .* Tanh( G2 );
// 
//     
//     ft = Sigmoid( G3 );
// 
//     bft = ft .* dc;
// 
//     ct = bft + bit;
//     ot = Sigmoid( G4 );
// 
//     mt = ot .* Tanh(ct);
//
//     Wmr = ParamFromRand(outputDim, cellDim, initWScale);
//     o = Times(Wmr, mt); 
// ]
class LSTMPLayer_PY
{
public:
    LSTMPLayer_PY(const CModelParams& params, const std::wstring& prefix)
        : wx(params.GetMatrixParams(prefix + L".wx")),
          b(params.GetVectorParams(prefix + L".b")),
          wh(params.GetMatrixParams(prefix + L".wh")),
          Wmr(params.GetMatrixParams(prefix + L".Wmr"))
    {
        Reset();
    }

    void Reset()
    {
        m_s = NewState();
        m_s.ct.SetElement(DEFAULT_HIDDEN_ACTIVATION);
        m_s.o.SetElement(DEFAULT_HIDDEN_ACTIVATION);
    }

    void Forward(const CVector& inputx)
    {
        Forward(m_s, m_s, inputx);
    }

    LSTMState NewState() const
    {
        return { Wmr->M, wh->M / 4 };
    }

    LSTMState DetachState()
    {
        return std::move(m_s);
    }

    void Forward(LSTMState& s, const LSTMState& s0, const CVector& inputx)
    {
        auto& o = s.o;
        auto& ct = s.ct;

        const auto& dh = s0.o;
        const auto& dc = s0.ct;

        _stack_CVector(G1, wh->M / 4);
        _stack_CVector(G2, wh->M / 4);
        _stack_CVector(G3, wh->M / 4);
        _stack_CVector(G4, wh->M / 4);

        {
            _stack_CVector(wxxpbpwhh, wh->M);
            {
                CVector& wxx(wxxpbpwhh);
                CVector& wxxpb(wxxpbpwhh);
                _stack_CVector(whh, wh->M);

                wxx.SetTimes(wx, inputx);
                wxxpb.SetPlus(wxx, b);

                whh.SetTimes(wh, dh);

                wxxpbpwhh.SetPlus(wxxpb, whh);
            }

            G1.SetRowSlice(0 * G1.M, G1.M, wxxpbpwhh);
            G2.SetRowSlice(1 * G1.M, G2.M, wxxpbpwhh);
            G3.SetRowSlice(2 * G1.M, G3.M, wxxpbpwhh);
            G4.SetRowSlice(3 * G1.M, G4.M, wxxpbpwhh);
        }

        CVector& it(G1);
        it.SetSigmoid(G1);

        CVector& bit(G2);
        G2.SetTanh(G2);
        bit.SetElementTimes(it, G2);

        CVector& ft(G3);
        ft.SetSigmoid(G3);

        CVector& bft(G3);
        bft.SetElementTimes(ft, dc);

        CVector& ot(G4);
        ct.SetPlus(bft, bit);
        ot.SetSigmoid(G4);

        CVector& tanh_ct(G3);
        tanh_ct.SetTanh(ct);

        CVector& mt(G4);
        mt.SetElementTimes(ot, tanh_ct);

        o.SetTimes(Wmr, mt);
    }

private:
    const IMatrix* wx;
    const CVector& b;
    const IMatrix* wh;

    const IMatrix* Wmr;

    LSTMState m_s;

public:
    // Do not use after DetachState() is called.
    // Instead, access through "o" field in each LSTMState.
    const CVector& o() const
    {
        return m_s.o;
    }
};


// LSTMPLayer_PY_LN(inputDim, cellDim, outputDim, inputx, initWScale, initBias) = 
// [
//     cellDimX2 = cellDim * 2;
//     cellDimX3 = cellDim * 3;
//     cellDimX4 = cellDim * 4;
// 
//     wx = ParamFromRand(cellDimX4, inputDim, initWScale); 
//     b = ParamFromValue(cellDimX4, 1, initBias);
//     wh = ParamFromRand(cellDimX4, outputDim, initWScale);
// 
//    
//     dh = PastValue(outputDim, o, timeStep=1);
//     dc = PastValue(cellDim, ct, timeStep=1);
// 
// 
//     wxx = wx * inputx;
//     wxxln = LayerNorm(cellDimX4, wxx, initBias, initWScale).o;
//     wxxpb = wxxln + b;
//     
//     whh = wh * dh;
//     whhln = LayerNorm(cellDimX4, whh, initBias, initWScale).o;
//     
//     wxxpbpwhh = wxxpb + whhln;
//             
//     G1 = RowSlice(0, cellDim, wxxpbpwhh);
//     G2 = RowSlice(cellDim, cellDim, wxxpbpwhh);
//     G3 = RowSlice(cellDimX2, cellDim, wxxpbpwhh);
//     G4 = RowSlice(cellDimX3, cellDim, wxxpbpwhh);
//     
//     it = Sigmoid (G1 );
// 
//     bit = it .* Tanh( G2 );
// 
//     
//     ft = Sigmoid( G3 );
// 
//     bft = ft .* dc;
// 
//     ct = bft + bit;
//     ctln = LayerNorm(cellDim,ct, initBias, initWScale).o;
//     ot = Sigmoid( G4 );
// 
//     mt = ot .* Tanh(ctln);
// 
//     Wmr = ParamFromRand(outputDim, cellDim, initWScale);
//     o = Times(Wmr, mt); 
// ]
class LSTMPLayer_PY_LN
{
public:
    LSTMPLayer_PY_LN(const CModelParams& params, const std::wstring& prefix)
        : wx(params.GetMatrixParams(prefix + L".wx")),
          b(params.GetVectorParams(prefix + L".b")),
          wh(params.GetMatrixParams(prefix + L".wh")),
          wxxln(params, prefix + L".wxxln"),
          whhln(params, prefix + L".whhln"),
          ctln(params, prefix + L".ctln"),
          Wmr(params.GetMatrixParams(prefix + L".Wmr"))
    {
        Reset();
    }

    void Reset()
    {
        m_s = NewState();
        m_s.ct.SetElement(DEFAULT_HIDDEN_ACTIVATION);
        m_s.o.SetElement(DEFAULT_HIDDEN_ACTIVATION);
    }

    void Forward(const CVector& inputx)
    {
        Forward(m_s, m_s, inputx);
    }

    LSTMState NewState() const
    {
        return { Wmr->M, wx->M / 4 };
    }

    LSTMState DetachState()
    {
        return std::move(m_s);
    }

    void Forward(LSTMState& s, const LSTMState& s0, const CVector& inputx)
    {
        auto& o = s.o;
        auto& ct = s.ct;

        const auto& dh = s0.o;
        const auto& dc = s0.ct;

        _stack_CVector(G1, wx->M / 4);
        _stack_CVector(G2, wx->M / 4);
        _stack_CVector(G3, wx->M / 4);
        _stack_CVector(G4, wx->M / 4);

        {
            _stack_CVector(wxxpbpwhh, wx->M);
            {
                CVector& wxx(wxxpbpwhh);
                CVector& wxxln_o(wxxpbpwhh);
                CVector& wxxpb(wxxpbpwhh);

                _stack_CVector(whh, wh->M);
                CVector& whhln_o(whh);

                wxx.SetTimes(wx, inputx);
                wxxln.Forward(wxxln_o, wxx);
                wxxpb.SetPlus(wxxln_o, b);

                whh.SetTimes(wh, dh);
                whhln.Forward(whhln_o, whh);

                wxxpbpwhh.SetPlus(wxxpb, whhln_o);
            }

            G1.SetRowSlice(0 * G1.M, G1.M, wxxpbpwhh);
            G2.SetRowSlice(1 * G1.M, G2.M, wxxpbpwhh);
            G3.SetRowSlice(2 * G1.M, G3.M, wxxpbpwhh);
            G4.SetRowSlice(3 * G1.M, G4.M, wxxpbpwhh);
        }

        CVector& it(G1);
        it.SetSigmoid(G1);

        CVector& bit(G2);
        G2.SetTanh(G2);
        bit.SetElementTimes(it, G2);

        CVector& ft(G3);
        ft.SetSigmoid(G3);

        CVector& bft(G3);
        bft.SetElementTimes(ft, dc);

        ct.SetPlus(bft, bit);
        CVector& ctln_o(G3);
        ctln.Forward(ctln_o, ct);
        CVector& ot(G4);
        ot.SetSigmoid(G4);

        ctln_o.SetTanh(ctln_o);
        CVector& mt(G4);
        mt.SetElementTimes(ot, ctln_o);

        o.SetTimes(Wmr, mt);
    }

private:
    const IMatrix* wx;
    const CVector& b;
    const IMatrix* wh;

    LayerNorm wxxln;
    LayerNorm whhln;
    LayerNorm ctln;

    const IMatrix* Wmr;

    LSTMState m_s;

public:
    // Do not use after DetachState() is called.
    // Instead, access through "o" field in each LSTMState.
    const CVector& o() const
    {
        return m_s.o;
    }
};


// SVD version
class LSTMPLayer_PY_LN_svd
{
public:
    LSTMPLayer_PY_LN_svd(const CModelParams& params, const std::wstring& prefix)
        : wx_U(params.GetMatrixParams(prefix + L".wx_U")),
          wx_V(params.GetMatrixParams(prefix + L".wx_V")),
          b(params.GetVectorParams(prefix + L".b")),
          wh_U(params.GetMatrixParams(prefix + L".wh_U")),
          wh_V(params.GetMatrixParams(prefix + L".wh_V")),
          wxxln(params, prefix + L".wxxln"),
          whhln(params, prefix + L".whhln"),
          ctln(params, prefix + L".ctln"),
          Wmr_U(params.GetMatrixParams(prefix + L".Wmr_U")),
          Wmr_V(params.GetMatrixParams(prefix + L".Wmr_V"))
    {
        Reset();

        // rfail("Deprecated.  Try optimize_model first.\n");
    }

    void Reset()
    {
        m_s = NewState();
        m_s.ct.SetElement(DEFAULT_HIDDEN_ACTIVATION);
        m_s.o.SetElement(DEFAULT_HIDDEN_ACTIVATION);
    }

    void Forward(const CVector& inputx)
    {
        Forward(m_s, m_s, inputx);
    }

    LSTMState NewState() const
    {
        return { Wmr_U->M, wx_U->M / 4 };
    }

    LSTMState DetachState()
    {
        return std::move(m_s);
    }

    void Forward(LSTMState& s, const LSTMState& s0, const CVector& inputx)
    {
        auto& o = s.o;
        auto& ct = s.ct;

        const auto& dh = s0.o;
        const auto& dc = s0.ct;

        _stack_CVector(G1, wx_U->M / 4);
        _stack_CVector(G2, wx_U->M / 4);
        _stack_CVector(G3, wx_U->M / 4);
        _stack_CVector(G4, wx_U->M / 4);

        {
            _stack_CVector(wxxpbpwhh, wx_U->M);
            {
                CVector& wxx(wxxpbpwhh);
                CVector& wxxln_o(wxxpbpwhh);
                CVector& wxxpb(wxxpbpwhh);

                _stack_CVector(whh, wh_U->M);
                CVector& whhln_o(whh);

                {
                    _stack_CVector(wxx1, wx_U->N);
                    wxx1.SetTimes(wx_V, inputx);
                    wxx.SetTimes(wx_U, wxx1);
                }
                wxxln.Forward(wxxln_o, wxx);
                wxxpb.SetPlus(wxxln_o, b);

                {
                    _stack_CVector(whh1, wh_U->N);
                    whh1.SetTimes(wh_V, dh);
                    whh.SetTimes(wh_U, whh1);
                }
                whhln.Forward(whhln_o, whh);

                wxxpbpwhh.SetPlus(wxxpb, whhln_o);
            }

            G1.SetRowSlice(0 * G1.M, G1.M, wxxpbpwhh);
            G2.SetRowSlice(1 * G1.M, G2.M, wxxpbpwhh);
            G3.SetRowSlice(2 * G1.M, G3.M, wxxpbpwhh);
            G4.SetRowSlice(3 * G1.M, G4.M, wxxpbpwhh);
        }

        CVector& it(G1);
        it.SetSigmoid(G1);

        CVector& bit(G2);
        G2.SetTanh(G2);
        bit.SetElementTimes(it, G2);

        CVector& ft(G3);
        ft.SetSigmoid(G3);

        CVector& bft(G3);
        bft.SetElementTimes(ft, dc);

        ct.SetPlus(bft, bit);
        CVector& ctln_o(G3);
        ctln.Forward(ctln_o, ct);
        CVector& ot(G4);
        ot.SetSigmoid(G4);

        ctln_o.SetTanh(ctln_o);
        CVector& mt(G4);
        mt.SetElementTimes(ot, ctln_o);

        {
            _stack_CVector(o1, Wmr_U->N);
            o1.SetTimes(Wmr_V, mt);
            // TODO: (perf)
            // 1) o1 is smaller than o, better to save as state
            // 2) next_layer.Wmr_V * Wmr_U can be precomputed and saved
            //    reduces both computation and memory (model size)
            o.SetTimes(Wmr_U, o1);
        }
    }

private:
    const IMatrix* wx_U;
    const IMatrix* wx_V;
    const CVector& b;
    const IMatrix* wh_U;
    const IMatrix* wh_V;

    LayerNorm wxxln;
    LayerNorm whhln;
    LayerNorm ctln;

    const IMatrix* Wmr_U;
    const IMatrix* Wmr_V;

    LSTMState m_s;

public:
    // Do not use after DetachState() is called.
    // Instead, access through "o" field in each LSTMState.
    const CVector& o() const
    {
        return m_s.o;
    }
};


// SVD version (cross-layer optimized)
class LSTMPLayer_PY_LN_svdopt
{
public:
    LSTMPLayer_PY_LN_svdopt(const CModelParams& params, const std::wstring& prefix)
        : wx_U(params.GetMatrixParams(prefix + L".wx_U")),
          wx_V(params.GetMatrixParams(prefix + L".wx_V")),
          b(params.GetVectorParams(prefix + L".b")),
          wh_U(params.GetMatrixParams(prefix + L".wh_U")),
          wh_V(params.GetMatrixParams(prefix + L".wh_V")),
          whh1_init(params.GetVectorParams(prefix + L".wh_V*dh_init")),
          wxxln(params, prefix + L".wxxln"),
          whhln(params, prefix + L".whhln"),
          ctln(params, prefix + L".ctln"),
          Wmr_V(params.GetMatrixParams(prefix + L".Wmr_V"))
    {
        Reset();
    }

    void Reset()
    {
        m_s = NewState();
        m_s.ct.SetElement(DEFAULT_HIDDEN_ACTIVATION);
        m_s.IsInitState = true;
    }

    void Forward(const CVector& inputx)
    {
        Forward(m_s, m_s, inputx);
    }

    struct State
    {
        State()
        {
        }

        State(uint32_t oM, uint32_t ctM)
            : o(oM),
              ct(ctM)
        {
        }

        State(State&& that)
            : IsInitState(that.IsInitState),
              o(std::move(that.o)),
              ct(std::move(that.ct))
        {
        }

        void operator=(State&& that)
        {
            IsInitState = that.IsInitState;
            o = std::move(that.o);
            ct = std::move(that.ct);
        }

        bool IsInitState = false;
        CVector o;
        CVector ct;
    };

    State NewState() const
    {
        return { Wmr_V->M, wx_U->M / 4 };
    }

    State DetachState()
    {
        return std::move(m_s);
    }

    void Forward(State& s, const State& s0, const CVector& inputx)
    {
        auto& o = s.o;
        auto& ct = s.ct;

        const auto& dh = s0.o;
        const auto& dc = s0.ct;

        _stack_CVector(G1, wx_U->M / 4);
        _stack_CVector(G2, wx_U->M / 4);
        _stack_CVector(G3, wx_U->M / 4);
        _stack_CVector(G4, wx_U->M / 4);

        {
            _stack_CVector(wxxpbpwhh, wx_U->M);
            {
                CVector& wxx(wxxpbpwhh);
                CVector& wxxln_o(wxxpbpwhh);
                CVector& wxxpb(wxxpbpwhh);

                _stack_CVector(whh, wh_U->M);
                CVector& whhln_o(whh);

                {
                    _stack_CVector(wxx1, wx_U->N);
                    wxx1.SetTimes(wx_V, inputx);
                    wxx.SetTimes(wx_U, wxx1);
                }
                wxxln.Forward(wxxln_o, wxx);
                wxxpb.SetPlus(wxxln_o, b);

                if (s0.IsInitState)
                {
                    whh.SetTimes(wh_U, whh1_init);
                }
                else
                {
                    _stack_CVector(whh1, wh_U->N);
                    whh1.SetTimes(wh_V, dh);
                    whh.SetTimes(wh_U, whh1);
                }

                whhln.Forward(whhln_o, whh);

                wxxpbpwhh.SetPlus(wxxpb, whhln_o);
            }

            G1.SetRowSlice(0 * G1.M, G1.M, wxxpbpwhh);
            G2.SetRowSlice(1 * G1.M, G2.M, wxxpbpwhh);
            G3.SetRowSlice(2 * G1.M, G3.M, wxxpbpwhh);
            G4.SetRowSlice(3 * G1.M, G4.M, wxxpbpwhh);
        }

        CVector& it(G1);
        it.SetSigmoid(G1);

        CVector& bit(G2);
        G2.SetTanh(G2);
        bit.SetElementTimes(it, G2);

        CVector& ft(G3);
        ft.SetSigmoid(G3);

        CVector& bft(G3);
        bft.SetElementTimes(ft, dc);

        ct.SetPlus(bft, bit);
        CVector& ctln_o(G3);
        ctln.Forward(ctln_o, ct);
        CVector& ot(G4);
        ot.SetSigmoid(G4);

        ctln_o.SetTanh(ctln_o);
        CVector& mt(G4);
        mt.SetElementTimes(ot, ctln_o);

        o.SetTimes(Wmr_V, mt);
        s.IsInitState = false;
    }

private:
    const IMatrix* wx_U;
    const IMatrix* wx_V;
    const CVector& b;
    const IMatrix* wh_U;
    const IMatrix* wh_V;
    const CVector& whh1_init;

    LayerNorm wxxln;
    LayerNorm whhln;
    LayerNorm ctln;

    const IMatrix* Wmr_V;

    State m_s;

public:
    // Do not use after DetachState() is called.
    // Instead, access through "o" field in each State.
    const CVector& o() const
    {
        return m_s.o;
    }
};


// DNNLinearLayer(inputDim, outputDim, x, initWScale, initBias) = 
// [
//     W = ParamFromRand(outputDim, inputDim, initWScale);
//     b = ParamFromValue(outputDim, 1, initBias); 
//     t = Times(W, x);
//     o = t + b;
// ]
class DNNLinearLayer
{
public:
    DNNLinearLayer(const CModelParams& params, const std::wstring& prefix)
        : W(params.GetMatrixParams(prefix + L".W")),
          b(params.GetVectorParams(prefix + L".b"))
    {
    }

    void Forward(CVector& o, const CVector& x)
    {
        CVector& t(o);
        t.SetTimes(W, x);
        o.SetPlus(t, b);
    }

    void Forward_OneHot(CVector& o, uint32_t j)
    {
        CVector& t(o);
        t.SetColumn(W, j);
        o.SetPlus(t, b);
    }

    uint32_t M() const
    {
        return W->M;
    }

    uint32_t N() const
    {
        return W->N;
    }

private:
    const IMatrix* W;
    const CVector& b;
};


// SVD version
class DNNLinearLayer_svd
{
public:
    DNNLinearLayer_svd(const CModelParams& params, const std::wstring& prefix)
        : W_U(params.GetMatrixParams(prefix + L".W_U")),
          W_V(params.GetMatrixParams(prefix + L".W_V")),
          b(params.GetVectorParams(prefix + L".b"))
    {
    }

    void Forward(CVector& o, const CVector& x)
    {
        CVector& t(o);

        {
            _stack_CVector(t1, W_U->N);
            t1.SetTimes(W_V, x);
            t.SetTimes(W_U, t1);
        }

        o.SetPlus(t, b);
    }

    void Forward_OneHot(CVector& o, uint32_t j)
    {
        CVector& t(o);

        {
            _stack_CVector(t1, W_U->N);
            t1.SetColumn(W_V, j);
            t.SetTimes(W_U, t1);
        }

        o.SetPlus(t, b);
    }

    uint32_t M() const
    {
        return W_U->M;
    }

    uint32_t N() const
    {
        return W_V->N;
    }

private:
    const IMatrix* W_U;
    const IMatrix* W_V;
    const CVector& b;
};


// LSTMPComponent(input_hidden_Dim, outputDim, cellDim, cellDimT2, cellDimT3, cellDimT4, inputx)
// {
//     # declarations of model parameters
//     W_ifgo = Parameter( cellDimT4, input_hidden_Dim, init=uniform, initValueScale=1)

//     bo = Parameter(cellDim, init=fixedvalue, value=0.0);
//     bc = Parameter(cellDim, init=fixedvalue, value=0.0);
//     bi = Parameter(cellDim, init=fixedvalue, value=0.0);
//     bf = Parameter(cellDim, init=fixedvalue, value=1.0);
//     
//     phole_i_c = Parameter(cellDim, init=uniform, initValueScale=1)
//     phole_f_c = Parameter(cellDim, init=uniform, initValueScale=1)
//     phole_o_c = Parameter(cellDim, init=uniform, initValueScale=1)
// 
//     W_r_m = Parameter(outputDim, cellDim, init=uniform, initValueScale=1)
// 
//     # obtain c_(t-1) and r_(t-1) by delay
//     r_t_prev = Delay(outputDim, output, delayTime=1)
//     c_t_prev = Delay(cellDim, c_t, delayTime=1)
// 
//     # F stands for forward propagation
//     inputx_r = RowStack(inputx, r_t_prev)
//     F_ifgo = Times(W_ifgo, inputx_r)
//     
//     F_i_c = ElementTimes(phole_i_c, c_t_prev)
//     F_f_c = ElementTimes(phole_f_c, c_t_prev)
//     F_o_c = ElementTimes(phole_o_c, c_t)
//     
//     # input and forget gate
//     i_t = Sigmoid(  Plus (Plus( RowSlice(0, cellDim, F_ifgo) , F_i_c), bi) )
//     f_t = Sigmoid(  Plus (Plus( RowSlice(cellDim, cellDim, F_ifgo) , F_f_c), bf) )

//     g_t_0 = RowSlice(cellDimT2, cellDim, F_ifgo)
//     g_t = Tanh(Plus(g_t_0, bc))

//     # output gate
//     o_t_0 = RowSlice(cellDimT3, cellDim, F_ifgo)
//     o_t = Sigmoid(  Plus(Plus( o_t_0 , F_o_c), bo) )
//     
//     # memory cell
//     c_t = Plus(ElementTimes(f_t, c_t_prev), ElementTimes(i_t, g_t))
  
//     # projection layer
//     m_t = ElementTimes(o_t, Tanh(c_t))
//     output = Times(W_r_m, m_t)
// }
class LSTMPComponent
{
public:
    LSTMPComponent(const CModelParams& params, const std::wstring& prefix)
        : W_ifgo(params.GetMatrixParams(prefix + L".W_ifgo")),

          bo(params.GetVectorParams(prefix + L".bo")),
          bc(params.GetVectorParams(prefix + L".bc")),
          bi(params.GetVectorParams(prefix + L".bi")),
          bf(params.GetVectorParams(prefix + L".bf")),

          phole_i_c(params.GetVectorParams(prefix + L".phole_i_c")),
          phole_f_c(params.GetVectorParams(prefix + L".phole_f_c")),
          phole_o_c(params.GetVectorParams(prefix + L".phole_o_c")),

          W_r_m(params.GetMatrixParams(prefix + L".W_r_m"))
    {
        Reset();
    }

    void Reset()
    {
        m_s = NewState();
        m_s.ct.SetElement(DEFAULT_HIDDEN_ACTIVATION);
        m_s.o.SetElement(DEFAULT_HIDDEN_ACTIVATION);
    }

    void Forward(const CVector& inputx)
    {
        Forward(m_s, m_s, inputx);
    }

    LSTMState NewState() const
    {
        return { W_r_m->M, bo.M };
    }

    LSTMState DetachState()
    {
        return std::move(m_s);
    }

    void Forward(LSTMState& s, const LSTMState& s0, const CVector& inputx)
    {
        auto cellDim = bo.M;
        auto cellDimT2 = cellDim * 2;
        auto cellDimT3 = cellDim * 3;

        auto& output = s.o;
        auto& c_t = s.ct;

        const auto& r_t_prev = s0.o;
        const auto& c_t_prev = s0.ct;

        _stack_CVector(F_ifgo, W_ifgo->M);
        {
            _stack_CVector(inputx_r, W_ifgo->N);
            inputx_r.SetRowStack(inputx, r_t_prev);
            F_ifgo.SetTimes(W_ifgo, inputx_r);
        }

        _stack_CVector(F_i_c, phole_i_c.M);
        _stack_CVector(F_f_c, phole_f_c.M);
        F_i_c.SetElementTimes(phole_i_c, c_t_prev);
        F_f_c.SetElementTimes(phole_f_c, c_t_prev);

        _stack_CVector(i_t, cellDim);
        i_t.SetRowSlice(0, cellDim, F_ifgo);
        i_t.SetPlus(i_t, F_i_c);
        i_t.SetPlus(i_t, bi);
        i_t.SetSigmoid(i_t);

        _stack_CVector(f_t, cellDim);
        f_t.SetRowSlice(cellDim, cellDim, F_ifgo);
        f_t.SetPlus(f_t, F_f_c);
        f_t.SetPlus(f_t, bf);
        f_t.SetSigmoid(f_t);

        _stack_CVector(g_t, cellDim);
        g_t.SetRowSlice(cellDimT2, cellDim, F_ifgo);
        g_t.SetPlus(g_t, bc);
        g_t.SetTanh(g_t);

        f_t.SetElementTimes(f_t, c_t_prev);
        i_t.SetElementTimes(i_t, g_t);
        c_t.SetPlus(f_t, i_t);

        // Notice the order difference (brainscript is apparently functional)
        CVector& F_o_c(g_t);
        F_o_c.SetElementTimes(phole_o_c, c_t);

        _stack_CVector(o_t, cellDim);
        o_t.SetRowSlice(cellDimT3, cellDim, F_ifgo);
        o_t.SetPlus(o_t, F_o_c);
        o_t.SetPlus(o_t, bo);
        o_t.SetSigmoid(o_t);

        CVector& m_t(g_t);
        m_t.SetTanh(c_t);
        m_t.SetElementTimes(o_t, m_t);

        output.SetTimes(W_r_m, m_t);
    }

private:
    const IMatrix* W_ifgo;

    const CVector& bo;
    const CVector& bc;
    const CVector& bi;
    const CVector& bf;

    const CVector& phole_i_c;
    const CVector& phole_f_c;
    const CVector& phole_o_c;

    const IMatrix* W_r_m;

    LSTMState m_s;

public:
    // Do not use after DetachState() is called.
    // Instead, access through "o" field in each State.
    const CVector& o() const
    {
        return m_s.o;
    }
};

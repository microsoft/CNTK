#pragma once

class IEncoder
{
public:
    virtual ~IEncoder() {}
    virtual void Reset() = 0;
    virtual const CVector& Forward(const float* feats, size_t baseFeatDim) = 0;
};

class IPredictor
{
public:
    virtual ~IPredictor() {}

    class IState
    {
    public:
        virtual ~IState() {}
    };

    virtual void Reset() = 0;
    virtual const CVector& Forward(unsigned int label) = 0;
    virtual std::unique_ptr<IState> NewState() const = 0;
    virtual std::unique_ptr<IState> DetachState() = 0;
    virtual const CVector& Forward(IState& s, const IState& s0, unsigned int label) = 0;
};

class IJoint
{
public:
    virtual ~IJoint() {}
    virtual const CVector& Forward(const CVector& EncodeOutputWLN, const CVector& DecodeOutputWLN) = 0;
    virtual const CVector& Forward_SoftMax(const CVector& EncodeOutputWLN, const CVector& DecodeOutputWLN) = 0;
};

#include "rnnt_nodes_1.h"
#include "rnnt_nodes_2.h"
#include "rnnt_nodes_3.h"
#include "rnnt_nodes_3svd.h"
#include "rnnt_nodes_3svdopt.h"

enum class ModelVersion
{
    v1 = 100,
    v2 = 200,
    v3 = 300,
    v3svd = 310,
    v3svdopt = 311,
};


ModelVersion get_model_version(const CModelParams& params)
{
    auto names = params.GetParamNames();

    if (names == v1ParamNames)
        return ModelVersion::v1;
    else if (names == v2ParamNames)
        return ModelVersion::v2;
    else if (names == v3ParamNames)
        return ModelVersion::v3;
    else if (names == v3svdParamNames)
        return ModelVersion::v3svd;
    else if (names == v3svdoptParamNames)
        return ModelVersion::v3svdopt;
    else
    {
        for (const auto& name: names)
            printf("        L\"%S\",\n", name.c_str());
        fflush(stdout);
        rfail("unknown model file\n");
    }
}

std::unique_ptr<IEncoder> make_unique_encoder(const CModelParams& params)
{
    auto v = get_model_version(params);
    switch (v)
    {
        case ModelVersion::v1:
        return std::make_unique<CEncoder_1>(params);

        case ModelVersion::v2:
        return std::make_unique<CEncoder_2>(params);

        case ModelVersion::v3:
        return std::make_unique<CEncoder_3>(params);

        case ModelVersion::v3svd:
        return std::make_unique<CEncoder_3svd>(params);

        case ModelVersion::v3svdopt:
        return std::make_unique<CEncoder_3svdopt>(params);

        default:
        rfail("unknown model version %d\n", int(v));
    }
}


std::unique_ptr<IPredictor> make_unique_predictor(const CModelParams& params)
{
    auto v = get_model_version(params);
    switch (v)
    {
        case ModelVersion::v1:
        return std::make_unique<CPredictor_1>(params);

        case ModelVersion::v2:
        return std::make_unique<CPredictor_2>(params);

        case ModelVersion::v3:
        return std::make_unique<CPredictor_3>(params);

        case ModelVersion::v3svd:
        return std::make_unique<CPredictor_3svd>(params);

        case ModelVersion::v3svdopt:
        return std::make_unique<CPredictor_3svdopt>(params);

        default:
        rfail("unknown model version %d\n", int(v));
    }
}


std::unique_ptr<IJoint> make_unique_joint(const CModelParams& params)
{
    auto v = get_model_version(params);
    switch (v)
    {
        case ModelVersion::v1:
        return std::make_unique<CJoint_1>(params);

        case ModelVersion::v2:
        return std::make_unique<CJoint_2>(params);

        case ModelVersion::v3:
        return std::make_unique<CJoint_3>(params);

        case ModelVersion::v3svd:
        return std::make_unique<CJoint_3svd>(params);

        case ModelVersion::v3svdopt:
        return std::make_unique<CJoint_3svdopt>(params);

        default:
        rfail("unknown model version %d\n", int(v));
    }
}

void optimize_model(CModelParams& params)
{
    // Hack: Remove "EncodeOutputWLN." and "DecodeOutputWLN." prefixes
    // TODO: When model's param names settle down, properly change rnnt_nodes
    // definition.
    auto names = params.GetParamNames();
    const wchar_t* Prefixes[] = { L"EncodeOutputWLN.", L"DecodeOutputWLN." };
    for (const auto& name: names)
        for (auto prefix: Prefixes)
            if (name.find(prefix) == 0)
                params.RenameParam(name, name.substr(wcslen(prefix)));

    auto v = get_model_version(params);
    switch (v)
    {
        case ModelVersion::v3svd:
        optimize_model_v3svd(params);
        break;

        default:
        rfail("not supported: model version %d\n", int(v));
    }
}

// ExperimentalNetworkBuilder.h -- interface to new version of NDL (and config) parser  --fseide

#pragma once

#include "Basics.h"
#include "IComputationNetBuilder.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    class ExperimentalNetworkBuilder : public IComputationNetBuilder<ElemType>
    {
        typedef shared_ptr<ComputationNetwork> ComputationNetworkPtr;
        DEVICEID_TYPE m_deviceId;
        ComputationNetworkPtr m_net;
        std::wstring m_sourceCode;
    public:
        // the constructor expects the entire source code as a wstring; if you want to read it from a file, use 'include "file"' inside
        ExperimentalNetworkBuilder(const wstring & sourceCode, DEVICEID_TYPE deviceId) : m_sourceCode(sourceCode), m_deviceId(deviceId) { }

        // build a ComputationNetwork from description language
        // TODO: change return type of these interfaces to shared_ptrs
        virtual /*IComputationNetBuilder::*/ComputationNetwork* BuildNetworkFromDescription(ComputationNetwork* = nullptr);
        // TODO: what is that function argument for?

        // load an existing file--this is the same code as for NDLNetworkBuilder.h (OK to copy it here because this is temporary code anyway)
        virtual /*IComputationNetBuilder::*/ComputationNetwork* LoadNetworkFromFile(const wstring& modelFileName, bool forceLoad = true,
                                                                                              bool bAllowNoCriterionNode = false, ComputationNetwork* anotherNetwork = nullptr)
        {
            if (!m_net || m_net->GetTotalNumberOfNodes() == 0 || forceLoad) //not built or force load
            {
                auto net = make_shared<ComputationNetwork>(m_deviceId);
                net->LoadFromFile<ElemType>(modelFileName, FileOptions::fileOptionsBinary, bAllowNoCriterionNode, anotherNetwork);
                m_net = net;
            }
            m_net->ResetEvalTimeStamp();
            return m_net.get();
        }
    };

}}}

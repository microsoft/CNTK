#if 0   // no longer needed
// ExperimentalNetworkBuilder.h -- interface to new version of NDL (and config) parser  --fseide

#pragma once

#include "Basics.h"
#include "IComputationNetBuilder.h"
#include "BestGpu.h"    // for DeviceFromConfig(), which will go away soon
#include <stdio.h>

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
        ExperimentalNetworkBuilder(const ConfigParameters & config) : m_sourceCode(config(L"ExperimentalNetworkBuilder")), m_deviceId(DeviceFromConfig(config))
        {
            if (m_deviceId < 0)
                fprintf(stderr, "ExperimentalNetworkBuilder using CPU\n");
            else
                fprintf(stderr, "ExperimentalNetworkBuilder using GPU %d\n", (int)m_deviceId);
        }
        ExperimentalNetworkBuilder(const ScriptableObjects::IConfigRecord &) { NOT_IMPLEMENTED; }

        // build a ComputationNetwork from description language
        // TODO: change return type of these interfaces to shared_ptrs
        virtual /*IComputationNetBuilder::*/ComputationNetworkPtr BuildNetworkFromDescription(ComputationNetwork* = nullptr) override;
        // TODO: that function argument is related to PairNetworkNode, which will go away (we don't support it here)

        // load an existing file--this is the same code as for NDLNetworkBuilder.h (OK to copy it here because this is temporary code anyway)
        virtual /*IComputationNetBuilder::*/ComputationNetwork* LoadNetworkFromFile(const wstring& modelFileName, bool forceLoad = true,
                                                                                    bool bAllowNoCriterionNode = false,
                                                                                    ComputationNetwork* anotherNetwork = nullptr) override
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
#endif

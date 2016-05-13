//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Basics.h"
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

// ===========================================================================
// ComputationEnvironment -- global network properties of interest to nodes
// ===========================================================================

// mode that the network is currently used in, which affects node behavior
enum class NetworkOperationMode
{
    training,    // training mode specifically means nodes should behave like training (e.g. Dropout should be active)
    inferring,   // inferring (e.g. BatchNorm should not update mean estimates)
    preComputing // precomputation is a part of training where most nodes should behave like they are inferring
};

// class to store global properties of the network that are of interest to the nodes
// For example, a network can be in 'training' or 'inference' mode, which affects what nodes like Dropout and BN do,
// or what the seq-2-seq decoder feedback signal is.
struct ComputationEnvironment
{
    // networkOperationMode tells whether we are training or inferring, which affects some nodes' behavior
    NetworkOperationMode m_networkOperationMode = NetworkOperationMode::inferring; // by default, a network is always able to infer
    bool IsInferring()     const { return m_networkOperationMode == NetworkOperationMode::inferring; }
    bool IsTraining()     const { return m_networkOperationMode == NetworkOperationMode::training; }
    bool IsPreComputing() const { return m_networkOperationMode == NetworkOperationMode::preComputing; }

    //set new value and return old one
    NetworkOperationMode SetOperationMode(NetworkOperationMode mode)
    {
        NetworkOperationMode oldMode = m_networkOperationMode;
        m_networkOperationMode = mode;
        return oldMode;
    }
    // more properties should be added here as needed
};
typedef std::shared_ptr<ComputationEnvironment> ComputationEnvironmentPtr;

// RAII wrapper for setting and reverting ComputationEnvironment::networkOperationMode
// E.g. ScopedNetworkOperationMode modeGuard(net, NetworkOperationMode::training);
// will set the mode until the end of the scope, and then revert to its old value automatically.
class ScopedNetworkOperationMode
{
    ComputationEnvironment& m_environment;
    NetworkOperationMode m_previousNetworkOperationMode;
    void operator=(const ScopedNetworkOperationMode&) = delete;
public:
    template<class ComputationNetwork> // using template to avoid dependency
    ScopedNetworkOperationMode(const std::shared_ptr<ComputationNetwork>& net, NetworkOperationMode networkOperationMode) :
        m_environment(net->Environment())
    {
        m_previousNetworkOperationMode = m_environment.SetOperationMode(networkOperationMode);
    }
    ~ScopedNetworkOperationMode() // destructor restores the previous mode
    {
        m_environment.SetOperationMode(m_previousNetworkOperationMode);
    }
};

}}}

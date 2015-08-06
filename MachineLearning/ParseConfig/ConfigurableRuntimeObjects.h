// ConfigurableRuntimeObjects.h -- base class for objects that can be instantiated from config

// ... not clear at this point whether this is necessary

#pragma once

#include <memory>   // for shared_ptr

namespace Microsoft{ namespace MSR { namespace CNTK {

    using namespace std;

    class ConfigurableRuntimeObject
    {
        //virtual void Init();    // init from config parameters
    };
    typedef shared_ptr<ConfigurableRuntimeObject> ConfigurableRuntimeObjectPtr;

}}} // end namespaces

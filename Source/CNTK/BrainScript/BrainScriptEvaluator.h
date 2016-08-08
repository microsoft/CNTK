// BrainScriptEvaluator.h -- execute what's given in a config file

#pragma once

#include "Basics.h"
#include "ScriptableObjects.h"
#include "BrainScriptParser.h"

#include <memory> // for shared_ptr

namespace Microsoft { namespace MSR { namespace BS {

using namespace std;
using namespace Microsoft::MSR::ScriptableObjects;

// -----------------------------------------------------------------------
// functions exposed by this module
// -----------------------------------------------------------------------

// understand and execute from the syntactic expression tree
ConfigValuePtr Evaluate(ExpressionPtr);                               // evaluate the expression tree
void Do(ExpressionPtr e);                                             // evaluate e.do
shared_ptr<Object> EvaluateField(ExpressionPtr e, const wstring& id); // for experimental CNTK integration

} } } // end namespaces

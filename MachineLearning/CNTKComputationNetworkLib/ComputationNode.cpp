//
// <copyright file="ComputationNode.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "ComputationNode.h"
#include "InputAndParamNodes.h"
#include "ComputationNetworkBuilder.h"  // TODO: We should only pull in NewComputationNodeFromConfig(). Nodes should not know about network at large.

namespace Microsoft { namespace MSR { namespace CNTK {

    // TODO: move more code here to speed up compilation

    // -----------------------------------------------------------------------
    // subroutines for Validate() implementations
    // -----------------------------------------------------------------------

    // helper function to infer the MBLayout for this node from inputs, for the *standard case*
    // the standard case is:
    //  - all inputs must share the same layout (e.g. adding two minibatches)
    //  - with the exception of NULL layouts (e.g. TimesNode)
    //  - all layouts may be NULL (e.g. W' = W * Exp(Stabilizer))
    //  - if there are more than one different layouts involved, this function will fail
    void ComputationNodeBase::InferMBLayoutFromInputsForStandardCase()
    {
        //wstring name = NodeName(); name;
        //fprintf(stderr, "\nDetermining Layout --> %ls:", name.c_str());
        MBLayoutPtr pMBLayout;  // starts with NULL layout
        for (auto child : m_inputs)
        {
            //wstring cname = child->NodeName(); cname;
            //fprintf(stderr, "  %ls(%s)", cname.c_str(), child->m_pMBLayout ? "." : "NULL");
            if (!child)                         // node not set yet (DelayedValueNodeBase seems to allow this)--BUGBUG: Then this function won't operate correctly.
                ;
            else if (!child->m_pMBLayout)       // NULL layout (typical for parameter nodes)
                ;
            else if (!pMBLayout)                // first non-NULL layout: just copy it
                pMBLayout = child->m_pMBLayout;
#if 0
            else if (!(*pMBLayout == *child->m_pMBLayout)) // got a layout--compare whether it is the same
//#if 0
                fprintf(stderr, "InferMBLayoutFromInputsForStandardCase: found inconsistent layout in node '%ls', mismatch detected for child '%ls'", NodeName().c_str(), child->NodeName().c_str());
//#else
                RuntimeError("InferMBLayoutFromInputsForStandardCase: found inconsistent layout in node '%ls', mismatch detected for child '%ls'", NodeName().c_str(), child->NodeName().c_str());
#endif
        }
        //fprintf(stderr, "  --> (%s)\n", pMBLayout ? "." : "NULL");
        // all are consistent: install it
        LinkToMBLayout(pMBLayout);
    }
    // single input that maps its input element-wise (e.g. Sigmoid)
    void ComputationNodeBase::ValidateUnaryMap(bool isFinalValidationPass)
    {
        assert(m_inputs.size() == 1);
        ComputationNodeBase::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();
        SetDims(m_inputs[0]->GetNumRows(), DetermineNumCols(m_inputs[0]));
        InferImageDimsFromInputs();
    }
    // binary zip operation, e.g. Plus
    // If allowScaling then one can be a sub-dimension of the other (if layout then only for rows, otherwise for cols, too).
    // This also helpfully resizes the children if not yet sized.
    void ComputationNodeBase::ValidateBinaryZip(bool isFinalValidationPass, bool allowMultiples)
    {
        assert(m_inputs.size() == 2);
        ComputationNodeBase::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase();

        ValidateInferBinaryChildrenDims();

        size_t rows0 = Inputs(0)->GetNumRows(), cols0 = Inputs(0)->GetNumCols();
        size_t rows1 = Inputs(1)->GetNumRows(), cols1 = Inputs(1)->GetNumCols();

        if (isFinalValidationPass && !(
               (rows0 == rows1 && (Inputs(0)->GetMBLayout() == Inputs(1)->GetMBLayout() || cols0 == cols1)) ||                                  // matching size (obvious case)
               (allowMultiples && (rows0 == 1 || rows1 == 1) && (Inputs(0)->GetMBLayout() == Inputs(1)->GetMBLayout() || cols0 == cols1)) ||    // one is row vec
               (allowMultiples && ((!HasMBLayout() && cols0 > cols1 && cols0 % cols1 == 0) || (cols0 == 1 && rows1 % rows0 == 0) || (cols1 == 1 && rows0 % rows1 == 0)))
           ))   // TODO: ^^ I don't understand the asymmetry of this last one
        {
            LogicError("The Matrix dimensions in the %ls %ls operation do not match.", NodeName().c_str(), OperationName().c_str());
        }

        SetDims(max(rows0, rows1), GetMBLayout() ? GetMBLayout()->GetNumCols() : max(cols0, cols1));
        InferImageDimsFromInputs();
    }
    // unary reduce-to-(1,1) operation, e.g. MatrixL1RegNode
    void ComputationNodeBase::ValidateUnaryReduce(bool isFinalValidationPass)
    {
        assert(m_inputs.size() == 1);
        ComputationNodeBase::Validate(isFinalValidationPass);
        m_pMBLayout = nullptr;    // this node does not hold mini-batch data
        SetDims(1, 1);
        InferImageDimsFromInputs();
    }
    // binary reduce-to-(1,1) operation, e.g. CrossEntropyWithSoftmaxNode
    // Currently only called by criterion nodes.
    // This function also infers child LearnableParameters. In case you wonder why this is needed for criterion nodes, there are edge cases, e.g. a
    // learnable parameter being regularized by a criterion node, where the learnable parameter is fed both into that criterion node and other places.
    void ComputationNodeBase::ValidateBinaryReduce(bool isFinalValidationPass)
    {
        ComputationNodeBase::Validate(isFinalValidationPass);
        m_pMBLayout = nullptr;              // this node does not hold mini-batch data
        ValidateInferBinaryChildrenDims();
        if (isFinalValidationPass &&
            !(Inputs(0)->GetNumRows() == Inputs(1)->GetNumRows() &&
              (Inputs(0)->HasMBLayout() || (Inputs(0)->GetNumCols() == Inputs(1)->GetNumCols()))))
            LogicError("The Matrix dimensions in the %ls %ls operation do not match.", NodeName().c_str(), OperationName().c_str());
        SetDims(1, 1);
        InferImageDimsFromInputs();
    }
    // helper function for validation
    // In bad cases of convolution, dimensions are quite complex to know.
    // This is a feature that allows a node to help resizing its input node to the expected value.
    // TODO: This is shaky by design.
    void ComputationNodeBase::ValidateInferBinaryChildrenDims()
    {
        // limited inference of children dimensions
        // if dimension not specified we assume two operands' dimensions should be the same
        // NOTE: The assert is set to check if >= 2 since this is called from nodes which have more than two children.
        //      The number of children is formally verified elsewhere, so this will not break consistency. 
        assert(m_inputs.size() >= 2);
        for (size_t index = 0; index < 2; index++)
        {
            auto in = Inputs(index);
            auto other = Inputs(1 - index);
            // borrow any unset dimension on one input from the other input
            size_t rows =                        in->GetNumRows() == 0  ? other->GetNumRows()/*borrow from peer*/ : in->GetNumRows()/*keep as is*/;
            size_t cols = (!in->HasMBLayout() && in->GetNumCols() == 0) ? other->GetNumCols()/*borrow from peer*/ : in->GetNumCols()/*keep as is*/;
            ValidateInferChildDims(index, rows, cols);
        }
    }
    template<class ElemType>
    void ComputationNode<ElemType>::ValidateInferChildDims(size_t i, size_t rows, size_t cols) //override final
    {
        if (Inputs(i)->OperationName() == OperationNameOf(LearnableParameter) && Inputs(i)->GetNumRows() == 0)
        {
            if (rows == 0 || cols == 0)
                LogicError("ValidateInferChildDims: Inferred matrix must not be empty.");
            Inputs(i)->SetDims(rows, cols);
            Inputs(i)->Validate(true);  // validate it properly
            // BUGBUG: ^^ Validate() calls are under the control of ValidateSubNetwork(). E.g. it checks whether something has changed & re-validates until there is no change. If we validate here, the change goes unnoticed.
            // big BUGBUG: This should do random initialization.
            Inputs(i)->FunctionValues().SetValue(0);
            fprintf(stderr, "ValidateInferChildDims: %ls %ls operation inferred, resized to (%d x %d), and (incorrectly) initialized to 0.\n", Inputs(i)->NodeName().c_str(), Inputs(i)->OperationName().c_str(), (int)rows, (int)cols);
        }
    }

    // -----------------------------------------------------------------------
    // others
    // -----------------------------------------------------------------------

    template<class ElemType>
    /*virtual*/ void ComputationNode<ElemType>::DumpNodeInfo(const bool /*printValues*/, File& fstream) const
    {
        fstream << L"\n" + NodeName() + L"=" + OperationName();

        if (!IsLeaf())
        {
            fstream << wstring(L"(");
            for (size_t i = 0; i<ChildrenSize(); i++)
            {
                if (i > 0)
                    fstream << wstring(L",");
                fstream << (Inputs(i) ? Inputs(i)->NodeName() : L"NULL");
            }
            fstream << wstring(L")");
        }
    }

    // -----------------------------------------------------------------------
    // instantiate the core class templates
    // -----------------------------------------------------------------------

    typedef Matrix<float> FloatMatrix;
    typedef Matrix<double> DoubleMatrix;

    atomic_ullong TimeStamp::s_timeStampCounter = ATOMIC_VAR_INIT(0);

    template<> std::map<size_t, std::map<size_t, FloatMatrix*>>  ComputationNode<float>::s_constOnes{};
    template<> std::map<size_t, std::map<size_t, DoubleMatrix*>> ComputationNode<double>::s_constOnes{};

    template class LearnableParameter<float>;
    template class LearnableParameter<double>;
}}}

namespace Microsoft { namespace MSR { namespace ScriptableObjects {

    using namespace Microsoft::MSR::CNTK;

    // -----------------------------------------------------------------------
    // register ComputationNode with the ScriptableObject system
    // -----------------------------------------------------------------------

    template<> shared_ptr<Object> MakeRuntimeObject<ComputationNodeBase>(const IConfigRecordPtr configp)
    {
        return NewComputationNodeFromConfig(configp);
    }

    ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<ComputationNodeBase> registerComputationNode(L"ComputationNode");
}}}

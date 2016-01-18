//
// <copyright file="wrapper.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include <windows.h>
#include <vcclr.h>
#include <string>
#include <utility>
#include <msclr\marshal_cppstd.h>

#include "Eval.h"

#using <System.dll>
#using <System.Collections.dll>

using namespace System;
using namespace System::Collections::Generic;
using namespace System::Collections;

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK 
{
    template<typename ElemType>
    using GetEvalProc = void(*)(IEvaluateModel<ElemType>**);

    template<typename ElemType>
    public ref class IEvaluateModelManaged
    {
    public:
        IEvaluateModelManaged(String^ funcName)
        {
            pin_ptr<const WCHAR> dllname = PtrToStringChars("evaldll.dll");
            auto hModule = LoadLibrary(dllname);

            msclr::interop::marshal_context context;
            const std::string func = context.marshal_as<std::string>(funcName);
            auto procAddress = GetProcAddress(hModule, func.c_str());
            auto getEvalProc = (GetEvalProc<ElemType>)procAddress;
            pin_ptr <IEvaluateModel<ElemType>*> p_eval = &m_eval;
            getEvalProc(p_eval);

            //pin_ptr <IEvaluateModel<ElemType>*> p_eval = &m_eval;
            //GetEvalF(p_eval);
        }

        void Init(String^ config)
        {
            msclr::interop::marshal_context context;
            const std::string stdConfig = context.marshal_as<std::string>(config);

            m_eval->Init(stdConfig);
        }

        void Destroy()
        {
            m_eval->Destroy();
        }

        void LoadModel(String^ modelFileName)
        {
            pin_ptr<const WCHAR> stdModelPath = PtrToStringChars(modelFileName);
            m_eval->LoadModel(stdModelPath);
        }

        void Evaluate(Dictionary<String^, List<ElemType>^>^ inputs, Dictionary<String^, List<ElemType>^>^ outputs)
        {
            std::map<std::wstring, std::vector<ElemType>*> stdInputs;
            std::map<std::wstring, std::vector<ElemType>*> stdOutputs;
            std::pair<std::wstring, std::vector<ElemType>*>* stdInput;
            std::pair<std::wstring, std::vector<ElemType>*>* stdOutput;

            for each (auto item in inputs)
            {
                pin_ptr<const WCHAR> key = PtrToStringChars(item.Key);
                stdInput = new std::pair<std::wstring, std::vector<ElemType>*>(key, CopyList(item.Value));
                stdInputs.insert(*stdInput);
            }

            for each (auto item in outputs)
            {
                pin_ptr<const WCHAR> key = PtrToStringChars(item.Key);
                stdOutput = new std::pair<std::wstring, std::vector<ElemType>*>(key, CopyList(item.Value));
                stdOutputs.insert(*stdOutput);
            }
            
            m_eval->Evaluate(stdInputs, stdOutputs);

            auto enumerator = outputs->Keys->GetEnumerator();
            for (std::map<std::wstring, std::vector<ElemType>*>::iterator ii = stdOutputs.begin(), e = stdOutputs.end(); ii != e; ii++)
            {
                // All this, to get the key (output layer name)
                std::vector<ElemType>* vector = (*ii).second;
                std::vector<ElemType> &refVector = (*vector);
                int index = 0;
                enumerator.MoveNext();
                String^ key = enumerator.Current;

                // Copy output to CLI structure
                for (std::vector<ElemType>::iterator ii = refVector.begin(), e = refVector.end(); ii != e; ii++)
                {
                    outputs[key][index++] = *ii;
                }
            }

            // Release the used memory
            for (std::map<std::wstring, std::vector<ElemType>*>::iterator ii = stdInputs.begin(), e = stdInputs.end(); ii != e;)
            {
                delete (*ii).second;
                stdInputs.erase(ii++); // Doesn't seem to release the memory, thus the delete stdInput call
            }
            delete stdInput;

            for (std::map<std::wstring, std::vector<ElemType>*>::iterator ii = stdOutputs.begin(), e = stdOutputs.end(); ii != e;)
            {
                delete (*ii).second;
                stdOutputs.erase(ii++); // Doesn't seem to release the memory
            }
            delete stdOutput;
        }

        List<ElemType>^ Evaluate(Dictionary<String^, List<ElemType>^>^ inputs, String^ outputKey, int outputSize)
        {
            std::map<std::wstring, std::vector<ElemType>*> stdInputs;
            std::map<std::wstring, std::vector<ElemType>*> stdOutputs;
            std::pair<std::wstring, std::vector<ElemType>*>* stdInput;

            for each (auto item in inputs)
            {
                pin_ptr<const WCHAR> key = PtrToStringChars(item.Key);
                stdInput = new std::pair<std::wstring, std::vector<ElemType>*>(key, CopyList(item.Value));
                stdInputs.insert(*stdInput);
            }
            
            pin_ptr<const WCHAR> key = PtrToStringChars(outputKey);
            std::vector<ElemType> stdOutputVector(outputSize);
            std::pair<std::wstring, std::vector<ElemType>*> stdOutput(key, &stdOutputVector);
            stdOutputs.insert(stdOutput);

            m_eval->Evaluate(stdInputs, stdOutputs);
            
            std::vector<ElemType>* vector = stdOutputs.begin()->second;
            std::vector<ElemType> refVector = (*vector);

            List<ElemType>^ output = gcnew List<ElemType>();

            // Copy output to CLI structure
            for (std::vector<ElemType>::iterator ii = (*vector).begin(), e = (*vector).end(); ii != e; ii++)
            {
                output->Add(*ii);
            }
            
            // Release the used memory
            for (std::map<std::wstring, std::vector<ElemType>*>::iterator ii = stdInputs.begin(), e = stdInputs.end(); ii != e;)
            {
                delete (*ii).second;
                stdInputs.erase(ii++); // Doesn't seem to release the memory
            }
            delete stdInput;

            return output;
        }

    private:
        IEvaluateModel<ElemType> *m_eval;

        std::vector<ElemType>* CopyList(List<ElemType>^ list)
        {
            std::vector<ElemType>* lower = new std::vector<ElemType>();
            for each (auto item in list)
            {
                ElemType val = item;
                lower->push_back(val);
            }

            return lower;
        }
    };

    public ref class IEvaluateModelManagedF : IEvaluateModelManaged<float>
    {
    public:
        IEvaluateModelManagedF::IEvaluateModelManagedF()
            : IEvaluateModelManaged("GetEvalF")
        {
        }
    };
    
    public ref class IEvaluateModelManagedD : IEvaluateModelManaged<double>
    {
    public:
        IEvaluateModelManagedD::IEvaluateModelManagedD()
            : IEvaluateModelManaged("GetEvalD")
        {
        }
    };
    
    void emit()
    {
        // This method tricks the compiler into emitting the methods of the classes
        // Refer to https://msdn.microsoft.com/en-us/library/ms177213.aspx for an
        // explanation to this insanity
        IEvaluateModelManagedF f;
        f.Init("");
        f.Evaluate(nullptr, nullptr);
        f.Evaluate(nullptr, "", 0);
        f.LoadModel("");
        f.Destroy();
        
        
        IEvaluateModelManagedD d;
        d.Init("");
        d.Evaluate(nullptr, nullptr);
        d.Evaluate(nullptr, "", 0);
        d.LoadModel("");
        d.Destroy();
    }
}}}
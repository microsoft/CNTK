//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// PythonWrapper.cpp -- Wrapping the native EvaluateModel interface for Python
//

#include <memory>
#include <boost/python.hpp>
#include <Windows.h>

#include "../../Common/Include/Eval.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Extensibility { namespace Managed {

    // Used for retrieving the model appropriate for the element type (float / double)
    template<typename ElemType>
    using GetEvalProc = void(*)(IEvaluateModel<ElemType>**);

    /// Managed wrapper for the native evaluation model
    template<typename ElemType>
    class IPythonEvaluateModel
    {
        typedef std::pair<std::wstring, std::vector<ElemType>*> MapEntry;

    public:
        /// <summary>Initializes a new instance of the <see cref="IPythonEvaluateModel"> class.</summary>
        /// <param name="funcName">Factory function name for retrieving the native model from the dll.</param>
        IPythonEvaluateModel(const std::string& funcName)
        {
            auto hModule = LoadLibraryA("evaldll.dll");
            if (hModule == nullptr)
            {
                throw std::runtime_error("Cannot find library: evaldll.dll");
            }           

            auto procAddress = GetProcAddress(hModule, funcName.c_str());
            auto getEvalProc = (GetEvalProc<ElemType>)procAddress;
            getEvalProc(&m_eval);
            if (m_eval == nullptr)
            {
                throw std::runtime_error("Cannot get IEvaluateModel.");
            }

        }

        /// <summary>Initializes the model evaluation library with a CNTK configuration</summary>
        /// <param name="config">Model configuration entries</param>
        void Init(const std::string& config)
        {
            if (m_eval == nullptr)
            {
                throw std::runtime_error("Object has been disposed.");
            }

            m_eval->Init(config);
        }

        /// <summary>Creates a network based from the network description in the configuration</summary>
        /// <param name="networkDescription">The configuration file containing the network description</param>
        void CreateNetwork(const std::string& networkDescription)
        {
            if (m_eval == nullptr)
            {
                throw std::runtime_error("Object has been disposed.");
            }
                
            m_eval->CreateNetwork(networkDescription);
        }

        /// <summary>Evaluates the model using a single forward feed pass and retrieves the output layer data</summary>
        /// <param name="outputKey"></param>
        /// <param name="outputSize"></param>
        /// <returns>Results for specified layer</returns>
        boost::python::list Evaluate(const std::wstring& outputKey, int outputSize)
        {
            if (m_eval == nullptr)
            {
                throw std::runtime_error("Object has been disposed.");
            }

            std::map<std::wstring, std::vector<ElemType>* > stdOutputs;

            std::shared_ptr<std::vector<ElemType> > pOutputVector(new std::vector<ElemType>());

            pOutputVector->resize(outputSize);
            
            stdOutputs[outputKey]=pOutputVector.get();

            m_eval->Evaluate(stdOutputs);
                
            boost::python::list ret;
            
            for (std::vector<ElemType>::iterator itr = pOutputVector->begin(); itr != pOutputVector->end(); ++itr)
            {
                ret.append(*itr);
            }

            return ret;
                        
        }

        

        /// <summary>Evaluates the model against input data and retrieves the output layer data</summary>
        /// <param name="inputs"></param>
        /// <param name="outputKey"></param>
        /// <param name="outputSize"></param>
        /// <returns>Results for specified layer</returns>
        boost::python::list Evaluate2(const boost::python::dict& inputs, const std::wstring& outputKey, int outputSize)
        {
            if (m_eval == nullptr)
            {
                throw std::runtime_error("Object has been disposed.");
            }

            std::map<std::wstring, std::vector<ElemType>* > stdInputs;
            std::vector<std::shared_ptr<std::vector<ElemType> >> resourceManagement;
            for (int i = 0; i < boost::python::len(inputs); ++i)
            {
                std::shared_ptr<std::vector<ElemType> > pInputVector(new std::vector<ElemType>());
                resourceManagement.push_back(pInputVector);
                boost::python::list theValues = boost::python::extract<boost::python::list>(inputs.values()[i]);
                for (int j = 0; j < boost::python::len(theValues); ++j)
                {
                    pInputVector->push_back(boost::python::extract<ElemType>(theValues[j]));
                }

                std::wstring key = boost::python::extract<std::wstring>(inputs.keys()[i]);
                stdInputs[key] = pInputVector.get();
            }
           

            std::map<std::wstring, std::vector<ElemType>* > stdOutputs;

            std::shared_ptr<std::vector<ElemType> > pOutputVector(new std::vector<ElemType>());

            pOutputVector->resize(outputSize);

            stdOutputs[outputKey]=pOutputVector.get();

            m_eval->Evaluate(stdInputs, stdOutputs);

            boost::python::list ret;

            for (std::vector<ElemType>::iterator itr = pOutputVector->begin(); itr != pOutputVector->end(); ++itr)
            {
                ret.append(*itr);
            }

            return ret;
        }

        std::string TEST()
        {
            return "TEST";
        }

#if 0 //Not IMPLEMENT

        /// <summary>Evaluates the model against input data and retrieves the output layer data</summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        void Evaluate3(const boost::python::dict& inputs, boost::python::dict& outputs)
        {
            if (m_eval == nullptr)
            {
                throw std::runtime_error("Object has been disposed.");
            }

            throw std::runtime_error("Not implemented.");
        }
#endif

        ~IPythonEvaluateModel()
        {
            if (m_eval != nullptr)
            {
                m_eval->Destroy();
                m_eval = nullptr;
            }
        }

    private:
        // Native model evaluation instance
        IEvaluateModel<ElemType> *m_eval;

    };

    /// <summary>Managed float-specific model evaluation class</summary>
    /// <remarks>This class is necessary due to how generics and templates work in CLR</remarks>
    class IPythonEvaluateModelF : public IPythonEvaluateModel<float>
    {
    public:
        IPythonEvaluateModelF()
            : IPythonEvaluateModel("GetEvalF")
        {
        }
    };

    /// <summary>Managed double-specific model evaluation class</summary>
    /// <remarks>This class is necessary due to how generics and templates work in CLR</remarks>
    class IPythonEvaluateModelD : public IPythonEvaluateModel<double>
    {
    public:
        IPythonEvaluateModelD()
            : IPythonEvaluateModel("GetEvalD")
        {
        }
    };

}}}}}

BOOST_PYTHON_MODULE(CNTKPythonWrapper)
{
    using namespace boost::python;
    using namespace Microsoft::MSR::CNTK::Extensibility::Managed;

    class_<IPythonEvaluateModelD>("IPythonEvaluateModelD", init<>())
        .def("Init", &IPythonEvaluateModelD::Init)
        .def("CreateNetwork", &IPythonEvaluateModelD::CreateNetwork)
        .def("Evaluate", &IPythonEvaluateModelD::Evaluate)
        .def("Evaluate2", &IPythonEvaluateModelD::Evaluate2)
        .def("TEST", &IPythonEvaluateModelD::TEST);

    class_<IPythonEvaluateModelF>("IPythonEvaluateModelF", init<>())
        .def("Init", &IPythonEvaluateModelF::Init)
        .def("CreateNetwork", &IPythonEvaluateModelF::CreateNetwork)
        .def("Evaluate", &IPythonEvaluateModelF::Evaluate)
        .def("Evaluate2", &IPythonEvaluateModelF::Evaluate2)
        .def("TEST", &IPythonEvaluateModelF::TEST);
    
}

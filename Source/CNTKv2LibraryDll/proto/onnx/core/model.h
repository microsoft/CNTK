#ifndef CORE_GRAPH_MODEL_H
#define CORE_GRAPH_MODEL_H

#include "graph.h"

namespace ONNXIR
{
    typedef std::unordered_map<std::string, std::string> ModelMetaData;

    // A machine learning model representation class.
    // Besides a main <Graph>, it also holds basic information, say,
    // version, domain, model author, license etc.
    class Model
    {
    public:

        const VERSION c_noVersion = INT64_MAX;

        Model(const std::string& p_graphName,
            bool p_isONNX = false,
            const ModelMetaData& p_modelMetaData = ModelMetaData());

        Model(const std::string& p_graphName,
            const std::string& p_graphDocString,
            const std::string& p_producerName,
            const std::string& p_producerVersion,
            const std::string& p_domain,
            VERSION p_modelVersion,
            const std::string& p_modelDocString,
            const ModelMetaData& p_modelMetaData = ModelMetaData());

        Model(const ModelProto& p_modelProto);

        // Get model's IR version.
        // Return <c_noVersion> if not specified.
        VERSION IrVersion() const;

        // Get model's producer name.
        // Return null pointer if not specified.
        const std::string& ProducerName() const;
        // Set model's producer name.
        void SetProducerName(const std::string& p_producerName);

        // Get model's producer version.
        // Return null pointer if not specified.
        const std::string& ProducerVersion() const;
        // Set model's producer version.
        void SetProducerVersion(const std::string& p_producerVersion);

        // Get model's domain.
        // Return null pointer if not specified.
        const std::string& Domain() const;
        // Set models' damain.
        void SetDomain(const std::string& p_domain);

        // Get model's version.
        // Return null pointer if not specified.
        VERSION ModelVersion() const;
        // Set models' version.
        void SetModelversion(VERSION p_modelVersion);

        // Get model's doc string.
        // Return null pointer if not specified.
        const std::string& DocString() const;
        // Set models' doc string.
        void SetDocString(const std::string& p_docString);

        const ModelMetaData& MetaData() const;

        // Get model's main graph.
        // The return pointer is owned by <*this> model.
        Graph* MainGraph();
        const Graph* MainGraph() const;

        // Get model's serlization proto data.
        const ModelProto& ToProto();

#ifdef _WIN32
        static Status Save(Model& p_model, const std::wstring& p_filePath);

        static Status Load(const std::wstring& p_filePath, /*out*/ std::shared_ptr<Model>* p_model);
#endif
        static Status Save(Model& p_model, const std::string& p_filePath);

        static Status Save(Model& p_model, int p_fd);

        static Status Load(const std::string& p_filePath, /*out*/ std::shared_ptr<Model>* p_model);

        static Status Load(int p_fd, /*out*/ std::shared_ptr<Model>* p_model);

        // 'int' rather than 'size_t' because of a protobuf design choice; let callers handle type checks
        static Status LoadFromBytes(int count, void *pBytes, /*out*/ std::shared_ptr<Model>* p_model);

    private:

        // Model data.
        ModelProto m_modelProto;
        // This is a duplication of <m_modelProto.metadata_props()>.
        // It gives better accessibility.
        ModelMetaData m_modelMetaData;

        // Main graph of the model.
        std::unique_ptr<Graph> m_graph;
    };
}

#endif

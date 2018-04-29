#ifdef _MSC_VER
#pragma warning(push)
// 'type' : forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable: 4800)
#endif
#include <google/protobuf/io/coded_stream.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <google/protobuf/io/zero_copy_stream_impl.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include "model.h"
#include "utils.h"

namespace ONNXIR
{
    Model::Model(const std::string& p_graphName,
        bool p_isONNX,
        const ModelMetaData& p_modelMetaData)
    {
        m_modelProto.reset(new ModelProto);
        m_modelProto->set_ir_version(Version::IR_VERSION);
        m_modelProto->mutable_graph()->set_name(p_graphName);
        m_modelMetaData = p_modelMetaData;
        for (auto& metaData : m_modelMetaData)
        {
            auto prop = m_modelProto->add_metadata_props();
            prop->set_key(metaData.first);
            prop->set_value(metaData.second);
        }
        // Set m_domainToVersion to contain related domains with latest version.
        AddImportOpSets(p_isONNX);
        m_graph.reset(new Graph(m_modelProto->mutable_graph(), m_domainToVersion, p_isONNX));
    }

    Model::Model(const ModelProto& p_modelProto)
        : Model(std::unique_ptr<ModelProto>(new ModelProto(p_modelProto)))
    {
    }

    Model::Model(std::unique_ptr<ModelProto> p_modelProto)
    {
        assert(nullptr != p_modelProto);
        m_modelProto.reset(p_modelProto.release());
        for (auto& prop : m_modelProto->metadata_props())
        {
            m_modelMetaData[prop.key()] = prop.value();
        }

        if (0 == m_modelProto->opset_import_size())
        {
            // Operator sets are not specified in this model.
            // Will use global operator store instead.
            AddImportOpSets(false);
        }
        else
        {
            for (auto& opSet : m_modelProto->opset_import())
            {
                m_domainToVersion[opSet.domain()] = static_cast<int>(opSet.version());
            }
        }

        if (m_modelProto->has_graph())
        {
            m_graph.reset(new Graph(m_modelProto->mutable_graph(), m_domainToVersion));
        }
    }

    VERSION Model::IrVersion() const
    {
        if (m_modelProto->has_ir_version())
        {
            return m_modelProto->ir_version();
        }
        return c_noVersion;
    }

    const std::string& Model::ProducerName() const
    {
        return m_modelProto->producer_name();
    }

    void Model::SetProducerName(const std::string& p_producerName)
    {
        m_modelProto->set_producer_name(p_producerName);
    }

    const std::string& Model::ProducerVersion() const
    {
        return m_modelProto->producer_version();
    }

    void Model::SetProducerVersion(const std::string& p_producerVersion)
    {
        m_modelProto->set_producer_version(p_producerVersion);
    }

    const std::string& Model::Domain() const
    {
        return m_modelProto->domain();
    }

    void Model::SetDomain(const std::string& p_domain)
    {
        m_modelProto->set_domain(p_domain);
    }

    VERSION Model::ModelVersion() const
    {
        if (m_modelProto->has_model_version())
        {
            return m_modelProto->model_version();
        }
        return c_noVersion;
    }

    void Model::SetModelversion(VERSION p_modelVersion)
    {
        m_modelProto->set_model_version(p_modelVersion);
    }

    const std::string& Model::DocString() const
    {
        return m_modelProto->doc_string();
    }

    void Model::SetDocString(const std::string& p_docString)
    {
        m_modelProto->set_doc_string(p_docString);
    }

    const ModelMetaData& Model::MetaData() const
    {
        return m_modelMetaData;
    }

    Graph* Model::MainGraph()
    {
        return m_graph.get();
    }

    const Graph* Model::MainGraph() const
    {
        return m_graph.get();
    }

    ModelProto Model::ToProto()
    {
        *(m_modelProto->mutable_graph()) = m_graph->ToGraphProto();
        return *m_modelProto;
    }

    void Model::AddImportOpSets(bool p_isONNX)
    {
        auto& domainToVersionRangeMap = OpSchemaRegistry::DomainToVersionRange::Instance().Map();
        for (auto& domainToVersionRange : domainToVersionRangeMap)
        {
            if (p_isONNX && domainToVersionRange.first.compare(c_onnxDomain) != 0)
            {
                // Constructing a pure ONNX model.
                // Only ops in ONNX domain should be used.
                continue;
            }

            m_domainToVersion[domainToVersionRange.first] = domainToVersionRange.second.second;
            auto opSetIdProto = m_modelProto->add_opset_import();
            opSetIdProto->set_domain(domainToVersionRange.first);
            opSetIdProto->set_version(domainToVersionRange.second.second);
        }
    }

#ifdef _WIN32
    Status Model::Load(const std::wstring& p_filePath, std::shared_ptr<Model>* p_model)
    {
        int fd;
        RETURN_IF_ERROR(FileOpenRd(p_filePath, &fd));
        auto status = Load(fd, p_model);
        RETURN_IF_ERROR(FileClose(fd));
        return status;
    }

    Status Model::Save(Model& p_model, const std::wstring& p_filePath)
    {
        int fd;
        RETURN_IF_ERROR(FileOpenWr(p_filePath, &fd));
        auto status = Save(p_model, fd);
        RETURN_IF_ERROR(FileClose(fd));
        return status;
    }

#endif

    Status Model::Load(const std::string& p_filePath, std::shared_ptr<Model>* p_model)
    {
        int fd;
        RETURN_IF_ERROR(FileOpenRd(p_filePath, &fd));
        auto status = Load(fd, p_model);
        RETURN_IF_ERROR(FileClose(fd));
        return status;
    }

    Status Model::LoadFromBytes(int count, void *pBytes, /*out*/ std::shared_ptr<Model>* p_model)
    {
        std::unique_ptr<ModelProto> modelProto(new ModelProto);
        bool result = modelProto->ParseFromArray(pBytes, count);
        if (!result)
        {
            return Status(ONNX, INVALID_PROTOBUF, "Protobuf parsing failed.");
        }

        (*p_model).reset(new Model(std::move(modelProto)));
        if ((*p_model)->MainGraph() != nullptr)
        {
            RETURN_IF_ERROR((*p_model)->MainGraph()->Resolve());
        }
        return Status::OK();
    }

    Status Model::Save(Model& p_model, const std::string& p_filePath)
    {
        int fd;
        RETURN_IF_ERROR(FileOpenWr(p_filePath, &fd));
        auto status = Save(p_model, fd);
        RETURN_IF_ERROR(FileClose(fd));
        return status;
    }

    using ::google::protobuf::io::ZeroCopyInputStream;
    using ::google::protobuf::io::FileInputStream;
    using ::google::protobuf::io::CodedInputStream;

    Status Model::Load(int p_fd, std::shared_ptr<Model>* p_model)
    {
        if (p_fd < 0 || nullptr == p_model)
        {
            return Status(ONNX, INVALID_ARGUMENT, "<p_fd> less than 0 or <p_model> is nullptr.");
        }

        std::unique_ptr<ZeroCopyInputStream> raw_input(new FileInputStream(p_fd));
        std::unique_ptr<CodedInputStream> coded_input(
            new CodedInputStream(raw_input.get()));
        // Allows protobuf library versions < 3.2.0 to parse messages greater than 64MB.
        coded_input->SetTotalBytesLimit(INT_MAX, INT_MAX);
        std::unique_ptr<ModelProto> modelProto(new ModelProto);
        bool result = modelProto->ParseFromCodedStream(coded_input.get());
        coded_input.reset();
        raw_input.reset();
        if (!result)
        {
            return Status(ONNX, INVALID_PROTOBUF, "Protobuf parsing failed.");
        }

        (*p_model).reset(new Model(std::move(modelProto)));
        if ((*p_model)->MainGraph() != nullptr)
        {
            RETURN_IF_ERROR((*p_model)->MainGraph()->Resolve());
        }
        return Status::OK();
    }

    Status Model::Save(Model& p_model, int p_fd)
    {
        if (p_fd < 0)
        {
            return Status(ONNX, INVALID_ARGUMENT, "<p_fd> is less than 0.");
        }

        RETURN_IF_ERROR(p_model.MainGraph()->Resolve());
        auto modelProto = p_model.ToProto();
        bool result = modelProto.SerializeToFileDescriptor(p_fd);
        if (result)
        {
            return Status::OK();
        }
        else
        {
            return Status(ONNX, INVALID_PROTOBUF, "Protobuf serialization failed.");
        }
    }
}

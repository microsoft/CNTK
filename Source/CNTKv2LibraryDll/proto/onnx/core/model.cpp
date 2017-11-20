#include <fcntl.h>
#include <fstream>
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
#include <sys/io.h>
#include <unistd.h>
#endif
#include "model.h"

namespace
{
#ifdef _WIN32
    inline Status FileOpenRd(const std::wstring& p_path, /*out*/ int* p_fd)
    {
        _wsopen_s(p_fd, p_path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
        if (0 > *p_fd)
        {
            return Status(SYSTEM, errno);
        }
        return Status::OK();
    }

    inline Status FileOpenWr(const std::wstring& p_path, /*out*/ int* p_fd)
    {
        _wsopen_s(p_fd, p_path.c_str(), _O_CREAT | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
        if (0 > *p_fd)
        {
            return Status(SYSTEM, errno);
        }
        return Status::OK();
    }
#endif

    inline Status FileOpenRd(const std::string& p_path, /*out*/ int* p_fd)
    {
#ifdef _WIN32
        _sopen_s(p_fd, p_path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
#else
        *p_fd = open(p_path.c_str(), O_RDONLY);
#endif
        if (0 > *p_fd)
        {
            return Status(SYSTEM, errno);
        }
        return Status::OK();
    }

    inline Status FileOpenWr(const std::string& p_path, /*out*/ int* p_fd)
    {
#ifdef _WIN32
        _sopen_s(p_fd, p_path.c_str(), _O_CREAT | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
#else
        *p_fd = open(p_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
#endif
        if (0 > *p_fd)
        {
            return Status(SYSTEM, errno);
        }
        return Status::OK();
    }

    inline Status FileClose(int fd)
    {
        int ret = 0;
#ifdef _WIN32
        ret = _close(fd);
#else
        ret = close(fd);
#endif
        if (0 != ret)
        {
            return Status(SYSTEM, errno);
        }
        return Status::OK();
    }
}

namespace ONNXIR
{
    Model::Model(const std::string& p_graphName, bool p_isONNX)
    {
        m_graph.reset(new Graph(p_graphName, p_isONNX));
    }

    Model::Model(const std::string& p_graphName,
        const std::string& p_graphDocString)
    {
        m_graph.reset(new Graph(p_graphName, p_graphDocString));
    }

    Model::Model(const std::string& p_graphName,
        const std::string& p_graphDocString,
        VERSION p_irVersion,
        const std::string& p_producerName,
        const std::string& p_producerVersion,
        const std::string& p_domain,
        VERSION p_modelVersion,
        const std::string& p_docString)
    {
        m_graph.reset(new Graph(p_graphName, p_graphDocString));
        m_modelProto.set_ir_version(p_irVersion);
        m_modelProto.set_producer_name(p_producerName);
        m_modelProto.set_producer_version(p_producerVersion);
        m_modelProto.set_domain(p_domain);
        m_modelProto.set_model_version(p_modelVersion);
        m_modelProto.set_doc_string(p_docString);
    }

    Model::Model(const ModelProto& p_modelProto)
    {
        m_modelProto = p_modelProto;
        if (m_modelProto.has_graph())
        {
            m_graph.reset(new Graph(m_modelProto.graph()));
        }
    }

    VERSION Model::IrVersion() const
    {
        if (m_modelProto.has_ir_version())
        {
            return m_modelProto.ir_version();
        }
        return c_noVersion;
    }

    void Model::SetIrVersion(VERSION p_irVersion)
    {
        m_modelProto.set_ir_version(p_irVersion);
    }

    const std::string& Model::ProducerName() const
    {
        return m_modelProto.producer_name();
    }

    void Model::SetProducerName(const std::string& p_producerName)
    {
        m_modelProto.set_producer_name(p_producerName);
    }

    const std::string& Model::ProducerVersion() const
    {
        return m_modelProto.producer_version();
    }

    void Model::SetProducerVersion(const std::string& p_producerVersion)
    {
        m_modelProto.set_producer_version(p_producerVersion);
    }

    const std::string& Model::Domain() const
    {
        return m_modelProto.domain();
    }

    void Model::SetDomain(const std::string& p_domain)
    {
        m_modelProto.set_domain(p_domain);
    }

    VERSION Model::ModelVersion() const
    {
        if (m_modelProto.has_model_version())
        {
            return m_modelProto.model_version();
        }
        return c_noVersion;
    }

    void Model::SetModelversion(VERSION p_modelVersion)
    {
        m_modelProto.set_model_version(p_modelVersion);
    }

    const std::string& Model::DocString() const
    {
        return m_modelProto.doc_string();
    }

    void Model::SetDocString(const std::string& p_docString)
    {
        m_modelProto.set_doc_string(p_docString);
    }

    Graph* Model::MainGraph()
    {
        return m_graph.get();
    }

    const Graph* Model::MainGraph() const
    {
        return m_graph.get();
    }

    const ModelProto& Model::ToProto()
    {
        *(m_modelProto.mutable_graph()) = m_graph->ToGraphProto();
        return m_modelProto;
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
        ModelProto modelProto;
        bool result = modelProto.ParseFromCodedStream(coded_input.get());
        coded_input.reset();
        raw_input.reset();
        if (!result)
        {
            return Status(ONNX, INVALID_PROTOBUF, "Protobuf parsing failed.");
        }

        (*p_model).reset(new Model(modelProto));
        RETURN_IF_ERROR((*p_model)->MainGraph()->Resolve());

        return Status::OK();
    }

    Status Model::Save(Model& p_model, int p_fd)
    {
        if (p_fd < 0)
        {
            return Status(ONNX, INVALID_ARGUMENT, "<p_fd> is less than 0.");
        }

        RETURN_IF_ERROR(p_model.MainGraph()->Resolve());
        auto& modelProto = p_model.ToProto();
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

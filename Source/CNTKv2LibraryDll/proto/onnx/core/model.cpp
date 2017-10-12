#pragma warning(push)
#pragma warning(disable : 4800 4610 4512 4510 4267 4127 4125 4100 4456 4189 4996)

#include <fcntl.h>
#include <fstream>
#include <google/protobuf/io/coded_stream.h>
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
    inline int FileOpenRd(const std::wstring& p_path)
    {
        int fd = -1;
        bool err = _wsopen_s(&fd, p_path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
        return fd;
    }

    inline int FileOpenWr(const std::wstring& p_path)
    {
        int fd = -1;
        _wsopen_s(&fd, p_path.c_str(), _O_CREAT | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
        return fd;
    }
#endif

    inline int FileOpenRd(const std::string& p_path)
    {
#ifdef _WIN32
        int fd = -1;
        _sopen_s(&fd, p_path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
        return fd;
#else
        return open(p_path.c_str(), O_RDONLY);
#endif
    }

    inline int FileOpenWr(const std::string& p_path)
    {
#ifdef _WIN32
        int fd = -1;
        _sopen_s(&fd, p_path.c_str(), _O_CREAT | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
        return fd;
#else
        return open(p_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
#endif
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
        const std::string& p_docString,
        const std::string& p_modelAuthor,
        const std::string& p_modelLicense)
    {
        m_graph.reset(new Graph(p_graphName, p_graphDocString));
        m_modelProto.set_ir_version(p_irVersion);
        m_modelProto.set_producer_name(p_producerName);
        m_modelProto.set_producer_version(p_producerVersion);
        m_modelProto.set_domain(p_domain);
        m_modelProto.set_model_version(p_modelVersion);
        m_modelProto.set_doc_string(p_docString);
        m_modelProto.set_model_author(p_modelAuthor);
        m_modelProto.set_model_license(p_modelLicense);
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

    const std::string& Model::ModelAuthor() const
    {
        return m_modelProto.model_author();
    }

    void Model::SetModelAuthor(const std::string& p_modelAuthor)
    {
        m_modelProto.set_model_author(p_modelAuthor);
    }

    const std::string& Model::ModelLicense() const
    {
        return m_modelProto.model_license();
    }

    void Model::SetModelLicense(const std::string& p_modelLicense)
    {
        m_modelProto.set_model_license(p_modelLicense);
    }

    Graph* Model::MainGraph()
    {
        return m_graph.get();
    }

    const ModelProto& Model::ToProto()
    {
        *(m_modelProto.mutable_graph()) = m_graph->ToGraphProto();
        return m_modelProto;
    }

#ifdef _WIN32
    bool Model::Load(const std::wstring& p_filePath, /*out*/ ModelProto* p_modelProto)
    {
        return Load(FileOpenRd(p_filePath), p_modelProto);
    }
    std::shared_ptr<Model> Model::Load(const std::wstring& p_filePath)
    {
        return Load(FileOpenRd(p_filePath));
    }
    bool Model::Save(Model& p_model, const std::wstring& p_filePath)
    {
        return Save(p_model.ToProto(), FileOpenWr(p_filePath));
    }
    bool Model::Save(const ModelProto& p_modelProto, const std::wstring& p_filePath)
    {
        return Save(p_modelProto, FileOpenWr(p_filePath));
    }
#endif

    bool Model::Load(const std::string& p_filePath, /*out*/ ModelProto* p_modelProto)
    {
        return Load(FileOpenRd(p_filePath), p_modelProto);
    }
    std::shared_ptr<Model> Model::Load(const std::string& p_filePath)
    {
        return Load(FileOpenRd(p_filePath));
    }
    bool Model::Save(Model& p_model, const std::string& p_filePath)
    {
        return Save(p_model.ToProto(), FileOpenWr(p_filePath));
    }
    bool Model::Save(const ModelProto& p_modelProto, const std::string& p_filePath)
    {
        return Save(p_modelProto, FileOpenWr(p_filePath));
    }

    using ::google::protobuf::io::ZeroCopyInputStream;
    using ::google::protobuf::io::FileInputStream;
    using ::google::protobuf::io::CodedInputStream;
    bool Model::Load(int p_fd, /*out*/ ModelProto* p_modelProto)
    {
        if (nullptr == p_modelProto || p_fd < 0)
        {
            return false;
        }
        std::unique_ptr<ZeroCopyInputStream> raw_input(new FileInputStream(p_fd));
        std::unique_ptr<CodedInputStream> coded_input(
            new CodedInputStream(raw_input.get()));
        // Allows protobuf library versions < 3.2.0 to parse messages greater than 64MB.
        coded_input->SetTotalBytesLimit(INT_MAX, INT_MAX);
        bool result = p_modelProto->ParseFromCodedStream(coded_input.get());
        coded_input.reset();
        raw_input.reset();
        close(p_fd);
        return result;
    }

    std::shared_ptr<Model> Model::Load(int p_fd)
    {
        ModelProto modelProto;
        bool result = Load(p_fd, &modelProto);
        if (!result || p_fd < 0)
        {
            return nullptr;
        }
        auto model = std::shared_ptr<Model>(new Model(modelProto));
        auto status = model->MainGraph()->Resolve();

        close(p_fd);
        if (status.Ok())
        {
            return model;
        }
        return nullptr;
    }

    bool Model::Save(const ModelProto& p_modelProto, int p_fd)
    {
        if (p_fd < 0)
        {
            return false;
        }
        bool result = p_modelProto.SerializeToFileDescriptor(p_fd);
        close(p_fd);
        return result;
    }
}

#pragma warning(pop)
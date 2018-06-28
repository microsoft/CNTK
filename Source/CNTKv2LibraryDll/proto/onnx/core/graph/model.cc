#ifdef _MSC_VER
#pragma warning(push)
// 'type' : forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable : 4800)
#endif
#include <google/protobuf/io/coded_stream.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "core/common/CommonSTD.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <memory>
#include "core/graph/model.h"
#include "core/graph/utils.h"
#include "core/graph/schema_registry.h"
// #include "gsl/pointers"
#include "gsl/gsl_util"

using namespace Lotus;
using namespace Lotus::Common;

namespace LotusIR {
Model::Model(const std::string& graph_name, bool is_onnx_domain_only, const ModelMetaData& model_metadata, const ILotusOpSchemaCollection* local_registry) {
  model_proto_ = std::make_unique<ModelProto>();
  model_proto_->set_ir_version(onnx::Version::IR_VERSION);
  model_proto_->mutable_graph()->set_name(graph_name);
  model_metadata_ = model_metadata;
  for (auto& metadata : model_metadata_) {
    auto prop = model_proto_->add_metadata_props();
    prop->set_key(metadata.first);
    prop->set_value(metadata.second);
  }

  // Set domain_to_version_ to contain related domains with latest version.
  std::unordered_map<std::string, int> domain_to_version;
  AddImportOpSets(is_onnx_domain_only, &domain_to_version, local_registry);

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)
  graph_.reset(new Graph(model_proto_->mutable_graph(), domain_to_version, IrVersion(), local_registry));
}

Model::Model(const ModelProto& model_proto, const ILotusOpSchemaCollection* local_registry)
    : Model(std::make_unique<ModelProto>(model_proto), local_registry) {
}

Model::Model(std::unique_ptr<ModelProto> model_proto, const ILotusOpSchemaCollection* local_registry) {
  assert(nullptr != model_proto);
  model_proto_.reset(model_proto.release());
  for (auto& prop : model_proto_->metadata_props()) {
    model_metadata_[prop.key()] = prop.value();
  }

  std::unordered_map<std::string, int> domain_to_version;
  if (0 == model_proto_->opset_import_size()) {
    // Operator sets are not specified in this model.
    // Will use global operator store instead.
    AddImportOpSets(false, &domain_to_version, local_registry);
  } else {
    for (auto& opSet : model_proto_->opset_import()) {
      domain_to_version[opSet.domain()] = gsl::narrow_cast<int>(opSet.version());
    }
  }

  if (model_proto_->has_graph()) {
    // create instance. need to call private ctor so can't use make_unique
    GSL_SUPPRESS(r .11)
    graph_.reset(new Graph(model_proto_->mutable_graph(), domain_to_version, IrVersion(), local_registry));
  }
}

Version Model::IrVersion() const {
  if (model_proto_->has_ir_version()) {
    return model_proto_->ir_version();
  }
  return kNoVersion;
}

const std::string& Model::ProducerName() const {
  return model_proto_->producer_name();
}

void Model::SetProducerName(const std::string& producer_name) {
  model_proto_->set_producer_name(producer_name);
}

const std::string& Model::ProducerVersion() const {
  return model_proto_->producer_version();
}

void Model::SetProducerVersion(const std::string& producer_version) {
  model_proto_->set_producer_version(producer_version);
}

const std::string& Model::Domain() const {
  return model_proto_->domain();
}

void Model::SetDomain(const std::string& domain) {
  model_proto_->set_domain(domain);
}

Version Model::ModelVersion() const {
  if (model_proto_->has_model_version()) {
    return model_proto_->model_version();
  }
  return kNoVersion;
}

void Model::SetModelversion(LotusIR::Version version) {
  model_proto_->set_model_version(version);
}

const std::string& Model::DocString() const {
  return model_proto_->doc_string();
}

void Model::SetDocString(const std::string& doc_string) {
  model_proto_->set_doc_string(doc_string);
}

const ModelMetaData& Model::MetaData() const noexcept {
  return model_metadata_;
}

Graph* Model::MainGraph() noexcept {
  return graph_.get();
}

const Graph* Model::MainGraph() const noexcept {
  return graph_.get();
}

ModelProto Model::ToProto() {
  *(model_proto_->mutable_graph()) = graph_->ToGraphProto();
  return *model_proto_;
}

void Model::AddImportOpSets(bool is_onnx_domain_only,
                            /*out*/ std::unordered_map<std::string, int>* domain_to_version,
                            const ILotusOpSchemaCollection* local_registry) {
  auto& domain_to_version_range_map = OpSchemaRegistry::DomainToVersionRange::Instance().Map();
  Domain_To_Version_Map local_domain_to_version_map = local_registry ? local_registry->DomainToVersionMap() : Domain_To_Version_Map();
  for (auto& domainToVersionRange : domain_to_version_range_map) {
    if (is_onnx_domain_only && domainToVersionRange.first.compare(kOnnxDomain) != 0) {
      // Constructing an onnx-domain-only model.
      // Only ops in ONNX domain should be used.
      continue;
    }

    int max = domainToVersionRange.second.second;
    //merge with local domain versions
    if (local_registry) {
      auto it = local_domain_to_version_map.find(domainToVersionRange.first);
      if (it != local_domain_to_version_map.end())
        max = std::max(it->second.second, max);
    }

    auto ignored = domain_to_version->insert({domainToVersionRange.first, max});
    auto opset_id_proto = model_proto_->add_opset_import();
    opset_id_proto->set_domain(domainToVersionRange.first);
    opset_id_proto->set_version(domainToVersionRange.second.second);
  }

  //merge the local domain versions
  if (local_registry) {
    for (auto& local_domain : local_domain_to_version_map) {
      if (is_onnx_domain_only && local_domain.first.compare(kOnnxDomain) != 0) {
        // Constructing an onnx-domain-only model.
        // Only ops in ONNX domain should be used.
        continue;
      }

      if (domain_to_version_range_map.end() != domain_to_version_range_map.find(local_domain.first)) {
        auto ignored = domain_to_version->insert({local_domain.first, local_domain.second.second});
        auto opset_id_proto = model_proto_->add_opset_import();
        opset_id_proto->set_domain(local_domain.first);
        opset_id_proto->set_version(local_domain.second.second);
      }
    }
  }
}

Status Model::Load(std::istream& model_istream, ModelProto* p_model_proto) {
  if (!model_istream.good()) {
    return Status(LOTUS, INVALID_ARGUMENT, "Invalid istream object.");
  }
  if (!p_model_proto) {
    return Status(LOTUS, INVALID_ARGUMENT, "Null model_proto ptr.");
  }
  const bool result = p_model_proto->ParseFromIstream(&model_istream);
  if (!result) {
    return Status(LOTUS, INVALID_PROTOBUF, "Failed to load model because protobuf parsing failed.");
  }
  return Status::OK();
}

Status Model::Load(const ModelProto& model_proto, std::shared_ptr<Model>& model, const ILotusOpSchemaCollection* local_registry) {
  // we expect a graph to be present
  if (!model_proto.has_graph()) {
    return Status(LOTUS, INVALID_ARGUMENT, "No graph was found in the protobuf.");
  }

  // need to call private ctor so can't use make_shared
  GSL_SUPPRESS(r .11)
  model.reset(new Model(model_proto, local_registry));

  if (model->MainGraph() != nullptr) {
    LOTUS_RETURN_IF_ERROR(model->MainGraph()->Resolve(true));
  }
  return Status::OK();
}

#ifdef _WIN32
GSL_SUPPRESS(r .30)  // spurious warnings. p_model is potentially reset in the internal call to Load
GSL_SUPPRESS(r .35)
Status Model::Load(const std::wstring& file_path, std::shared_ptr<Model>& p_model, const ILotusOpSchemaCollection* local_registry) {
  int fd;
  LOTUS_RETURN_IF_ERROR(FileOpenRd(file_path, &fd));
  auto status = Load(fd, p_model, local_registry);
  LOTUS_RETURN_IF_ERROR(FileClose(fd));
  return status;
}

Status Model::Save(Model& model, const std::wstring& file_path) {
  int fd;
  LOTUS_RETURN_IF_ERROR(FileOpenWr(file_path, &fd));
  auto status = Save(model, fd);
  LOTUS_RETURN_IF_ERROR(FileClose(fd));
  return status;
}

#endif

GSL_SUPPRESS(r .30)  // spurious warnings. p_model is potentially reset in the internal call to Load
GSL_SUPPRESS(r .35)
Status Model::Load(const std::string& file_path, std::shared_ptr<Model>& p_model, const ILotusOpSchemaCollection* local_registry) {
  int fd;
  if (!FileOpenRd(file_path, &fd).IsOK()) {
    return Status(LOTUS, NO_MODEL, "Failed to open: " + file_path);
  }
  auto status = Load(fd, p_model, local_registry);
  LOTUS_RETURN_IF_ERROR(FileClose(fd));
  return status;
}

Status Model::LoadFromBytes(int count, void* p_bytes, /*out*/ std::shared_ptr<Model>& p_model, const ILotusOpSchemaCollection* local_registry) {
  std::unique_ptr<ModelProto> modelProto = std::make_unique<ModelProto>();
  const bool result = modelProto->ParseFromArray(p_bytes, count);
  if (!result) {
    return Status(LOTUS, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }

  p_model = std::make_shared<Model>(std::move(modelProto), local_registry);
  if (p_model->MainGraph() != nullptr) {
    LOTUS_RETURN_IF_ERROR(p_model->MainGraph()->Resolve(true));
  }
  return Status::OK();
}

Status Model::Save(Model& model, const std::string& file_path) {
  int fd;
  LOTUS_RETURN_IF_ERROR(FileOpenWr(file_path, &fd));
  auto status = Save(model, fd);
  LOTUS_RETURN_IF_ERROR(FileClose(fd));
  return status;
}

using ::google::protobuf::io::CodedInputStream;
using ::google::protobuf::io::FileInputStream;
using ::google::protobuf::io::ZeroCopyInputStream;

Status Model::Load(int fd, std::shared_ptr<Model>& p_model, const ILotusOpSchemaCollection* local_registry) {
  if (fd < 0) {
    return Status(LOTUS, INVALID_ARGUMENT, "<p_fd> less than 0.");
  }

  auto raw_input = std::unique_ptr<ZeroCopyInputStream>(std::make_unique<FileInputStream>(fd));
  auto coded_input = std::make_unique<CodedInputStream>(raw_input.get());

  // Allows protobuf library versions < 3.2.0 to parse messages greater than 64MB.
  coded_input->SetTotalBytesLimit(INT_MAX, INT_MAX);

  std::unique_ptr<ModelProto> model_proto = std::make_unique<ModelProto>();
  const bool result = model_proto->ParseFromCodedStream(coded_input.get());
  coded_input.reset();
  raw_input.reset();

  if (!result) {
    return Status(LOTUS, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }

  p_model = std::make_shared<Model>(std::move(model_proto), local_registry);
  if (p_model->MainGraph() != nullptr) {
    LOTUS_RETURN_IF_ERROR(p_model->MainGraph()->Resolve(true));
  }
  return Status::OK();
}

Status Model::Save(Model& model, int p_fd) {
  if (p_fd < 0) {
    return Status(LOTUS, INVALID_ARGUMENT, "<p_fd> is less than 0.");
  }

  LOTUS_RETURN_IF_ERROR(model.MainGraph()->Resolve());
  auto model_proto = model.ToProto();
  const bool result = model_proto.SerializeToFileDescriptor(p_fd);
  if (result) {
    return Status::OK();
  } else {
    return Status(LOTUS, INVALID_PROTOBUF, "Protobuf serialization failed.");
  }
}
}  // namespace LotusIR

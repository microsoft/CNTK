#pragma once

#include "core/graph/graph.h"

// #include "gsl/pointers"

namespace LotusIR {
typedef std::unordered_map<std::string, std::string> ModelMetaData;

// A machine learning model representation class.
// Besides a main <Graph>, it also holds basic information, say,
// model version, model domain, model author, license etc.
class Model {
 public:
  const Version kNoVersion = INT64_MAX;

  // Construct model from scratch.
  explicit Model(const std::string& graph_name,
                 bool is_onnx_domain_only = false,
                 const ModelMetaData& model_metadata = ModelMetaData(),
                 const ILotusOpSchemaCollection* local_registry = nullptr);

  // NOTE: after calling this constructor, <*this> model will
  // hold a copy of <model_proto>.
  explicit Model(const ModelProto& model_proto, const ILotusOpSchemaCollection* local_registry = nullptr);

  // NOTE: after calling this constructor, <*this> model will
  // own the <model_proto>.
  explicit Model(std::unique_ptr<ModelProto> model_proto, const ILotusOpSchemaCollection* local_registry = nullptr);

  // Get model's IR version.
  // Return <kNoVersion> if not specified.
  Version IrVersion() const;

  // Get model's producer name.
  // Return null pointer if not specified.
  const std::string& ProducerName() const;
  // Set model's producer name.
  void SetProducerName(const std::string& producer_name);

  // Get model's producer version.
  // Return null pointer if not specified.
  const std::string& ProducerVersion() const;
  // Set model's producer version.
  void SetProducerVersion(const std::string& producer_version);

  // Get model's domain.
  // Return null pointer if not specified.
  const std::string& Domain() const;
  // Set models' domain.
  void SetDomain(const std::string& domain);

  // Get model's version.
  // Return null pointer if not specified.
  Version ModelVersion() const;
  // Set models' version.
  void SetModelversion(LotusIR::Version model_version);

  // Get model's doc string.
  // Return null pointer if not specified.
  const std::string& DocString() const;
  // Set models' doc string.
  void SetDocString(const std::string& doc_string);

  const ModelMetaData& MetaData() const noexcept;

  // Get model's main graph.
  // The return pointer is owned by <*this> model.
  // TODO(Task:131) Model::MainGraph can return reference as the value is never null
  Graph* MainGraph() noexcept;
  const Graph* MainGraph() const noexcept;

  // Get model's serialization proto data.
  ModelProto ToProto();

#ifdef _WIN32
  static Lotus::Common::Status Save(Model& model, const std::wstring& file_path);

  // TODO(Task:132) Use of shared_ptr<X>* in Load/Save methods is confusing.
  static Lotus::Common::Status Load(const std::wstring& file_path, /*out*/ std::shared_ptr<Model>& p_model,
                                    const ILotusOpSchemaCollection* local_registry = nullptr);
#endif
  static Lotus::Common::Status Save(Model& model, const std::string& file_path);

  static Lotus::Common::Status Save(Model& model, int fd);

  static Lotus::Common::Status Load(std::istream& model_istream, ModelProto* p_model_proto);

  static Lotus::Common::Status Load(const std::string& file_path,
                                    /*out*/ std::shared_ptr<Model>& p_model,
                                    const ILotusOpSchemaCollection* local_registry = nullptr);

  static Lotus::Common::Status Load(int fd, /*out*/ std::shared_ptr<Model>& p_model,
                                    const ILotusOpSchemaCollection* local_registry = nullptr);

  // 'int' rather than 'size_t' because of a protobuf design choice; let callers handle type checks
  static Lotus::Common::Status LoadFromBytes(int count, void* pBytes, /*out*/ std::shared_ptr<Model>& p_model,
                                             const ILotusOpSchemaCollection* local_registry = nullptr);

  static Lotus::Common::Status Load(const ModelProto& model_proto, /*out*/ std::shared_ptr<Model>& p_model,
                                    const ILotusOpSchemaCollection* local_registry = nullptr);

 private:
  // Set <domain_to_version_> and <model_proto_> to contain related domains
  // with latest version in OpSchemaRegistry.
  // if <is_onnx_domain_only> is true, then only onnx domain will be contained.
  // otherwise, ml domain will also be contained.
  void AddImportOpSets(bool is_onnx_domain_only,
                       /*out*/ std::unordered_map<std::string, int>* domain_to_version,
                       const ILotusOpSchemaCollection* local_registry);

  // Model data.
  std::unique_ptr<ModelProto> model_proto_;

  // This is a duplication of <model_proto_.metadata_props()>.
  // It gives better accessibility.
  ModelMetaData model_metadata_;

  // Main graph of the model.
  std::unique_ptr<Graph> graph_;
};
}  // namespace LotusIR

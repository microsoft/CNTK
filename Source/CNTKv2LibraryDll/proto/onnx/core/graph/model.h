// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <list>
#include <unordered_map>
#include <memory>
#include <climits>
#include <string>
#include "core/graph/function_container.h"
#include "core/graph/graph.h"

#include "gsl/pointers"

namespace onnxruntime {
typedef std::unordered_map<std::string, std::string> ModelMetaData;
using ILotusOpSchemaRegistryList = std::list<std::shared_ptr<ILotusOpSchemaCollection>>;

// A machine learning model representation class.
// Besides a main <Graph>, it also holds basic information, say,
// model version, model domain, model author, license etc.
class Model {
 public:
  static constexpr Version kNoVersion = INT64_MAX;

  // Construct model from scratch.
  explicit Model(const std::string& graph_name,
                 bool is_onnx_domain_only = false,
                 const ModelMetaData& model_metadata = ModelMetaData(),
                 const ILotusOpSchemaRegistryList* local_registries = nullptr,
                 const std::unordered_map<std::string, int>& domain_to_version = {});

  // NOTE: after calling this constructor, <*this> model will
  // hold a copy of <model_proto>.
  explicit Model(const ONNX_NAMESPACE::ModelProto& model_proto,
                 const ILotusOpSchemaRegistryList* local_registries = nullptr);

  // NOTE: after calling this constructor, <*this> model will
  // own the <model_proto>.
  explicit Model(std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto,
                 const ILotusOpSchemaRegistryList* local_registries = nullptr);

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
  void SetModelversion(onnxruntime::Version model_version);

  // Get model's doc string.
  // Return null pointer if not specified.
  const std::string& DocString() const;
  // Set models' doc string.
  void SetDocString(const std::string& doc_string);

  const ModelMetaData& MetaData() const noexcept;

  // Get model's main graph.
  Graph& MainGraph() noexcept;
  const Graph& MainGraph() const noexcept;

  // Get model's serialization proto data.
  ONNX_NAMESPACE::ModelProto ToProto();

#ifdef _WIN32
  static ::onnxruntime::common::Status Save(Model& model, const std::wstring& file_path);

  // TODO(Task:132) Use of shared_ptr<X>* in Load/Save methods is confusing.
  static ::onnxruntime::common::Status Load(const std::wstring& file_path, /*out*/ std::shared_ptr<Model>& p_model,
                                            const ILotusOpSchemaRegistryList* local_registry = nullptr);
#endif
  static ::onnxruntime::common::Status Save(Model& model, const std::string& file_path);

  static ::onnxruntime::common::Status Save(Model& model, int fd);

  static ::onnxruntime::common::Status Load(std::istream& model_istream, ONNX_NAMESPACE::ModelProto* p_model_proto);

  static ::onnxruntime::common::Status Load(const std::string& file_path,
                                            /*out*/ std::shared_ptr<Model>& p_model,
                                            const ILotusOpSchemaRegistryList* local_registries = nullptr);

  static ::onnxruntime::common::Status Load(int fd, /*out*/ std::shared_ptr<Model>& p_model,
                                            const ILotusOpSchemaRegistryList* local_registries = nullptr);

  // 'int' rather than 'size_t' because of a protobuf design choice; let callers handle type checks
  static ::onnxruntime::common::Status LoadFromBytes(int count, void* pBytes, /*out*/ std::shared_ptr<Model>& p_model,
                                                     const ILotusOpSchemaRegistryList* local_registries = nullptr);

  static ::onnxruntime::common::Status Load(const ONNX_NAMESPACE::ModelProto& model_proto, /*out*/ std::shared_ptr<Model>& p_model,
                                            const ILotusOpSchemaRegistryList* local_registries = nullptr);

  static ::onnxruntime::common::Status Load(std::unique_ptr<ONNX_NAMESPACE::ModelProto> p_model_proto, /*out*/ std::shared_ptr<Model>& p_model,
                                            const ILotusOpSchemaRegistryList* local_registries = nullptr);

 private:
  // Model data.
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto_;

  // This is a duplication of <model_proto_.metadata_props()>.
  // It gives better accessibility.
  ModelMetaData model_metadata_;

  // Main graph of the model.
  std::unique_ptr<Graph> graph_;
};
}  // namespace onnxruntime

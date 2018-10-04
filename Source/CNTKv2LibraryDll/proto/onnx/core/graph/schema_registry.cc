// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/schema_registry.h"

namespace onnxruntime {
// Add customized domain to min/max version.
::onnxruntime::common::Status LotusOpSchemaRegistry::SetBaselineAndOpsetVersionForDomain(
    const std::string& domain,
    int baseline_opset_version,
    int opset_version) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = domain_version_range_map_.find(domain);
  if (domain_version_range_map_.end() != it) {
    return ::onnxruntime::common::Status(::onnxruntime::common::LOTUS, ::onnxruntime::common::FAIL, "Domain already set in registry");
  }

  domain_version_range_map_[domain].baseline_opset_version = baseline_opset_version;
  domain_version_range_map_[domain].opset_version = opset_version;

  return ::onnxruntime::common::Status::OK();
}

Domain_To_Version_Map LotusOpSchemaRegistry::GetLatestOpsetVersions(bool is_onnx_only) const {
  Domain_To_Version_Map domain_version_map;

  for (auto& domain : domain_version_range_map_) {
    if (is_onnx_only && domain.first.compare(kOnnxDomain) != 0)
      continue;
    domain_version_map[domain.first] = domain.second.opset_version;
  }

  return domain_version_map;
}

::onnxruntime::common::Status LotusOpSchemaRegistry::RegisterOpSet(
    std::vector<ONNX_NAMESPACE::OpSchema>& schemas,
    const std::string& domain,
    int baseline_opset_version,
    int opset_version) {
  LOTUS_RETURN_IF_ERROR(SetBaselineAndOpsetVersionForDomain(domain, baseline_opset_version, opset_version));
  for (auto& schema : schemas)
    LOTUS_RETURN_IF_ERROR(RegisterOpSchema(std::move(schema)));
  return ::onnxruntime::common::Status::OK();
}

::onnxruntime::common::Status LotusOpSchemaRegistry::RegisterOpSchema(ONNX_NAMESPACE::OpSchema&& op_schema) {
  return RegisterOpSchemaInternal(std::move(op_schema));
}

::onnxruntime::common::Status LotusOpSchemaRegistry::RegisterOpSchemaInternal(ONNX_NAMESPACE::OpSchema&& op_schema) {
  try {
    op_schema.Finalize();
  } catch (const std::exception& e) {
    return ::onnxruntime::common::Status(::onnxruntime::common::LOTUS, ::onnxruntime::common::INVALID_ARGUMENT, "Schema error: " + std::string(e.what()));
  }

  auto& op_name = op_schema.Name();
  auto& op_domain = op_schema.domain();
  auto ver = op_schema.SinceVersion();

  if (map_[op_name][op_domain].count(ver)) {
    const auto& schema = map_[op_name][op_domain][ver];
    std::ostringstream ostream;
    ostream << "Trying to register schema with name " << op_name
            << " (domain: " << op_domain << " version: " << ver
            << ") from file " << op_schema.file() << " line "
            << op_schema.line()
            << ", but it is already registered from file "
            << schema.file() << " line " << schema.line() << std::endl;
    return ::onnxruntime::common::Status(::onnxruntime::common::LOTUS, ::onnxruntime::common::INVALID_ARGUMENT, ostream.str());
  }

  auto ver_range_it = domain_version_range_map_.find(op_domain);
  if (ver_range_it == domain_version_range_map_.end()) {
    std::ostringstream ostream;
    ostream << "Trying to register schema with name " << op_name
            << " (domain: " << op_domain << " version: " << ver
            << ") from file " << op_schema.file() << " line "
            << op_schema.line() << ", but it its domain is not"
            << "known by the checker." << std::endl;
    return ::onnxruntime::common::Status(::onnxruntime::common::LOTUS, ::onnxruntime::common::INVALID_ARGUMENT, ostream.str());
  }
  if (ver > ver_range_it->second.opset_version) {
    std::ostringstream ostream;
    ostream
        << "Trying to register schema with name " << op_name
        << " (domain: " << op_domain << " version: " << ver
        << ") from file " << op_schema.file() << " line "
        << op_schema.line() << ", but it its version is higher"
        << "than the operator set version " << ver_range_it->second.opset_version << std::endl;
    return ::onnxruntime::common::Status(::onnxruntime::common::LOTUS, ::onnxruntime::common::INVALID_ARGUMENT, ostream.str());
  }
  GSL_SUPPRESS(es .84)
  map_[op_name][op_domain].emplace(std::make_pair(ver, op_schema));
  return ::onnxruntime::common::Status::OK();
}

// Return the schema with biggest version, which is not greater than specified
// <op_set_version> in specified domain. The value of earliest_opset_where_unchanged
// is also set to the earliest version preceding op_set_version where the operator
// is known to be unchanged.
void LotusOpSchemaRegistry::GetSchemaAndHistory(
    const std::string& key,
    const int op_set_version,
    const std::string& domain,
    const ONNX_NAMESPACE::OpSchema** latest_schema,
    int* earliest_opset_where_unchanged) const {
  *latest_schema = nullptr;
  *earliest_opset_where_unchanged = std::numeric_limits<int>::max();

  // Determine if this registry contains the requested domain at the same or later
  // version
  auto domain_map_it = domain_version_range_map_.find(domain);
  if (domain_map_it != domain_version_range_map_.end() &&
      domain_map_it->second.opset_version >= op_set_version) {
    // If the baseline version is not larger than the requested version, initialize
    // the version at which the operator is unchanged to the baseline.  This will
    // be updated below if a schema is found.
    if (domain_map_it->second.baseline_opset_version <= op_set_version) {
      assert(domain_map_it->second.baseline_opset_version < domain_map_it->second.opset_version);
      *earliest_opset_where_unchanged = std::max(1, domain_map_it->second.baseline_opset_version);
    }

    auto it = map_.find(key);
    if (it == map_.end())
      return;
    auto s_it = it->second.find(domain);
    if (s_it != it->second.end()) {
      auto pos = s_it->second.lower_bound(op_set_version);
      if (s_it->second.begin() == pos && pos->first > op_set_version) {
        // All versions are greater than specified version.
        return;
      }

      if (s_it->second.end() == pos || pos->first > op_set_version) {
        // All versions are less than specified version, or,
        // The <pos> version is greater than specified version.
        --pos;
      }

      assert(pos->first <= op_set_version);

      if (pos->second.SinceVersion() <= op_set_version) {
        *latest_schema = &(pos->second);
        *earliest_opset_where_unchanged = (*latest_schema)->SinceVersion();
      }
    }
  }
}

void SchemaRegistryManager::RegisterRegistry(std::shared_ptr<ILotusOpSchemaCollection> registry) {
  registries.push_front(registry);
}

Domain_To_Version_Map SchemaRegistryManager::GetLatestOpsetVersions(bool is_onnx_only) const {
  Domain_To_Version_Map domain_version_map;

  // Build the map using each of the registries
  for (auto& registry : registries) {
    Domain_To_Version_Map latest_opset_versions_in_reg = registry->GetLatestOpsetVersions(is_onnx_only);

    for (auto& local_domain : latest_opset_versions_in_reg) {
      auto iter = domain_version_map.find(local_domain.first);

      // If the map doesn't yet contain this domain, insert it with this registry's value.
      // Otherwise, merge the existing range in the map.
      if (iter == domain_version_map.end()) {
        GSL_SUPPRESS(es .84)
        domain_version_map.insert(local_domain);
      } else {
        iter->second = std::max(iter->second, local_domain.second);
      }
    }
  }

  // check the ONNX schema registry
  auto& onnx_domain_version_map =
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().Map();

  for (auto domain : onnx_domain_version_map) {
    if (is_onnx_only && domain.first.compare(kOnnxDomain) != 0)
      continue;
    auto it = domain_version_map.find(domain.first);
    if (it == domain_version_map.end()) {
      GSL_SUPPRESS(es .84)
      domain_version_map.insert(std::make_pair(domain.first, domain.second.second));
    } else {
      it->second = std::max(it->second, domain.second.second);
    }
  }

  return domain_version_map;
}

// Return the schema with biggest version, which is not greater than specified
// <op_set_version> in specified domain. The value of earliest_opset_where_unchanged
// is also set to the earliest version preceding op_set_version where the operator
// is known to be unchanged.
void SchemaRegistryManager::GetSchemaAndHistory(
    const std::string& key,
    const int op_set_version,
    const std::string& domain,
    const ONNX_NAMESPACE::OpSchema** latest_schema,
    int* earliest_opset_where_unchanged) const {
  // A greedy algorithm is used to search for a schema registration in some registry,
  // while potentially inferring from other registries the allowed schema version
  // given the op-set version.  Each time a registry fails to locate the schema
  // but indicates that this schema was unchanged across its version span, the search
  // is restarted with a reduced op-set version.
  std::vector<int> unchecked_registry_indices(registries.size());
  std::iota(unchecked_registry_indices.begin(), unchecked_registry_indices.end(), 0);

  std::vector<int> checked_registry_indices;
  int version = op_set_version;
  while (!unchecked_registry_indices.empty()) {
    int index = unchecked_registry_indices.back();
    unchecked_registry_indices.pop_back();

    int new_version = std::numeric_limits<int>::max();
    registries[index]->GetSchemaAndHistory(key, version, domain, latest_schema, &new_version);
    if (*latest_schema != nullptr) {
      assert(new_version <= version && new_version <= op_set_version);
      *earliest_opset_where_unchanged = new_version;
      return;
    }

    if (new_version < version) {
      GSL_SUPPRESS(es .84)
      unchecked_registry_indices.insert(unchecked_registry_indices.end(),
                                        checked_registry_indices.begin(),
                                        checked_registry_indices.end());
      checked_registry_indices.clear();
      version = new_version;
    }

    checked_registry_indices.push_back(index);
  }

  // if not found in registered custom schema registry, search in ONNX schema registry
  *latest_schema = ONNX_NAMESPACE::OpSchemaRegistry::Schema(key, version, domain);
  if (*latest_schema != nullptr) {
    *earliest_opset_where_unchanged = (*latest_schema)->SinceVersion();
  }
}

}  // namespace onnxruntime

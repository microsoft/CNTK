#pragma once
#include "core/graph/constants.h"
#include "core/common/common.h"
#include "core/common/status.h"
#include "onnx/defs/schema.h"
#include <mutex>
#include "sstream"

namespace LotusIR {
using OpName_Domain_Version_Schema_Map = std::unordered_map<
    std::string,
    std::unordered_map<std::string, std::map<ONNX_NAMESPACE::OperatorSetVersion, ONNX_NAMESPACE::OpSchema>>>;

using Domain_To_Version_Map = std::unordered_map<std::string, std::pair<int, int>>;

class ILotusOpSchemaCollection {
 public:
  virtual Domain_To_Version_Map DomainToVersionMap() const = 0;

  virtual const ONNX_NAMESPACE::OpSchema* Schema(
      const std::string& key,
      const std::string& domain) const = 0;

  virtual const ONNX_NAMESPACE::OpSchema* Schema(
      const std::string& key,
      const int maxInclusiveVersion,
      const std::string& domain) const = 0;
};

// OpSchemaRegistry is a singleton in onnx, so we have to duplicate it in Lotus
// If later onnx design changed, we don't need it any more.
class LotusOpSchemaRegistry {
 public:
  LotusOpSchemaRegistry() = default;
  // Add customized domain to min/max version.
  // Onnx partners are able to use onnx operator schema api to
  // register customized op in their own domain.
  Lotus::Common::Status AddDomainToVersion(
      const std::string& domain,
      int max_version) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = domain_version_map_.find(domain);
    if (domain_version_map_.end() != it) {
      it->second.second = std::max(max_version, it->second.second);
      it->second.second = std::min(max_version, it->second.second);
    } else
      domain_version_map_[domain] = std::make_pair(max_version, max_version);
    return Lotus::Common::Status::OK();
  }

  const Domain_To_Version_Map& DomainVersionMap() const noexcept {
    return domain_version_map_;
  }

  Lotus::Common::Status RegisterOpSchema(ONNX_NAMESPACE::OpSchema& op_schema) {
    try {
      op_schema.Finalize();
    } catch (const std::exception& e) {
      return Lotus::Common::Status(Lotus::Common::LOTUS, Lotus::Common::INVALID_ARGUMENT, "Schema error: " + std::string(e.what()));
    }

    auto& op_name = op_schema.Name();
    auto& op_domain = op_schema.domain();
    auto ver = op_schema.SinceVersion();

    if (map_[op_name][op_domain].count(ver)) {
      const auto& schema = map_[op_name][op_domain][ver];
      std::stringstream ostream;
      ostream << "Trying to register schema with name " << op_name
              << " (domain: " << op_domain << " version: " << ver
              << ") from file " << op_schema.file() << " line "
              << op_schema.line()
              << ", but it is already registered from file "
              << schema.file() << " line " << schema.line() << std::endl;
      return Lotus::Common::Status(Lotus::Common::LOTUS, Lotus::Common::INVALID_ARGUMENT, ostream.str());
    }

    auto ver_range_map = DomainVersionMap();
    auto ver_range_it = ver_range_map.find(op_domain);
    if (ver_range_it == ver_range_map.end()) {
      std::stringstream ostream;
      ostream << "Trying to register schema with name " << op_name
              << " (domain: " << op_domain << " version: " << ver
              << ") from file " << op_schema.file() << " line "
              << op_schema.line() << ", but it its domain is not"
              << "known by the checker." << std::endl;
      return Lotus::Common::Status(Lotus::Common::LOTUS, Lotus::Common::INVALID_ARGUMENT, ostream.str());
    }
    auto lower_bound_incl = ver_range_it->second.first;
    auto upper_bound_incl = ver_range_it->second.second;
    if (!(lower_bound_incl <= ver && upper_bound_incl >= ver)) {
      std::stringstream ostream;
      ostream
          << "Trying to register schema with name " << op_name
          << " (domain: " << op_domain << " version: " << ver
          << ") from file " << op_schema.file() << " line "
          << op_schema.line() << ", but it its version is not"
          << "in the inclusive range [" << lower_bound_incl << ", "
          << upper_bound_incl << "] (usually, this means you "
          << "bumped the operator version but "
          << "forgot to update the version range in DomainToVersionRange "
          << "in onnx/defs/schema.h)." << std::endl;
      return Lotus::Common::Status(Lotus::Common::LOTUS, Lotus::Common::INVALID_ARGUMENT, ostream.str());
    }

    auto ignored = map_[op_name][op_domain].emplace(std::make_pair(ver, op_schema));
    return Lotus::Common::Status::OK();
  }

// conversion of kOnnxDomain to std::string creates unnamed temporary.  Suppress C26444 (es.84) the hard way.
// GSL_SUPPRESS(es.84) doesn't work as the default arg temporary isn't in a scope the suppress attribute handles.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26444)
#endif

  // Return the latest schema for an operator in specified domain.
  // Domain with default value ONNX_DOMAIN means ONNX.
  const ONNX_NAMESPACE::OpSchema* Schema(
      const std::string& key,
      const std::string& domain = kOnnxDomain) const {
    auto it = map_.find(key);
    if (it != map_.end()) {
      auto s_it = it->second.find(domain);
      if (s_it != it->second.end()) {
        return &s_it->second.rbegin()->second;
      } else {
        return nullptr;
      }
    } else {
      return nullptr;
    }
  }

  // Return the schema with biggest version, which is not greater than specified
  // <maxInclusiveVersion> in specified domain. Domain with default value
  // ONNX_DOMAIN means ONNX.
  const ONNX_NAMESPACE::OpSchema* Schema(
      const std::string& key,
      const int maxInclusiveVersion,
      const std::string& domain = kOnnxDomain) const {
    auto it = map_.find(key);
    if (it == map_.end())
      return nullptr;
    auto s_it = it->second.find(domain);
    if (s_it != it->second.end()) {
      auto pos = s_it->second.lower_bound(maxInclusiveVersion);
      if (s_it->second.begin() == pos && pos->first > maxInclusiveVersion) {
        // All versions are greater than specified version.
        return nullptr;
      }
      if (s_it->second.end() == pos || pos->first > maxInclusiveVersion) {
        // All versions are less than specified version, or,
        // The <pos> version is greater than specified version.
        --pos;
        return &(pos->second);
      }
      // Schema with exact version as specified one exists.
      return &(pos->second);
    } else {
      return nullptr;
    }
  }

#ifdef _MSC_VER
#pragma warning(pop)  // C26444
#endif

 private:
  std::mutex mutex_;

  OpName_Domain_Version_Schema_Map map_;
  Domain_To_Version_Map domain_version_map_;

 public:
  const std::vector<ONNX_NAMESPACE::OpSchema> get_all_schemas_with_history() {
    std::vector<ONNX_NAMESPACE::OpSchema> r;
    for (auto x : map_) {
      for (auto y : x.second) {
        for (auto z : y.second) {
          r.emplace_back(z.second);
        }
      }
    }
    return r;
  }

  const std::vector<ONNX_NAMESPACE::OpSchema> get_all_schemas() const {
    std::vector<ONNX_NAMESPACE::OpSchema> r;
    for (auto x : map_) {
      for (auto y : x.second) {
        auto& version2schema = y.second;
        r.emplace_back(version2schema.rbegin()->second);
      }
    }
    return r;
  }
};
}  // namespace LotusIR

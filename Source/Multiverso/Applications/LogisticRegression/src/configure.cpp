#include "configure.h"

#include <vector>
#include <unordered_map>

#include "util/log.h"
#include "multiverso/io/io.h"

namespace logreg {

#define CONFIG_PARSEVALUE(var,  type)   \
  ParseValue(&mp, #var, &var, type)

#define CONFIG_PARSE_INT(var)      \
  CONFIG_PARSEVALUE(var, kInt)

#define CONFIG_PARSE_BOOL(var)     \
  CONFIG_PARSEVALUE(var, kBool)

#define CONFIG_PARSE_STRING(var)   \
  CONFIG_PARSEVALUE(var, kString)

#define CONFIG_PARSE_DOUBLE(var)   \
  CONFIG_PARSEVALUE(var, kDouble)

#define CONFIG_PARSE_FLOAT(var)    \
  CONFIG_PARSEVALUE(var, kFloat)

#define CONFIG_PARSE_ULL(var)      \
  CONFIG_PARSEVALUE(var, kULL)

Configure::Configure(const std::string& config_file) {
  multiverso::TextReader reader(multiverso::URI(config_file), 1024);
  std::string line, key, value;
  std::unordered_map<std::string, std::string> mp;
  mp.reserve(20);
  size_t pos = -1;
  while (reader.GetLine(line)) {
    pos = line.find("=");
    if (pos == (size_t)-1) {
      Log::Write(Error, "Invalid configure line %s. Use key=value\n", 
        line.c_str());
      continue;
    }
    mp[line.substr(0, pos)] = 
      line.substr(pos + 1, line.length() - pos - 1);
  }
  
  CONFIG_PARSE_ULL(input_size);
  CONFIG_PARSE_INT(output_size);

  CONFIG_PARSE_INT(train_epoch);
  CONFIG_PARSE_INT(minibatch_size);

  CONFIG_PARSE_DOUBLE(regular_coef);
  CONFIG_PARSE_DOUBLE(learning_rate);
  CONFIG_PARSE_DOUBLE(learning_rate_coef);

  CONFIG_PARSE_DOUBLE(lambda1);
  CONFIG_PARSE_DOUBLE(lambda2);
  CONFIG_PARSE_DOUBLE(beta);
  CONFIG_PARSE_DOUBLE(alpha);

  CONFIG_PARSE_BOOL(use_ps);
  CONFIG_PARSE_BOOL(sparse);
  CONFIG_PARSE_BOOL(pipeline);
  CONFIG_PARSE_INT(sync_frequency);

  CONFIG_PARSE_STRING(train_file);
  CONFIG_PARSE_STRING(reader_type);
  CONFIG_PARSE_STRING(test_file);
  CONFIG_PARSE_INT(read_buffer_size);
  CONFIG_PARSE_INT(show_time_per_sample);

  CONFIG_PARSE_STRING(init_model_file);
  CONFIG_PARSE_STRING(output_model_file);
  CONFIG_PARSE_STRING(output_file);

  CONFIG_PARSE_STRING(updater_type);
  CONFIG_PARSE_STRING(objective_type);
  CONFIG_PARSE_STRING(regular_type);
}


void Configure::ParseValue(std::unordered_map<std::string, std::string>* mp,
  const std::string& key, void* value, ValueType type) {
  if (mp->find(key) == mp->end()) return;
  switch (type) {
  case logreg::Configure::kInt:
    ParseInt((*mp)[key], static_cast<int*>(value));
    break;
  case logreg::Configure::kFloat:
    ParseFloat((*mp)[key], static_cast<float*>(value));
    break;
  case logreg::Configure::kDouble:
    ParseDouble((*mp)[key], static_cast<double*>(value));
    break;
  case logreg::Configure::kString:
    *(std::string*)value = (*mp)[key];
    break;
  case logreg::Configure::kBool:
    ParseBool((*mp)[key], static_cast<bool*>(value));
    break;
  case logreg::Configure::kULL:
    ParseULL((*mp)[key], static_cast<unsigned long long*>(value));
    break;
  default:
    Log::Write(Error, "Unknown configure type\n");
    break;
  }
}

inline void Configure::ParseInt(const std::string value, int *vptr) {
  *vptr = atoi(value.c_str());
}

inline void Configure::ParseBool(const std::string value, bool* vptr) {
  *vptr = (value == "true");
}

inline void Configure::ParseFloat(const std::string value, float* vptr) {
  *vptr = strtof(value.c_str(), nullptr);
}

inline void Configure::ParseDouble(const std::string value, double* vptr) {
  *vptr = strtod(value.c_str(), nullptr);
}

inline void Configure::ParseULL(const std::string value, 
  unsigned long long* vptr) {
  *vptr = strtoull(value.c_str(), nullptr, 10);
}
}  // namespace logreg

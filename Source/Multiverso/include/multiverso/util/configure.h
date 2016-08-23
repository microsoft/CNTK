#ifndef MULTIVERSO_UTIL_CONFIGURE_H_
#define MULTIVERSO_UTIL_CONFIGURE_H_

#include <string>
#include <unordered_map>

namespace multiverso {

namespace configure {

template<typename T>
struct Command {
  T value;
  std::string description;
};

// used to register and keep flags
template<typename T>
class FlagRegister {
public:
  void RegisterFlag(const std::string& name,
                           const T& default_value, const std::string& text) {
    commands[name] = { default_value, text };
  }

  // set flag value if in the defined list
  bool SetFlagIfFound(const std::string& key, const T& value) {
    if (commands.find(key) != commands.end()) {
      commands[key].value = value;
      return true;
    }
    return false;
  }

  T& GetValue(const std::string& name) {
    return commands[name].value;
  }

  const std::string& GetInfo(const std::string& name) {
    return commands[name].description;
  }

  // get flag register instance
  static FlagRegister* Get() {
    static FlagRegister register_;
    return &register_;
  }

private:
  std::unordered_map<std::string, Command<T>> commands;

  FlagRegister() {}
  FlagRegister(FlagRegister<T>&) = delete;
};

template<typename T>
class FlagRegisterHelper {
public:
  FlagRegisterHelper(const std::string name, T val, const std::string &text) {
    FlagRegister<T>::Get()->RegisterFlag(name, val, text);
  }
};

// declare the variable as MV_CONFIG_##name
#define DECLARE_CONFIGURE(type, name)                                       \
  static const type& MV_CONFIG_##name = configure::FlagRegister<type>       \
  ::Get()->GetValue(#name);

// register a flag, use MV_CONFIG_##name to use
// \param type variable type
// \param name variable name
// \param default_vale
// \text description
#define DEFINE_CONFIGURE(type, name, default_value, text)                   \
  namespace configure {                                                     \
    FlagRegisterHelper<type> internal_configure_helper_##name(              \
      #name, default_value, text);                                          \
  }                                                                         \
  DECLARE_CONFIGURE(type, name)

}  // namespace configure

void ParseCMDFlags(int *argc, char* argv[]);

template <typename T>
void SetCMDFlag(const std::string& name, const T& value) {
  CHECK(configure::FlagRegister<T>::Get()->SetFlagIfFound(name, value));
}

#define MV_DEFINE_int(name, default_value, text) \
  DEFINE_CONFIGURE(int, name, default_value, text)

#define MV_DECLARE_int(name)  \
  DECLARE_CONFIGURE(int, name)

#define MV_DEFINE_string(name, default_value, text) \
  DEFINE_CONFIGURE(std::string, name, default_value, text)

#define MV_DECLARE_string(name)  \
  DECLARE_CONFIGURE(std::string, name)

#define MV_DEFINE_bool(name, default_value, text) \
  DEFINE_CONFIGURE(bool, name, default_value, text)

#define MV_DECLARE_bool(name)  \
  DECLARE_CONFIGURE(bool, name)
 
#define MV_DEFINE_double(name, default_value, text) \
  DEFINE_CONFIGURE(double, name, default_value, text)

#define MV_DECLARE_double(name)  \
  DECLARE_CONFIGURE(double, name)
}  // namespace multiverso

#endif  // MULTIVERSO_UTIL_CONFIGURE_H_

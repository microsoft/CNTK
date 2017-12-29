#ifndef JS_SHELL_H
#define JS_SHELL_H

#include <vector>
#include <string>

typedef void* HANDLE;
typedef void* MODULE;

class JSShell {

public:
  enum Engine {
    JSC = 0,
    V8
  };

public:

  JSShell() {}

  virtual ~JSShell() = 0;

  static JSShell* Create(Engine engine = JSC);

  std::string LoadModule(const std::string& name, HANDLE* library);

  virtual bool RunScript(const std::string& scriptPath);

  virtual bool RunShell();

  void setModulePath(const std::vector<std::string>& modulePath) {
    module_path = modulePath;
  }

protected:

  virtual bool InitializeEngine() = 0;

  virtual bool ExecuteScript(const std::string& source, const std::string& scriptPath) = 0;

  virtual bool DisposeEngine() = 0;

  static std::string ReadFile(const std::string& fileName);

protected:

  std::vector<HANDLE> loaded_modules;
  std::vector<std::string> module_path;

};

#endif // JS_SHELL_H

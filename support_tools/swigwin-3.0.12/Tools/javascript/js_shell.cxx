#include "js_shell.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef __GNUC__
#ifdef __APPLE__
#define LIBRARY_EXT ".bundle"
#else
#define LIBRARY_EXT ".so"
#endif
#include <dlfcn.h>
#define LOAD_LIBRARY(name) dlopen(name, RTLD_LAZY)
#define CLOSE_LIBRARY(handle) dlclose(handle)
#define LIBRARY_ERROR dlerror
#define LIBRARYFILE(name) std::string("lib").append(name).append(LIBRARY_EXT)
#else
#error "implement dll loading"
#endif


JSShell::~JSShell() {

  for(std::vector<HANDLE>::iterator it = loaded_modules.begin();
    it != loaded_modules.end(); ++it) {
      HANDLE handle = *it;
      CLOSE_LIBRARY(handle);
  }

}

// TODO: this could be done more intelligent...
// - can we achieve source file relative loading?
// - better path resolution
std::string JSShell::LoadModule(const std::string& name, HANDLE* library) {

  // works only for posix like OSs
  size_t pathIdx = name.find_last_of("/");

  std::string lib_name;
  std::string module_name;

  if (pathIdx == std::string::npos) {
    module_name = name;
    lib_name = std::string(name).append(LIBRARY_EXT);
  } else {
    std::string path = name.substr(0, pathIdx+1);
    module_name = name.substr(pathIdx+1);
    lib_name = path.append(module_name).append(LIBRARY_EXT);
  }

  std::string lib_path;
  HANDLE handle = 0;

  for (int i = 0; i < module_path.size(); ++i) {
    lib_path = module_path[i] + "/" + lib_name;
    if (access( lib_path.c_str(), F_OK ) != -1) {
      handle = LOAD_LIBRARY(lib_path.c_str());
    }
  }

  if(handle == 0) {
    std::cerr << "Could not find module " << lib_path << ":"
              << std::endl << LIBRARY_ERROR() << std::endl;
    return 0;
  }

  loaded_modules.push_back(handle);

  *library = handle;

  return module_name;
}

bool JSShell::RunScript(const std::string& scriptPath) {
  std::string source = ReadFile(scriptPath);
  if(!InitializeEngine()) return false;

  // Node.js compatibility: make `print` available as `console.log()`
  ExecuteScript("var console = {}; console.log = print;", "<console>");

  if(!ExecuteScript(source, scriptPath)) {
    return false;
  }

  return DisposeEngine();
}

bool JSShell::RunShell() {

  if(!InitializeEngine()) return false;

  static const int kBufferSize = 1024;
  while (true) {
    char buffer[kBufferSize];
    printf("> ");
    char* str = fgets(buffer, kBufferSize, stdin);
    if (str == NULL) break;
    std::string source(str);
    ExecuteScript(source, "(shell)");
  }
  printf("\n");
  return true;
}

std::string JSShell::ReadFile(const std::string& fileName)
{
  std::string script;

  std::ifstream file(fileName.c_str());
  if (file.is_open()) {
    while ( file.good() ) {
      std::string line;
      getline(file, line);
      script.append(line);
      script.append("\n");
    }
    file.close();
  } else {
    std::cout << "Unable to open file " << fileName << "." << std::endl;
  }

  return script;
}

#ifdef ENABLE_JSC
extern JSShell* JSCShell_Create();
#endif
#ifdef ENABLE_V8
extern JSShell* V8Shell_Create();
#endif

typedef JSShell*(*ShellFactory)();

static ShellFactory js_shell_factories[2] = {
#ifdef ENABLE_JSC
JSCShell_Create,
#else
0,
#endif
#ifdef ENABLE_V8
V8Shell_Create,
#else
0,
#endif
};

JSShell *JSShell::Create(Engine engine) {
  if(js_shell_factories[engine] == 0) {
    throw "Engine not available.";
  }
  return js_shell_factories[engine]();
}

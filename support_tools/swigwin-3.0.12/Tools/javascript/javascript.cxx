#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <string>
#include <vector>

#include "js_shell.h"

void print_usage() {
  std::cout << "javascript [-i] [-jsc|-v8] [-l module] <js-file>" << std::endl;
}

int main(int argc, char* argv[]) {

#if defined(JAVASCRIPT_INTERPRETER_STOP)
  std::cout << "Attach your Debugger and press any key to continue" << std::endl;
  std::cin.get();
#endif

  std::string scriptPath = "";

  bool interactive = false;
  JSShell* shell = 0;

  std::vector<std::string> modulePath;
  modulePath.push_back(".");

  for (int idx = 1; idx < argc; ++idx) {
    if(strcmp(argv[idx], "-v8") == 0) {
      shell = JSShell::Create(JSShell::V8);
    } else if(strcmp(argv[idx], "-jsc") == 0) {
      shell = JSShell::Create(JSShell::JSC);
    } else if(strcmp(argv[idx], "-i") == 0) {
      interactive = true;
    } else if(strcmp(argv[idx], "-L") == 0) {
      modulePath.push_back(argv[++idx]);
    } else {
      scriptPath = argv[idx];
    }
  }

  if (shell == 0) {
    shell = JSShell::Create();
  }

  shell->setModulePath(modulePath);

  bool failed = false;

  if(interactive) {
    failed = !(shell->RunShell());
  } else {
    failed = !(shell->RunScript(scriptPath));
  }

  if (failed) {
    delete shell;
    printf("FAIL: Error during execution of script.\n");
    return 1;
  }

  delete shell;

  return 0;
}

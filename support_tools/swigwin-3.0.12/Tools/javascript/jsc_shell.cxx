#include <JavaScriptCore/JavaScript.h>

#include "js_shell.h"

#include <iostream>
#include <stdio.h>

#ifdef __GNUC__
#include <dlfcn.h>
#define LOAD_SYMBOL(handle, name) dlsym(handle, name)
#else
#error "implement dll loading"
#endif

class JSCShell: public JSShell {

typedef int (*JSCIntializer)(JSGlobalContextRef context, JSObjectRef *module);

public:

  JSCShell() {};

  virtual ~JSCShell();

protected:

  virtual bool InitializeEngine();

  virtual bool ExecuteScript(const std::string& source, const std::string& scriptPath);

  virtual bool DisposeEngine();

private:

  JSObjectRef Import(const std::string &moduleName);

  static JSValueRef Print(JSContextRef context, JSObjectRef object, JSObjectRef globalobj, size_t argc, const JSValueRef args[], JSValueRef* ex);

  static JSValueRef Require(JSContextRef context, JSObjectRef object, JSObjectRef globalobj, size_t argc, const JSValueRef args[], JSValueRef* ex);

  static bool RegisterFunction(JSGlobalContextRef context, JSObjectRef object, const char* functionName, JSObjectCallAsFunctionCallback cbFunction);

  static void PrintError(JSContextRef, JSValueRef);

private:

  JSGlobalContextRef context;
};

JSCShell::~JSCShell() {
  if(context != 0) {
    JSGlobalContextRelease(context);
    context = 0;
  }
}

bool JSCShell::InitializeEngine() {
  if(context != 0) {
    JSGlobalContextRelease(context);
    context = 0;
  }
  // TODO: check for initialization errors
  context = JSGlobalContextCreate(NULL);
  if(context == 0) return false;
  JSObjectRef globalObject = JSContextGetGlobalObject(context);

  // store this for later use
  JSClassDefinition __shell_classdef__ = JSClassDefinition();

  JSClassRef __shell_class__ = JSClassCreate(&__shell_classdef__);
  JSObjectRef __shell__ = JSObjectMake(context, __shell_class__, 0);
  bool success = JSObjectSetPrivate(__shell__, (void*) (long) this);
  if (!success) {
    std::cerr << "Could not register the shell in the Javascript context" << std::endl;
    return false;
  }
  JSStringRef shellKey = JSStringCreateWithUTF8CString("__shell__");
  JSObjectSetProperty(context, globalObject, shellKey, __shell__, kJSPropertyAttributeReadOnly, NULL);
  JSStringRelease(shellKey);

  JSCShell::RegisterFunction(context, globalObject, "print", JSCShell::Print);
  JSCShell::RegisterFunction(context, globalObject, "require", JSCShell::Require);

  return true;
}

bool JSCShell::ExecuteScript(const std::string& source, const std::string& scriptPath) {
  JSStringRef jsScript;
  JSStringRef sourceURL;
  JSValueRef ex;
  jsScript = JSStringCreateWithUTF8CString(source.c_str());
  sourceURL = JSStringCreateWithUTF8CString(scriptPath.c_str());
  JSValueRef jsResult = JSEvaluateScript(context, jsScript, 0, sourceURL, 0, &ex);
  JSStringRelease(jsScript);
  if (jsResult == NULL && ex != NULL) {
      JSCShell::PrintError(context, ex);
      return false;
  }
  return true;
}

bool JSCShell::DisposeEngine() {
  JSGlobalContextRelease(context);
  context = 0;
  return true;
}

JSValueRef JSCShell::Print(JSContextRef context, JSObjectRef object,
                           JSObjectRef globalobj, size_t argc,
                           const JSValueRef args[], JSValueRef* ex) {
  if (argc > 0)
  {
    JSStringRef string = JSValueToStringCopy(context, args[0], NULL);
    size_t numChars = JSStringGetMaximumUTF8CStringSize(string);
    char *stringUTF8 = new char[numChars];
    JSStringGetUTF8CString(string, stringUTF8, numChars);
    printf("%s\n", stringUTF8);

    delete[] stringUTF8;
  }

  return JSValueMakeUndefined(context);
}

// Attention: this feature should not create too high expectations.
// It is only capable of loading things relative to the execution directory
// and not relative to the parent script.
JSValueRef JSCShell::Require(JSContextRef context, JSObjectRef object,
                           JSObjectRef globalObj, size_t argc,
                           const JSValueRef args[], JSValueRef* ex) {
  JSObjectRef module;

  JSStringRef shellKey = JSStringCreateWithUTF8CString("__shell__");
  JSValueRef shellAsVal = JSObjectGetProperty(context, globalObj, shellKey, NULL);
  JSStringRelease(shellKey);
  JSObjectRef shell = JSValueToObject(context, shellAsVal, 0);
  JSCShell *_this = (JSCShell*) (long) JSObjectGetPrivate(shell);

  if (argc > 0)
  {
    JSStringRef string = JSValueToStringCopy(context, args[0], NULL);
    size_t numChars = JSStringGetMaximumUTF8CStringSize(string);
    char *stringUTF8 = new char[numChars];
    JSStringGetUTF8CString(string, stringUTF8, numChars);

    std::string modulePath(stringUTF8);
    module = _this->Import(modulePath);

    delete[] stringUTF8;
  }

  if (module) {
    return module;
  } else {
    printf("Ooops.\n");
    return JSValueMakeUndefined(context);
  }
}

JSObjectRef JSCShell::Import(const std::string& module_path) {

  HANDLE library;
  std::string module_name = LoadModule(module_path, &library);

  if (library == 0) {
    printf("Could not load module.");
    return 0;
  }

  std::string symname = std::string(module_name).append("_initialize");

  JSCIntializer init_function = reinterpret_cast<JSCIntializer>((long) LOAD_SYMBOL(library, symname.c_str()));
  if(init_function == 0) {
    printf("Could not find module's initializer function.");
    return 0;
  }

  JSObjectRef module;
  init_function(context, &module);

  return module;
}

bool JSCShell::RegisterFunction(JSGlobalContextRef context, JSObjectRef object,
                        const char* functionName, JSObjectCallAsFunctionCallback callback) {
    JSStringRef js_functionName = JSStringCreateWithUTF8CString(functionName);
    JSObjectSetProperty(context, object, js_functionName,
                        JSObjectMakeFunctionWithCallback(context, js_functionName, callback),
                        kJSPropertyAttributeNone, NULL);
    JSStringRelease(js_functionName);
    return true;
}

void JSCShell::PrintError(JSContextRef ctx, JSValueRef err) {
  char *buffer;
  size_t length;

  JSStringRef string = JSValueToStringCopy(ctx, err, 0);
  length = JSStringGetLength(string);
  buffer   = new char[length+1];
  JSStringGetUTF8CString(string, buffer, length+1);
  std::string errMsg(buffer);
  JSStringRelease(string);
  delete[] buffer;

  JSObjectRef errObj = JSValueToObject(ctx, err, 0);

  if(errObj == 0) {
    std::cerr << errMsg << std::endl;
    return;
  }

  JSStringRef sourceURLKey = JSStringCreateWithUTF8CString("sourceURL");
  JSStringRef sourceURLStr = JSValueToStringCopy(ctx, JSObjectGetProperty(ctx, errObj, sourceURLKey, 0), 0);
  length = JSStringGetLength(sourceURLStr);
  buffer   = new char[length+1];
  JSStringGetUTF8CString(sourceURLStr, buffer, length+1);
  std::string sourceURL(buffer);
  delete[] buffer;
  JSStringRelease(sourceURLStr);
  JSStringRelease(sourceURLKey);

  JSStringRef lineKey = JSStringCreateWithUTF8CString("line");
  JSValueRef jsLine = JSObjectGetProperty(ctx, errObj, lineKey, 0);
  int line = (int) JSValueToNumber(ctx, jsLine, 0);
  JSStringRelease(lineKey);

  std::cerr << sourceURL << ":" << line << ":" << errMsg << std::endl;
}

JSShell* JSCShell_Create() {
  return new JSCShell();
}

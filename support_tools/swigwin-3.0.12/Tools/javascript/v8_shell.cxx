#include <assert.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <v8.h>
#include <vector>

#include "js_shell.h"

typedef int (*V8ExtensionInitializer) (v8::Handle<v8::Object> module);

// Note: these typedefs and defines are used to deal with  v8 API changes since version 3.19.00

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031903)
typedef v8::Handle<v8::Value> SwigV8ReturnValue;
typedef v8::Arguments SwigV8Arguments;
typedef v8::AccessorInfo SwigV8PropertyCallbackInfo;
#define SWIGV8_RETURN(val) return scope.Close(val)
#define SWIGV8_RETURN_INFO(val, info) return scope.Close(val)
#else
typedef void SwigV8ReturnValue;
typedef v8::FunctionCallbackInfo<v8::Value> SwigV8Arguments;
typedef v8::PropertyCallbackInfo<v8::Value> SwigV8PropertyCallbackInfo;
#define SWIGV8_RETURN(val) args.GetReturnValue().Set(val); return
#define SWIGV8_RETURN_INFO(val, info) info.GetReturnValue().Set(val); return
#endif


#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x032117)
#define SWIGV8_HANDLESCOPE() v8::HandleScope scope
#define SWIGV8_HANDLESCOPE_ESC() v8::HandleScope scope
#define SWIGV8_ESCAPE(val) return scope.Close(val)
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x032318)
#define SWIGV8_HANDLESCOPE() v8::HandleScope scope(v8::Isolate::GetCurrent());
#define SWIGV8_HANDLESCOPE_ESC() v8::HandleScope scope(v8::Isolate::GetCurrent());
#define SWIGV8_ESCAPE(val) return scope.Close(val)
#else
#define SWIGV8_HANDLESCOPE() v8::HandleScope scope(v8::Isolate::GetCurrent());
#define SWIGV8_HANDLESCOPE_ESC() v8::EscapableHandleScope scope(v8::Isolate::GetCurrent());
#define SWIGV8_ESCAPE(val) return scope.Escape(val)
#endif

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x032318)
#define SWIGV8_CURRENT_CONTEXT() v8::Context::GetCurrent()
#define SWIGV8_STRING_NEW(str) v8::String::New(str)
#define SWIGV8_FUNCTEMPLATE_NEW(func) v8::FunctionTemplate::New(func)
#define SWIGV8_OBJECT_NEW() v8::Object::New()
#define SWIGV8_EXTERNAL_NEW(val) v8::External::New(val)
#define SWIGV8_UNDEFINED() v8::Undefined()
#else
#define SWIGV8_CURRENT_CONTEXT() v8::Isolate::GetCurrent()->GetCurrentContext()
#define SWIGV8_STRING_NEW(str) v8::String::NewFromUtf8(v8::Isolate::GetCurrent(), str)
#define SWIGV8_FUNCTEMPLATE_NEW(func) v8::FunctionTemplate::New(v8::Isolate::GetCurrent(), func)
#define SWIGV8_OBJECT_NEW() v8::Object::New(v8::Isolate::GetCurrent())
#define SWIGV8_EXTERNAL_NEW(val) v8::External::New(v8::Isolate::GetCurrent(), val)
#define SWIGV8_UNDEFINED() v8::Undefined(v8::Isolate::GetCurrent())
#endif


#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031900)
typedef v8::Persistent<v8::Context> SwigV8Context;
#else
typedef v8::Local<v8::Context> SwigV8Context;
#endif

class V8Shell: public JSShell {

public:
  V8Shell();

  virtual ~V8Shell();

  virtual bool RunScript(const std::string &scriptPath);

  virtual bool RunShell();


protected:

  virtual bool InitializeEngine();

  virtual bool ExecuteScript(const std::string &source, const std::string &scriptPath);

  virtual bool DisposeEngine();

private:

  v8::Handle<v8::Value> Import(const std::string &moduleName);

  SwigV8Context CreateShellContext();

  void ReportException(v8::TryCatch *handler);

  static SwigV8ReturnValue Print(const SwigV8Arguments &args);

  static SwigV8ReturnValue Require(const SwigV8Arguments &args);

  static SwigV8ReturnValue Quit(const SwigV8Arguments &args);

  static SwigV8ReturnValue Version(const SwigV8Arguments &args);

  static const char* ToCString(const v8::String::Utf8Value &value);

};

#ifdef __GNUC__
#include <dlfcn.h>
#define LOAD_SYMBOL(handle, name) dlsym(handle, name)
#else
#error "implement dll loading"
#endif

V8Shell::V8Shell() {}

V8Shell::~V8Shell() {}

bool V8Shell::RunScript(const std::string &scriptPath) {
  std::string source = ReadFile(scriptPath);

  v8::Isolate *isolate = v8::Isolate::New();
  v8::Isolate::Scope isolate_scope(isolate);

  SWIGV8_HANDLESCOPE();

  SwigV8Context context = CreateShellContext();

  if (context.IsEmpty()) {
      printf("Could not create context.\n");
      return false;
  }

  context->Enter();

  // Store a pointer to this shell for later use

  v8::Handle<v8::Object> global = context->Global();
  v8::Local<v8::External> __shell__ = SWIGV8_EXTERNAL_NEW((void*) (long) this);

  global->SetHiddenValue(SWIGV8_STRING_NEW("__shell__"), __shell__);

  // Node.js compatibility: make `print` available as `console.log()`
  ExecuteScript("var console = {}; console.log = print;", "<console>");

  bool success = ExecuteScript(source, scriptPath);

  // Cleanup

  context->Exit();

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
    context.Dispose();
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031900)
    context.Dispose(v8::Isolate::GetCurrent());
#else
//    context.Dispose();
#endif

//  v8::V8::Dispose();

  return success;
}

bool V8Shell::RunShell() {
  SWIGV8_HANDLESCOPE();

  SwigV8Context context = CreateShellContext();

  if (context.IsEmpty()) {
      printf("Could not create context.\n");
      return false;
  }

  context->Enter();

  v8::Context::Scope context_scope(context);

  ExecuteScript("var console = {}; console.log = print;", "<console>");

  static const int kBufferSize = 1024;
  while (true) {
    char buffer[kBufferSize];
    printf("> ");
    char *str = fgets(buffer, kBufferSize, stdin);
    if (str == NULL) break;
    std::string source(str);
    ExecuteScript(source, "(shell)");
  }
  printf("\n");

  // Cleanup

  context->Exit();

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
    context.Dispose();
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031900)
    context.Dispose(v8::Isolate::GetCurrent());
#else
//    context.Dispose();
#endif

//  v8::V8::Dispose();

  return true;
}


bool V8Shell::InitializeEngine() {
  return true;
}

bool V8Shell::ExecuteScript(const std::string &source, const std::string &name) {
  SWIGV8_HANDLESCOPE();

  v8::TryCatch try_catch;
  v8::Handle<v8::Script> script = v8::Script::Compile(SWIGV8_STRING_NEW(source.c_str()), SWIGV8_STRING_NEW(name.c_str()));

  // Stop if script is empty
  if (script.IsEmpty()) {
    // Print errors that happened during compilation.
    ReportException(&try_catch);
    return false;
  }

  v8::Handle<v8::Value> result = script->Run();

  // Print errors that happened during execution.
  if (try_catch.HasCaught()) {
    ReportException(&try_catch);
    return false;
  } else {
    return true;
  }
}

bool V8Shell::DisposeEngine() {
  return true;
}

SwigV8Context V8Shell::CreateShellContext() {
  // Create a template for the global object.
  v8::Handle<v8::ObjectTemplate> global = v8::ObjectTemplate::New();

  // Bind global functions
  global->Set(SWIGV8_STRING_NEW("print"), SWIGV8_FUNCTEMPLATE_NEW(V8Shell::Print));
  global->Set(SWIGV8_STRING_NEW("quit"), SWIGV8_FUNCTEMPLATE_NEW(V8Shell::Quit));
  global->Set(SWIGV8_STRING_NEW("require"), SWIGV8_FUNCTEMPLATE_NEW(V8Shell::Require));
  global->Set(SWIGV8_STRING_NEW("version"), SWIGV8_FUNCTEMPLATE_NEW(V8Shell::Version));

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031900)
  SwigV8Context context = v8::Context::New(NULL, global);
  return context;
#else
  SwigV8Context context = v8::Context::New(v8::Isolate::GetCurrent(), NULL, global);
  return context;
#endif
}

v8::Handle<v8::Value> V8Shell::Import(const std::string &module_path)
{
  SWIGV8_HANDLESCOPE_ESC();

  HANDLE library;
  std::string module_name = LoadModule(module_path, &library);

  std::string symname = std::string(module_name).append("_initialize");

  V8ExtensionInitializer init_function = reinterpret_cast<V8ExtensionInitializer>((long) LOAD_SYMBOL(library, symname.c_str()));

  if(init_function == 0) {
    printf("Could not find initializer function.");

    return SWIGV8_UNDEFINED();
  }

  v8::Local<v8::Object> module = SWIGV8_OBJECT_NEW();
  init_function(module);

  SWIGV8_ESCAPE(module);
}

SwigV8ReturnValue V8Shell::Print(const SwigV8Arguments &args) {
  SWIGV8_HANDLESCOPE();

  bool first = true;
  for (int i = 0; i < args.Length(); i++) {

    if (first) {
      first = false;
    } else {
      printf(" ");
    }
    v8::String::Utf8Value str(args[i]);
    const char *cstr = V8Shell::ToCString(str);
    printf("%s", cstr);
  }
  printf("\n");
  fflush(stdout);

  SWIGV8_RETURN(SWIGV8_UNDEFINED());
}

SwigV8ReturnValue V8Shell::Require(const SwigV8Arguments &args) {
  SWIGV8_HANDLESCOPE();

  if (args.Length() != 1) {
    printf("Illegal arguments for `require`");
  };

  v8::String::Utf8Value str(args[0]);
  const char *cstr = V8Shell::ToCString(str);
  std::string moduleName(cstr);

  v8::Local<v8::Object> global = SWIGV8_CURRENT_CONTEXT()->Global();

  v8::Local<v8::Value> hidden = global->GetHiddenValue(SWIGV8_STRING_NEW("__shell__"));
  v8::Local<v8::External> __shell__ = v8::Local<v8::External>::Cast(hidden);
  V8Shell *_this = (V8Shell *) (long) __shell__->Value();

  v8::Handle<v8::Value> module = _this->Import(moduleName);

  SWIGV8_RETURN(module);
}

SwigV8ReturnValue V8Shell::Quit(const SwigV8Arguments &args) {
  SWIGV8_HANDLESCOPE();

  int exit_code = args[0]->Int32Value();
  fflush(stdout);
  fflush(stderr);
  exit(exit_code);

  SWIGV8_RETURN(SWIGV8_UNDEFINED());
}

SwigV8ReturnValue V8Shell::Version(const SwigV8Arguments &args) {
    SWIGV8_HANDLESCOPE();
    SWIGV8_RETURN(SWIGV8_STRING_NEW(v8::V8::GetVersion()));
}

void V8Shell::ReportException(v8::TryCatch *try_catch) {
  SWIGV8_HANDLESCOPE();

  v8::String::Utf8Value exception(try_catch->Exception());
  const char *exception_string = V8Shell::ToCString(exception);
  v8::Handle<v8::Message> message = try_catch->Message();
  if (message.IsEmpty()) {
    // V8 didn't provide any extra information about this error; just
    // print the exception.
    printf("%s\n", exception_string);
  } else {
    // Print (filename):(line number): (message).
    v8::String::Utf8Value filename(message->GetScriptResourceName());
    const char *filename_string = V8Shell::ToCString(filename);
    int linenum = message->GetLineNumber();
    printf("%s:%i: %s\n", filename_string, linenum, exception_string);
    // Print line of source code.
    v8::String::Utf8Value sourceline(message->GetSourceLine());
    const char *sourceline_string = V8Shell::ToCString(sourceline);
    printf("%s\n", sourceline_string);
    // Print wavy underline (GetUnderline is deprecated).
    int start = message->GetStartColumn();
    for (int i = 0; i < start; i++) {
      printf(" ");
    }
    int end = message->GetEndColumn();
    for (int i = start; i < end; i++) {
      printf("^");
    }
    printf("\n");
    v8::String::Utf8Value stack_trace(try_catch->StackTrace());
    if (stack_trace.length() > 0) {
      const char *stack_trace_string = V8Shell::ToCString(stack_trace);
      printf("%s\n", stack_trace_string);
    }
  }
}

// Extracts a C string from a V8 Utf8Value.
const char *V8Shell::ToCString(const v8::String::Utf8Value &value) {
  return *value ? *value : "<string conversion failed>";
}

JSShell *V8Shell_Create() {
  return new V8Shell();
}

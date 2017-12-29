%module overload_bool

%inline %{
const char* overloaded(bool value) { return "bool"; }
const char* overloaded(int value) { return "int"; }
const char* overloaded(const char *value) { return "string"; }

const char* boolfunction(bool value) { return value ? "true" : "false"; }
const char* intfunction(int value) { return "int"; }


// Const references
const char* overloaded_ref(bool const& value) { return "bool"; }
const char* overloaded_ref(int const& value) { return "int"; }
const char* overloaded_ref(const char *value) { return "string"; }

const char* boolfunction_ref(bool const& value) { return value ? "true" : "false"; }
const char* intfunction_ref(int const& value) { return "int"; }
%}

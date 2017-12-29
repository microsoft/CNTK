%module python_strict_unicode

%include <std_string.i>
%include <std_wstring.i>

%begin %{
#define SWIG_PYTHON_STRICT_BYTE_CHAR
#define SWIG_PYTHON_STRICT_UNICODE_WCHAR
%}

%inline %{
std::string double_str(const std::string& in)
{
  return in + in;
}

char *same_str(char* in)
{
  return in;
}

std::wstring double_wstr(const std::wstring& in)
{
  return in + in;
}

wchar_t *same_wstr(wchar_t* in)
{
  return in;
}

std::wstring overload(const std::wstring& in)
{
  return L"UNICODE";
}

std::string overload(const std::string& in)
{
  return "BYTES";
}
%}

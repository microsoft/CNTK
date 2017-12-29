%module operator_pointer_ref

%{
#if defined(_MSC_VER)
  #pragma warning(disable: 4996) // 'strdup': The POSIX name for this item is deprecated. Instead, use the ISO C++ conformant name: _strdup. See online help for details.
#endif
%}

%rename(AsCharStarRef) operator char*&;

%inline %{
class MyClass {
public:
    MyClass (const char *s_ = "")
	: s(strdup(s_ ? s_ : ""))
    { }

    ~MyClass () 
    { free(s); }

    operator char*&()
    { return s; }

private:
    char *s;
};
%}

%module ordering

// Ruby used to fail on the ordering of the two Class declarations below

struct Klass {
  int variable;
};

%{
struct Klass {
  int variable;
};
%}


// Testing the order of various code block sections

%runtime %{
   class RuntimeSection {};
%}

%header %{
   class HeaderSection {};
   void HeaderMethod(RuntimeSection rs) {}
%}

%wrapper %{
   void WrapperMethod(HeaderSection hs, RuntimeSection rs) {}
%}


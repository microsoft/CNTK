%module typedef_array_member
%inline %{

typedef char amember[20];

struct Foo {
   amember x;
};

%}


%ignore jbuf_tag;
%inline %{

  typedef struct jbuf_tag
  {
    int mask;
  } jbuf[1];
  
  struct Ast_channel {
    jbuf jmp[32];
  };

%}

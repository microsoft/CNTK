%module fragments


%fragment("Hello","header") %{
/* hello!!! */
int foobar(int a)
{
  return a;
}  
%}

/*
 this fragment include the previous fragment if needed.
*/

%fragment("Hi","header",fragment="Hello") %{
/* hi!!! */
int bar(int a)
{
  return foobar(a);
}  
%}

%typemap(in,fragment="Hi") int hola "$1 = 123;";


%inline %{

int bar(int a);

int foo(int hola) 
{
  return bar(hola);
}

%}

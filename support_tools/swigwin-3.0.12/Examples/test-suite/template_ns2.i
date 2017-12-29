// Tests compilation of uninstantiated templates in a namespace

%module template_ns2

%inline %{

namespace foo {
   template<class T> class bar {
   };
   bar<int> *test1(bar<int> *x) { return x; }
   typedef int Integer;

   bar<Integer *> *test2(bar<Integer *> *x) { return x; }
}
%}


%module("templatereduce") template_typedef_ptr

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Test<int, C*>; /* Ruby, wrong constant name */

 /*
   Use the "templatereduce" feature to force swig to reduce the template
   typedef as much as possible.

   This fixes cases like this one, but it can prevent some
   typemaps from working.
 */

%inline %{
 struct C{};
 typedef C* pC;

 template <class A, class B>
   struct Test 
   {
     Test (A a, B b)
     {
     }
     
   };

 
 template <class A, class B>
   struct Test<A, B*> 
   {
     Test (B* a)
     {
     }
     
   };
%}

  
%template(test_pC) Test<int, pC>;

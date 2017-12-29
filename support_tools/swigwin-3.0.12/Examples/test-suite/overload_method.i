/* This test confirms the fix to sourceforge bug #3478922 for R */

%module overload_method
%inline %{
class Base
   {
   public:

   Base() : x(42) {}

   int method() const { return x; }

   void overloaded_method(int aArg) { x = aArg; }
   int overloaded_method() const { return x; }

   private:

   int x;
   };
%}

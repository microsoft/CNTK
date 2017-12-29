/* Swig 1.3.6 does not understand initialization of static class
   constants like this.  SF Bug #445221, reported by Krzysztof
   Kozminski <kozminski@users.sf.net>. 
*/

%module static_const_member

#if SWIGJAVA
%javaconst(1) EN;
%javaconst(1) CHARTEST;
#elif SWIGCSHARP
%csconst(1) EN;
%csconst(1) CHARTEST;
#elif SWIGD
%dmanifestconst EN;
%dmanifestconst CHARTEST;
#endif

%inline %{
  
class X {
public:
  static const int PN = 0;
  static const int CN = 1;
  static const int EN = 2;
  static const char CHARTEST = 'A';
};

%}

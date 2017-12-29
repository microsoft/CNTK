%module arrays_scope

%inline %{

enum { ASIZE = 256 };
namespace foo {
   enum { BBSIZE = 512 };
   class Bar {
   public:
      enum { CCSIZE = 768 };
      int  adata[ASIZE];
      int  bdata[BBSIZE];
      int  cdata[CCSIZE];
      void blah(int x[ASIZE], int y[BBSIZE], int z[CCSIZE]) { };
   };
}

%}


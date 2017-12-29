%module conversion_ns_template
%{ 
 namespace oss 
 { 
   struct Hi
   {
     Hi(){}     
     Hi(int){}     
   };

   enum Test {One, Two}; 

   template <Test> 
   struct Foo { 
     Foo(){}
   }; 

   template <Test T>   
   struct Bar { 
     Bar(){ }
     Bar(int){ }
     
#if !defined(__SUNPRO_CC)
     operator int() { return 0; }
#endif
     operator int&() { static int num = 0; return num; }
#if !defined(__SUNPRO_CC)
     operator Foo<T>() { return Foo<T>(); }
#endif
     operator Foo<T>&() { return *(new Foo<T>()); }
   }; 
  } 
%} 

 namespace oss 
 { 
   enum Test {One, Two}; 
 
   // these works 
   %ignore Hi::Hi(); 
   %rename(create) Hi::Hi(int); 

   struct Hi 
   {
     Hi();
     Hi(int);
   };

   template <Test> 
   struct Foo { 
     Foo();
   }; 
 
   // these works 
   %rename(hello1) Bar<One>::operator int&(); 
   %ignore Bar<One>::operator int(); 
   %rename(hello2) Bar<One>::operator Foo<oss::One>&(); 
   %ignore Bar<One>::operator Foo<oss::One>();
    
   // these don't
   %ignore Bar<One>::Bar(); 
   %rename(Bar_create) Bar<One>::Bar(int); 
 
   template <Test T>   
   struct Bar { 
     Bar();
     Bar(int);
     operator int(); 
     operator int&(); 
     operator Foo<T>(); 
     operator Foo<T>&(); 
   }; 
  } 

  
namespace oss 
{ 
  %template(Foo_One) Foo<One>; 
  %template(Bar_One) Bar<One>; 
} 

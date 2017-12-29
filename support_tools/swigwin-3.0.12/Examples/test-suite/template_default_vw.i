%module template_default_vw

%inline %{
class SomeClass{ }; 
 
template<class T>  
class Handle { 
public: 
    Handle( T* t=0 ) { };   
    // ... 
}; 
 
typedef Handle<SomeClass> hSomeClass; 
class AnotherClass { 
public: 
  void someFunc( hSomeClass a = hSomeClass() ) { }; 
}; 

%}

%template() Handle<SomeClass>;




%module namespace_spaces 

%{
#if defined(__SUNPRO_CC)
#pragma error_messages (off, wbadasg) /* Assigning extern "C" ... */
#endif
%}

%inline %{ 
 
class Foo { 
public: 
    Foo(); 
    ~Foo(); 
    typedef int Integer; 
    int blah(int x); 
    int spam(int x); 
    Integer bar(Integer x); 
    void (Foo:: *func_ptr) (int);
}; 
 
inline Foo :: Foo () {}
 
inline Foo :: ~Foo () {}
     
inline int Foo  
::blah(int x) { return 2*x; } 
 
inline int Foo :: spam(int x) { return -x; } 
 
inline Foo :: Integer Foo :: bar(Foo :: Integer x) { return x; } 
 
%} 

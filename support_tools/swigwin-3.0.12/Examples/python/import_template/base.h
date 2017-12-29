#include <stdio.h>

template<class T> class Base {
 public:
     Base() { }
     virtual ~Base() { }
     virtual void A() {
         printf("I'm Base::A\n");
     }
     void B() {
       printf("I'm Base::B\n");
     }
     virtual Base<T> *toBase() {
       return static_cast<Base<T> *>(this);
     }
};
 
        

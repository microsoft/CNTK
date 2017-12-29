class Base {
 public:
     Base() { }
     virtual ~Base() { }
     virtual const char * A() const {
         return "Base::A";
     }
     const char * B() const {
       return "Base::B";
     }
     virtual Base *toBase() {
       return static_cast<Base *>(this);
     }
};
 
        

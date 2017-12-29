template<class T> class SmartPtr {
public:
   SmartPtr(T *realPtr = 0) { pointee = realPtr; }
   T *operator->() const {
       return pointee;
   }
   T &operator*() const {
      return *pointee;
   }
private:
   T *pointee;
};


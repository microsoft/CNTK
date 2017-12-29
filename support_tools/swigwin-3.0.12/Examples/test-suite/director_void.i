%module(directors="1") director_void

%warnfilter(SWIGWARN_TYPEMAP_DIRECTOROUT_PTR) voidPtrOut;

%feature("director") DirectorVoidPointer;

#if defined(SWIGCSHARP)
%apply void *VOID_INT_PTR { void * }
#endif

%inline %{
class DirectorVoidPointer {
  int *ptr;
public:
  DirectorVoidPointer(int val) : ptr(new int(val)) {}
  virtual ~DirectorVoidPointer() { delete ptr; }

  virtual void * voidPtrOut() { return ptr; }
  virtual int voidPtrIn(void *p) {
    return nonVirtualVoidPtrIn(p);
  }

  void setNewValue(int val) {
    delete ptr;
    ptr = new int(val);
  }
  void *nonVirtualVoidPtrOut() { return ptr; }
  int nonVirtualVoidPtrIn(void *p) {
    int val = *(int *)p;
    setNewValue(val + 100);
    return *ptr;
  }
};

struct Caller {
  int callVirtualIn(DirectorVoidPointer *d, int num) {
    return d->voidPtrIn(&num);
  }
  int callVirtualOut(DirectorVoidPointer *d) {
    return *(int *)d->voidPtrOut();
  }
  static int VoidToInt(void *p) {
    return *(int *)p;
  }
};
%}


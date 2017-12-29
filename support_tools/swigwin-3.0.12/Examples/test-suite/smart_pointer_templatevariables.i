%module smart_pointer_templatevariables

%inline %{
template <class _CharT>
struct basic_string {
    int npos;
};

template<class T>
struct Ptr {
    Ptr(T *p = 0) : ptr(p) {}
    ~Ptr() { delete ptr; }
    T *operator->() const { return ptr; }
private:
    T *ptr;
};

template <typename KernelPixelT>
struct DiffImContainer {
    int id;
// static members seem to be can of worms. Note that SWIG wraps them as non-static members. Why?
// Note CHANGES entry 10/14/2003. Static const variables are not wrapped as constants but as a read only variable. Why?
//    static short xyz;
//    static const short constvar = 555;
};
//template<typename KernelPixelT> short DiffImContainer<KernelPixelT>::xyz = 0;

DiffImContainer<double>* create(int id, short xyz) { 
  DiffImContainer<double> *d = new DiffImContainer<double>();
  d->id = id;
//  DiffImContainer<double>::xyz = xyz;
  return d;
}
%}

%template(BasicString)                     basic_string<char>;
%template(DiffImContainer_D)               DiffImContainer<double>;
%template(DiffImContainerPtr_D)            Ptr<DiffImContainer<double> >;


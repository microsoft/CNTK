%module smart_pointer_templatemethods

%inline %{
namespace ns {
 
template <typename T>
class Ptr
{
public:
  Ptr () {}
  T *operator -> () { return 0; }
};
 
typedef unsigned short uint16_t;
class InterfaceId
{
public:
  InterfaceId (uint16_t iid) {}
  InterfaceId() {}
};
 
template <typename K> class Objekt
{
public:
  Objekt () {}
  virtual ~Objekt () {}
  Ptr<K> QueryInterface (InterfaceId iid) const { return Ptr<K>(); }
  void DisposeObjekt (void) {}
};

class Objct
{
public:
  Objct () {}
  virtual ~Objct () {}
  template <typename T> Ptr<T> QueryInterface (InterfaceId iid) const { return Ptr<T>(); }
  void DisposeObjct (void) {}
};
 
#ifdef SWIG
%template(PtrObjct) Ptr<Objct>;
%template(PtrInt) Ptr<int>;
%template(ObjektInt) Objekt<int>;
%template(PtrObjektInt) Ptr<Objekt<int> >;
%template(QueryInterfaceObjct) Objct::QueryInterface<Objct>;
#endif

} // namespace
 
%}


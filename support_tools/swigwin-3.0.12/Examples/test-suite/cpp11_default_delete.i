/* This testcase checks whether SWIG correctly parses C++11 explicitly defaulted functions and deleted functions */
%module cpp11_default_delete

%warnfilter(SWIGWARN_LANG_OVERLOAD_IGNORED, SWIGWARN_LANG_OVERLOAD_SHADOW) trivial::trivial(trivial&&);
%warnfilter(SWIGWARN_LANG_OVERLOAD_IGNORED, SWIGWARN_LANG_OVERLOAD_SHADOW) trivial::operator =(trivial&&);

%rename(Assignment) *::operator=;

%inline %{

class NonCopyable {
public:
  NonCopyable& operator=(const NonCopyable&) = delete; /* Removes operator= */
  NonCopyable(const NonCopyable&) = delete; /* Removed copy constructor */
  NonCopyable() = default; /* Explicitly allows the empty constructor */
  void *operator new(size_t) = delete; /* Removes new NonCopyable */
};

struct A1 {
  void func(int i) {}
  A1() = default;
  ~A1() = default;
  void func(double i) = delete;  /* Don't cast double to int. Compiler returns an error */
private:
  A1(const A1&);
};
A1::A1(const A1&) = default;

struct A2 {
  void func(int i) {}
  virtual void fff(int) = delete;
  virtual ~A2() = default;
  template<class T> void func(T) = delete;
};

struct trivial {
  trivial() = default;
  trivial(const trivial&) = default;
  trivial(trivial&&) = default;
  trivial& operator=(const trivial&) = default;
  trivial& operator=(trivial&&) = default;
  ~trivial() = default;
};

struct nontrivial1 {
  nontrivial1();
};
nontrivial1::nontrivial1() = default;

struct sometype {
  sometype() = delete;
  sometype(int) = delete;
  sometype(double);
};
sometype::sometype(double) {}

/* Not working with prerelease of gcc-4.8
struct nonew {
  void *operator new(std::size_t) = delete;
  void *operator new[](std::size_t) = delete;
};
*/

struct moveonly {
  moveonly() = default;
  moveonly(const moveonly&) = delete;
  moveonly(moveonly&&) = default;
  moveonly& operator=(const moveonly&) = delete;
  moveonly& operator=(moveonly&&) = default;
  ~moveonly() = default;
};

struct ConstructorThrow {
  ConstructorThrow() throw() = default;
  ConstructorThrow(const ConstructorThrow&) throw() = delete;
  ConstructorThrow(ConstructorThrow&&) throw() = delete;
  ConstructorThrow& operator=(const ConstructorThrow&) throw() = delete;
  ~ConstructorThrow() throw() = default;
};

%}

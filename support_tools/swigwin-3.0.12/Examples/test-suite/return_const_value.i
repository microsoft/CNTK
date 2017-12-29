/* This test a return by value constant SWIGTYPE.
It was reported in bug 899332 by Jermey Brown (jhbrown94) */

%module return_const_value


%ignore Foo_ptr::operator=(const Foo_ptr&);

%inline %{

class Foo {
public:
    int _val;
    Foo(int x): _val(x) {}
    int getVal() const {
        return _val;
    }
};

class Foo_ptr {
    Foo *_ptr;
    mutable bool _own;
  
public:
  Foo_ptr(Foo *p, bool own = false): _ptr(p), _own(own) {}
  static Foo_ptr getPtr() {
    return Foo_ptr(new Foo(17), true);
  }
  static const Foo_ptr getConstPtr() {
    return Foo_ptr(new Foo(17), true);
  }
  const Foo *operator->() {
    return _ptr;
  }

  Foo_ptr(const Foo_ptr& f) : _ptr(f._ptr), _own(f._own) 
  {
    f._own = 0;
  }

  Foo_ptr& operator=(const Foo_ptr& f) {
    _ptr = f._ptr;
    _own = f._own;
    f._own = 0;
    return *this;
  }

  ~Foo_ptr() {
    if(_own) delete _ptr;
  }  
};

%}

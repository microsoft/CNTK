%module smart_pointer_const_overload

%warnfilter(SWIGWARN_LANG_OVERLOAD_IGNORED) Bar::operator->;      // Overloaded method Bar::operator ->() ignored
%warnfilter(SWIGWARN_LANG_OVERLOAD_IGNORED) Bar2::operator->;     // Overloaded method Bar2::operator ->() ignored

%inline %{
int CONST_ACCESS = 1;
int MUTABLE_ACCESS = 2;

int *new_int(int ivalue) {
  int *i = (int *) malloc(sizeof(ivalue));
  *i = ivalue;
  return i;
}

int get_int(int *i) {
  return *i;
}

void set_int(int *i, int ivalue) {
  *i = ivalue;
}

void delete_int(int *i) {
  free(i);
}

struct Foo {
   int x;
   int * const xp;
   const int y;
   const int *yp;
   int access;
   Foo() : x(0), xp(&x), y(0), yp(&y), access(0) { }
   int getx() const { return x; }
   void setx(int x_) { x = x_; }
   static void statMethod() {}
};
%}

%extend Foo {
   int getx2() const { return self->x; }
   void setx2(int x_) { self->x = x_; }
};

%inline %{
class Bar {
   Foo *f;
public:
   Bar(Foo *f) : f(f) { }
   const Foo *operator->() const {
       f->access = CONST_ACCESS;
       return f;
   }
   Foo *operator->() {
       f->access = MUTABLE_ACCESS;
       return f;
   }
};

class Bar2 {
   Foo *f;
public:
   Bar2(Foo *f) : f(f) { }
   Foo *operator->() {
       f->access = MUTABLE_ACCESS;
       return f;
   }
   const Foo *operator->() const {
       f->access = CONST_ACCESS;
       return f;
   }
};
%}

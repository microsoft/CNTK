/* File: template_basic.i */
%module "template_basic"

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) vector<int>;         /* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) vector<double>;      /* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) vector<int (*)[10]>; /* Ruby, wrong class name */

/* Let's just grab the original header file here */

%{
#ifdef max
#undef max
#endif
%}

%inline %{

template<class T> T max(const T a, const T b) { return  a>b ? a : b; }

template<class T> class vector {
  T *v;
  int sz;
 public:
  vector(int _sz) {
    v = new T[_sz];
    sz = _sz;
  }
  T &get(int index) {
    return v[index];
  }
  void set(int index, T &val) {
    v[index] = val;
  }
  // This really doesn't do anything except test const handling
  void testconst(const T x) { }
};

%}

/* Now instantiate some specific template declarations */

%template(maxint) max<int>;
%template(maxdouble) max<double>;
%template(vecint) vector<int>;
%template(vecdouble) vector<double>;

/* Now try to break constness */

%template(maxintp) max<int (*)[10]>;
%template(vecintp) vector<int (*)[10]>;



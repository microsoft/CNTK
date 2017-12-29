%module using_inherit

#ifdef SWIGLUA	// lua only has one numeric type, so some overloads shadow each other creating warnings
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) Foo::test;
#endif

%inline %{

class Foo {
public:
     int test(int x) { return x; }
     double test(double x) { return x; };
};

class Bar : public Foo {
public:
     using Foo::test;
};

class Bar2 : public Foo {
public:
     int test(int x) { return x*2; }
     double test(double x) { return x*2; };
     using Foo::test;
};

class Bar3 : public Foo {
public:
     int test(int x) { return x*2; }
     double test(double x) { return x*2; };
     using Foo::test;
};

class Bar4 : public Foo {
public:
     int test(int x) { return x*2; }
     using Foo::test;
     double test(double x) { return x*2; };
};

class Fred1 : public Foo {
public:
     using Foo::test;
     double test(double x) { return x*2; };
};

class Fred2 : public Foo {
public:
     double test(double x) { return x*2; };
     using Foo::test;
};

%}


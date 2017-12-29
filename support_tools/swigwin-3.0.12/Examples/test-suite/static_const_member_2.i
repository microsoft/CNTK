%module static_const_member_2

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) oss::modules::CavityPackFlags::forward_field;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) oss::modules::CavityPackFlags::backward_field;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) oss::modules::Test::current_profile;

%inline %{ 
 namespace oss 
 {   
   namespace modules
   {
     struct CavityPackFlags 
     {
       typedef unsigned int viewflags;
       static const viewflags forward_field  = 1 << 0;
       static const viewflags backward_field = 1 << 1;
       static const viewflags cavity_flags;
       static viewflags flags;
       static const int &reftest;
     };     

     template <class T>
     struct Test : CavityPackFlags
     {
       enum {LeftIndex, RightIndex};
       static const viewflags current_profile  = 1 << 2;
     };
   }
 }

%} 

%{

int refvalue = 42;
const int &oss::modules::CavityPackFlags::reftest = refvalue;

%}

%{
  using oss::modules::CavityPackFlags;
  
  const CavityPackFlags::viewflags 
    CavityPackFlags::cavity_flags = 
    CavityPackFlags::forward_field | CavityPackFlags::backward_field;

  CavityPackFlags::viewflags 
    CavityPackFlags::flags = 0;

%}

%template(Test_int) oss::modules::Test<int>;




%inline %{

class Foo
{
public:
  int val;
  
  Foo(int v) : val(v)
  {
  }
  
  static const Foo BAR;
  static const Foo BAZ;
};

%}
%{
  const Foo Foo::BAR = Foo(1);
  const Foo Foo::BAZ = Foo(2);
%}

// Tests passing of references to primitive datatypes
%module primitive_ref

%define ref(type,name)
%inline %{
const type &ref_##name(const type &x) {
   static type y = x;
   return y;
}
%}
%enddef

ref(int,int);
ref(unsigned int, uint);
ref(short, short);
ref(unsigned short, ushort);
ref(long,long);
ref(unsigned long, ulong);
ref(signed char, schar);
ref(unsigned char, uchar);
ref(char, char);
ref(float, float);
ref(double, double);
ref(bool, bool);
ref(long long, longlong);
ref(unsigned long long, ulonglong);


%inline %{

int ref_over(int a)
{
  return a;
}

 struct A 
 {
   int v;
   A(int V) :v(V) {}
 };
 
int ref_over(const A& a)
{
  return a.v;
}

%}

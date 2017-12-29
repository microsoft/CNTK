%module constructor_ignore

%ignore Space::Delta1::Delta1();
%ignore Space::Delta2::Delta2(int i);
%ignore Space::Delta3::Delta3;
%ignore Space::Delta4::Delta4;

%inline %{
namespace Space {
  struct Delta1 {
  };
  struct Delta2 {
    Delta2(int i) {}
  };
  struct Delta3 {
    Delta3(const Delta3&) {}
    Delta3() {}
    Delta3(int i) {}
  };
  struct Delta4 {
  };
}
%}

%copyctor;
%ignore Space::Delta5::Delta5;
%ignore Space::Delta6::Delta6(const Space::Delta6&);

%inline %{
namespace Space {
  struct Delta5 {
  };
  struct Delta6 {
  };
}
%}

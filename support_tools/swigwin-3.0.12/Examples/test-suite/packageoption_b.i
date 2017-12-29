%module(package="CommonPackage") "packageoption_b";

%inline %{
class B
{
 public:
  int testInt() { return 4; }
};

%}

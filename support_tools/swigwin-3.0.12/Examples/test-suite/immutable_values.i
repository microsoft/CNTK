// test to make sure setters are not generated for constants

%module immutable_values


%immutable;
%mutable;

%inline %{
#define ABC -11
enum count {Zero, One, Two}; %}


%clearimmutable;

%inline %{
#define XYZ -22
enum backwards {Tre=3, Duo=2, Uno=1};
%}


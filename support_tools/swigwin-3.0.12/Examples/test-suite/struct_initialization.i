// Test declaration and initialization of structs (C code)
%module struct_initialization

%inline %{

/* Named types */
struct StructA {
   int x;
} instanceA1;

struct StructB {
   int x;
} instanceB1, instanceB2, instanceB3;

struct StructC {
   int x;
} instanceC1 = { 10 };

struct StructD {
   int x;
} instanceD1 = { 10 }, instanceD2 = { 20 }, instanceD3 = { 30 };

struct StructE {
   int x;
} instanceE1[3] = { { 1 }, { 2 }, { 3} };

struct StructF {
   int x;
} instanceF1[3] = { { 1 }, { 2 } }, instanceF2[2] = { { -1 }, { -2 } }, instanceF3[2] = { { 11 }, { 22 } };

%}

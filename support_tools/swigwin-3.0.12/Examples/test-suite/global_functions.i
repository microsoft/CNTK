%module global_functions

%inline %{
void global_void(void) {}
int global_one(int i) { return i; }
int global_two(int i, int j) { return i+j; }
%}


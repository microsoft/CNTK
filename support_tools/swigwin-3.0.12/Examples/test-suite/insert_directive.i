%module insert_directive

// check %insert and the order of each insert section is correct

%begin %{
// %inserted code %begin
int inserted_begin(int i) { return i; }
%}

%runtime %{
// %inserted code %runtime
int inserted_runtime(int i) { return inserted_begin(i); }
%}

%{
// %inserted code %header
int inserted_header1(int i) { return inserted_runtime(i); }
%}

%header %{
// %inserted code %header
int inserted_header2(int i) { return inserted_header1(i); }
%}

%{
// %inserted code %header
int inserted_header3(int i) { return inserted_header2(i); }
%}

%header "insert_directive.h"

%wrapper %{
// %inserted code %wrapper
int inserted_wrapper(int i) { return inserted_header4(i); }
%}

%init %{
// %inserted code %init
int SWIGUNUSED inserted_init_value = inserted_wrapper(0);
%}

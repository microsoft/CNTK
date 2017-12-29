/* File : example.i */
%module example

/* in this file there are two sorting functions
and three different ways to wrap them.

See the lua code for how they are called
*/

%include <carrays.i>    // array helpers

// this declares a batch of function for manipulating C integer arrays
%array_functions(int,int)

// this adds some lua code directly into the module
// warning: you need the example. prefix if you want it added into the module
// admittedly this code is a bit tedious, but its a one off effort
%luacode {
function example.sort_int2(t)
-- local len=table.maxn(t) -- the len - maxn deprecated in 5.3
 local len=0; for _ in pairs(t) do len=len+1 end
 local arr=example.new_int(len)
 for i=1,len do
  example.int_setitem(arr,i-1,t[i]) -- note: C index is one less then lua index
 end
 example.sort_int(arr,len) -- call the fn
 -- copy back
 for i=1,len do
  t[i]=example.int_getitem(arr,i-1) -- note: C index is one less then lua index
 end
 example.delete_int(arr) -- must delete it
end
}

// this way uses the SWIG-Lua typemaps to do the conversion for us
// the %apply command states to apply this wherever the argument signature matches
%include <typemaps.i>
%apply (double *INOUT,int) {(double* arr,int len)};

%inline %{
extern void sort_int(int* arr, int len);
extern void sort_double(double* arr, int len);
%}

%module typemap_arrays

// Test that previously non-working array typemaps special variables are working

%typemap(in) SWIGTYPE[ANY] {
  _should_not_be_used_and_will_not_compile_
}

// Check $basemangle expands to _p_int and $basetype expands to int *
%typemap(in) int *nums[3] (int *temp[3]) {
  $basetype var1$basemangle = new int(10);
  $basetype var2$basemangle = new int(20);
  $basetype var3$basemangle = new int(30);
  temp[0] = var1_p_int;
  temp[1] = var2_p_int;
  temp[2] = var3_p_int;
  $1 = temp;
}

%inline %{
int sumA(int *nums[3]) {
  int sum = 0;
  for (int i=0; i<3; ++i) {
    int *p = nums[i];
    if (p)
      sum += *p;
  }
  return sum;
}
%}

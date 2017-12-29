%module xxx

%typemap(in, warning="1000:Test warning for 'in' typemap for $1_type $1_name ($1) - argnum: $argnum &1_ltype: $&1_ltype descriptor: $1_descriptor") int ""

%typemap(out, warning="1001:Test warning for 'out' typemap for $1_type $1_name ($1) - name: $name symname: $symname &1_ltype: $&1_ltype descriptor: $1_descriptor") double ""

double mmm(int abc, short def, int);

%module xxx

%pythoncode %{
    def foo():
	a = 1 # This line starts with a tab instead of 8 spaces.
        return 2
%}

%module xxx

%pythoncode %{
    def one():
        print "in one"
%}

%pythoncode %{
        print "still in one"

    def two():
        print "in two"
%}

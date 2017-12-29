%module xxx

%pythoncode %{
  def extra():
  	print "extra a" # indentation is 2 spaces then tab
	  print "extra b" # indentation is tab then 2 spaces
%}

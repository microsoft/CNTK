exec("swigtest.start", -1);

try
    Spam = new_Spam()
catch
    swigtesterror();
end
  
if Foo_blah(Spam)<>0 then swigtesterror; end

exec("swigtest.quit", -1);

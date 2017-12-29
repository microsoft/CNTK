exec("swigtest.start", -1);

try
    // This call must fail because the constructor does not exist
    Spam = new_Spam()
    swigtesterror();
catch
end

exec("swigtest.quit", -1);

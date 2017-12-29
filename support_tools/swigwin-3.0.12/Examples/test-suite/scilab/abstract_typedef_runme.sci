exec("swigtest.start", -1);

try
    e = new_Engine();
catch
    swigtesterror();
end

try
    a = new_A();
catch
    swigtesterror();
end

// TODO: test write method

exec("swigtest.quit", -1);

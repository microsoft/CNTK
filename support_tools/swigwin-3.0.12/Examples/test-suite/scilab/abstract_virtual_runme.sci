exec("swigtest.start", -1);

try
    d = new_D();
catch
    swigtesterror();
end

try
    delete_D(d);
catch
    swigtesterror();
end

try
    e = new_E();
catch
    swigtesterror();
end

try
    delete_E(e);
catch
    swigtesterror();
end

exec("swigtest.quit", -1);

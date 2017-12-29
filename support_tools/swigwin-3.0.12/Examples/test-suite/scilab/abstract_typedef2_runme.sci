exec("swigtest.start", -1);

try
    a = new_A_UF();
catch
    swigtesterror();
end

try
    delete_A_UF(a);
catch
    swigtesterror();
end

exec("swigtest.quit", -1);

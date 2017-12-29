exec("swigtest.start", -1);

try
    D = new_D();
catch
    swigtesterror();
end
if A_do_x(D) <> 1 then swigtesterror(); end

try
    delete_D(D);
catch
    swigtesterror();
end

exec("swigtest.quit", -1);

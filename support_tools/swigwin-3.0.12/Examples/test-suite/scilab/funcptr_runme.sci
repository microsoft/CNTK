exec("swigtest.start", -1);

if add(7, 9) <> 16 then swigtesterror(); end
if do_op(7, 9, funcvar_get()) <> 16 then swigtesterror(); end

exec("swigtest.quit", -1);

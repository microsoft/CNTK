exec("swigtest.start", -1);

try
   x = fmod(M_PI_get(), M_1_PI_get())
catch
    swigtesterror();
end

exec("swigtest.quit", -1);
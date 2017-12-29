exec("swigtest.start", -1);

try
    a = new_Bar();
    Bar_x_set(a,100);
catch
    swigtesterror();
end
if Bar_x_get(a) <> 100 then swigtesterror(); end

exec("swigtest.quit", -1);

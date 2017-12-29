exec("swigtest.start", -1);

try
    x = new_LineObj();
    LineObj_numpoints_set(x, 100);
catch
    swigtesterror();
end
if LineObj_numpoints_get(x) <> 100 then swigtesterror(); end

if MS_NOOVERRIDE_get() <> -1111 then swigtesterror(); end

try
    y = make_a();
    A_t_a_set(y, 200);
catch
    swigtesterror();
end
if A_t_a_get(y) <> 200 then swigtesterror(); end

try
    A_t_b_set(y, 300);
catch
    swigtesterror();
end
if A_t_b_get(y) <> 300 then swigtesterror(); end

exec("swigtest.quit", -1);


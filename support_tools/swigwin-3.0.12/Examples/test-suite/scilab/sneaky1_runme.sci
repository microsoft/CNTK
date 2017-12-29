exec("swigtest.start", -1);

try
    x = add(3, 4);
catch
    swigtesterror();
end
if x <> 7 then swigtesterror(); end

try
    y = subtract(3,4);
catch
    swigtesterror();
end
if y <> -1 then swigtesterror(); end

try
    z = mul(3,4);
catch
    swigtesterror();
end
if z <> 12 then swigtesterror(); end

try
    w = divide(3,4);
catch
    swigtesterror();
end
if w <> 0 then swigtesterror(); end

exec("swigtest.quit", -1);

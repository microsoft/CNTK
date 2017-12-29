exec("swigtest.start", -1);

try
    small = new_SmallStruct();
    SmallStruct_jill_set(small, 200);

    big = new_BigStruct();
    BigStruct_jack_set(big, 300);

    Jill = SmallStruct_jill_get(small);
catch
    swigtesterror();
end
if Jill <> 200 then swigtesterror(); end

try
    Jack = BigStruct_jack_get(big);
catch
    swigtesterror();
end
if Jack <> 300 then swigtesterror(); end

exec("swigtest.quit", -1);

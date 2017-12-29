exec("swigtest.start", -1);

try
    baseInt = new_BaseInt();
catch
    swigtesterror();
end

try
    delete_BaseInt(baseInt);
catch
    swigtesterror();
end

try
    derivedInt = new_DerivedInt();
catch
    swigtesterror();
end

try
    delete_DerivedInt(derivedInt);
catch
    swigtesterror();
end

try
    bottomInt = new_BottomInt();
catch
    swigtesterror();
end

try
    delete_BottomInt(bottomInt);
catch
    swigtesterror();
end

exec("swigtest.quit", -1);

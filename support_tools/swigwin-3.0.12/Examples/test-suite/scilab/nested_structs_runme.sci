exec("swigtest.start", -1);

try
    outer = new_Outer();
    setValues(outer, 10);

    inner1 = Outer_inner1_get(outer);
    inner2 = Outer_inner2_get(outer);
    inner3 = Outer_inner3_get(outer);
    inner4 = Outer_inner4_get(outer);
catch
    swigtesterror();
end

checkequal(Outer_inner1_val_get(inner1), 10, "Outer_inner1_val_get(inner1)");
checkequal(Outer_inner1_val_get(inner2), 20, "Outer_inner1_val_get(inner2)");
checkequal(Outer_inner1_val_get(inner3), 20, "Outer_inner1_val_get(inner3)");
checkequal(Outer_inner1_val_get(inner4), 40, "Outer_inner1_val_get(inner4)");

try
    inside1 = Outer_inside1_get(outer);
    inside2 = Outer_inside2_get(outer);
    inside3 = Outer_inside3_get(outer);
    inside4 = Outer_inside4_get(outer);
catch
    swigtesterror();
end

checkequal(Named_val_get(inside1), 100, "Named_val_get(inside1)");
checkequal(Named_val_get(inside2), 200, "Named_val_get(inside2)");
checkequal(Named_val_get(inside3), 200, "Named_val_get(inside3)");
checkequal(Named_val_get(inside4), 400, "Named_val_get(inside4)");

exec("swigtest.quit", -1);

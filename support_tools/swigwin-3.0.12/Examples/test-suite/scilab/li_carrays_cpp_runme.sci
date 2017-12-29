exec("swigtest.start", -1);

d = new_intArray(10);

intArray_setitem(d, 0, 7);

intArray_setitem(d, 5, intArray_getitem(d, 0) + 3);

checkequal(intArray_getitem(d, 5) + intArray_getitem(d, 0), 17, "d(5) + d(0) <> 17");

exec("swigtest.quit", -1);

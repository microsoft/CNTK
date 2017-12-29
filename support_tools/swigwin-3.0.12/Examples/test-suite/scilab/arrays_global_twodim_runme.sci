exec("swigtest.start", -1);

a = [1, 2, 3, 4; 5, 6, 7, 8;]

//try
//    array_d_set(a);
//catch
//    swigtesterror();
//end
//if array_d_get() <> a then swigtesterror(); end

exec("swigtest.quit", -1);

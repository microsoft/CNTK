exec("swigtest.start", -1);

function testArray(arrayName, arraySetFunc, arrayGetFunc, in_values, ..
  expected_out_values)
  try
    arraySetFunc(in_values);
  catch
    swigtesterror("error in " + arrayName + "_set()");
  end
  try
    checkequal(arrayGetFunc(), expected_out_values, arrayName + "_get()");
  catch
    swigtesterror("error in " + arrayName + "_get()");
  end
endfunction

m = [-10, 20];
um = [10, 20];
testArray("array_c", array_c_set, array_c_get, ['ab'], ['ab']);
testArray("array_sc", array_sc_set, array_sc_get, m, m);
testArray("array_sc", array_sc_set, array_sc_get, int8(m), m);
testArray("array_uc", array_uc_set, array_uc_get, uint8(um), um);
testArray("array_s", array_s_set, array_s_get, m, m);
testArray("array_s", array_s_set, array_s_get, int16(m), m);
testArray("array_us", array_us_set, array_us_get, uint16(um), um);
testArray("array_i", array_i_set, array_i_get, m, m);
testArray("array_i", array_i_set, array_i_get, int32(m), m);
testArray("array_ui", array_ui_set, array_ui_get, uint32(um), um);
testArray("array_l", array_l_set, array_l_get, m, m);
testArray("array_l", array_l_set, array_l_get, int32(m), m);
testArray("array_ul", array_ul_set, array_ul_get, uint32(um), um);
testArray("array_f", array_f_set, array_f_get, [-2.5, 2.5], [-2.5, 2.5]);
testArray("array_d", array_d_set, array_d_get, [-10.5, 20.4], [-10.5, 20.4]);

checkequal(array_const_i_get(), [10, 20], "array_const_i_get()");

ierr = execstr('array_i_set([0:10]', 'errcatch');
if ierr == 0 then swigtesterror("Overflow error expected"); end

checkequal(BeginString_FIX44a_get(), "FIX.a.a", "BeginString_FIX44a_get()");
checkequal(BeginString_FIX44b_get(), "FIX.b.b", "BeginString_FIX44b_get()");
checkequal(BeginString_FIX44c_get(), "FIX.c.c", "BeginString_FIX44c_get()");
checkequal(BeginString_FIX44d_get(), "FIX.d.d", "BeginString_FIX44d_get()");
BeginString_FIX44b_set(strcat(["12","\0","45"]));
checkequal(BeginString_FIX44b_get(), "12\045", "BeginString_FIX44b_get()");
checkequal(BeginString_FIX44d_get(), "FIX.d.d", "BeginString_FIX44d_get()");
checkequal(BeginString_FIX44e_get(), "FIX.e.e", "BeginString_FIX44e_get()");
checkequal(BeginString_FIX44f_get(), "FIX.f.f", "BeginString_FIX44f_get()");

checkequal(test_a("hello","hi","chello","chi"), "hi", "test_a()");

checkequal(test_b("1234567","hi"), "1234567", "test_b()");

exec("swigtest.quit", -1);

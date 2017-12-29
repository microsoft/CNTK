exec("swigtest.start", -1);

function checkBool(bool_val, expected_bool_val)
  if typeof(bool_val) <> "boolean" then swigtesterror(); end
  if bool_val <> expected_bool_val then swigtesterror(); end
endfunction

checkBool(constbool_get(), %f);

checkBool(bool1_get(), %t);
checkBool(bool2_get(), %f);

checkBool(bo(%t), %t);
checkBool(bo(%f), %f);

bs = new_BoolSt();
checkBool(BoolSt_m_bool1_get(bs), %t);
checkBool(BoolSt_m_bool2_get(bs), %f);

exec("swigtest.quit", -1);

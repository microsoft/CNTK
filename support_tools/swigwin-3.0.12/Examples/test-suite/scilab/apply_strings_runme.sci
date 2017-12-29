exec("swigtest.start", -1);

testString = "Scilab test string";

checkequal(UCharFunction(testString), testString, "UCharFunction(testString)");
checkequal(SCharFunction(testString), testString, "SCharFunction(testString)");
checkequal(CUCharFunction(testString), testString, "CUCharFunction(testString)");
checkequal(CSCharFunction(testString), testString, "CSCharFunction(testString)");
//checkequal(CharFunction(testString), testString, "CharFunction(testString)");
//checkequal(CCharFunction(testString), testString, "CCharFunction(testString)");

try
  tNum = new_TNum();
catch
  swigtesterror();
end
//TNumber_DigitsMemberA_get()
//TNumber_DigitsMemberA_set
//TNumber_DigitsMemberB_get()
//TNumber_DigitsMemberB_set
try
  delete_TNum(tNum);
catch
  swigtesterror();
end

try
  dirTest = new_DirTest();
catch
  swigtesterror();
end

checkequal(DirTest_UCharFunction(dirTest, testString), testString, "DirTest_UCharFunction");
checkequal(DirTest_SCharFunction(dirTest, testString), testString, "DirTest_SCharFunction(dirTest, testString)");
checkequal(DirTest_CUCharFunction(dirTest, testString), testString, "DirTest_CUCharFunction(dirTest, testString)");
checkequal(DirTest_CSCharFunction(dirTest, testString), testString, "DirTest_CSharFunction(dirTest, testString)");
//checkequal(DirTest_CharFunction(dirTest, testString), testString, "DirTest_CharFunction(dirTest, testString)");
//checkequal(DirTest_CCharFunction(dirTest, testString), testString, "DirTest_CCharFunction(dirTest, testString)");
try
  delete_DirTest(dirTest);
catch
  swigtesterror();
end

exec("swigtest.quit", -1);

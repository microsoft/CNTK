exec("swigtest.start", -1);

smallnum = -127;
checkequal(CharValFunction(smallnum), smallnum, "CharValFunction(smallnum)");
checkequal(CCharValFunction(smallnum), smallnum, "CCharValFunction(smallnum)");
checkequal(CCharRefFunction(smallnum), smallnum, "CCharRefFunction(smallnum)");

try
  globalchar_set(smallnum);
catch
  swigtesterror();
end
checkequal(globalchar_get(), smallnum, "globalchar_get()");
checkequal(globalconstchar_get(), -110, "globalconstchar_get()");

try
  dirTest = new_DirTest();
catch
  swigtesterror();
end

checkequal(DirTest_CharValFunction(dirTest, smallnum), smallnum, "DirTest_CharValFunction(dirTest, smallnum)");
checkequal(DirTest_CCharValFunction(dirTest, smallnum), smallnum, "DirTest_CCharValFunction(dirTest, smallnum)");
checkequal(DirTest_CCharRefFunction(dirTest, smallnum), smallnum, "DirTest_CCharRefFunction(dirTest, smallnum)");

// TODO Too long identifiers
//if dirTest_memberchar_get(dirTest) <> -111 then swigtesterror(); end
//try
//  dirTest_memberchar_set(dirTest, smallnum)
//catch
//  swigtesterror();
//end
//if dirTest_memberchar_get(dirTest) <> smallnum then swigtesterror(); end

//if dirTest_memberconstchar_get(dirTest) <> -112 then swigtesterror(); end

try
  delete_DirTest(dirTest);
catch
  swigtesterror();
end

exec("swigtest.quit", -1);

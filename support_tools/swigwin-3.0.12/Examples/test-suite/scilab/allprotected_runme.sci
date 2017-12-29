exec("swigtest.start", -1);

// Class Klass
try
  klass = new_Klass("allprotected_klass")
catch
  swigtesterror(lasterror());
end

checkequal(Klass_getName(klass), "allprotected_klass",  "Klass_getName(new_Klass(""allprotected_klass""))");

// Class PubBase
try
  pubBase = new_PubBase("allprotected_PubBase");
catch
  swigtesterror(lasterror());
end

checkequal(PubBase_virtualMethod(pubBase), "PublicBase", "PubBase_virtualMethod(pubBase)");

class = PubBase_instanceMethod(pubBase, klass);
checkequal(Klass_getName(class), "allprotected_klass", "Klass_getName(PubBase_instanceMethod(pubBase, klass))");

class = PubBase_instanceOverload(pubBase, klass);
checkequal(Klass_getName(class), "allprotected_klass", "Klass_getName(PubBase_instanceOverloaded(pubBase, klass))");

class = PubBase_instanceOverload(pubBase, klass, "allprotected_klass2");
checkequal(Klass_getName(class), "allprotected_klass2", "Klass_getName(PubBase_instanceOverloaded(pubBase, klass, ""allprotected_klass2""))");

class = PubBase_staticMethod(klass);
checkequal(Klass_getName(class), "allprotected_klass", "Klass_getName(PubBase_staticMethod(klass))");

class = PubBase_staticOverloaded(klass);
checkequal(Klass_getName(class), "allprotected_klass", "Klass_getName(PubBase_staticOverloaded(klass))");


class = PubBase_staticOverloaded(klass, "allprotected_klass3");
checkequal(Klass_getName(class), "allprotected_klass3", "Klass_getName(PubBase_staticOverloaded(klass, ""allprotected_klass3""))");

checkequal(PubBase_EnumVal1_get(), 0, "PubBase_EnumVal1_get()");
checkequal(PubBase_EnumVal2_get(), 1, "(PubBase_EnumVal2_get()");


PubBase_instanceMemb_set(pubBase, 12);
checkequal(PubBase_instanceMemb_get(pubBase), 12, "PubBase_instanceMemb_get(pubBase)");

checkequal(PubBase_staticConstM_get(), 20, "PubBase_staticConstM_get()");
checkequal(PubBase_staticMember_get(), 10, "PubBase_staticMember_get()")

PubBase_stringMember_set(pubBase, "dummy");
checkequal(PubBase_stringMember_get(pubBase), "dummy", "PubBase_stringMember_get()");

// TODO does not work (wrong ENUM mapping?)
//PubBase_anEnum_get(PubBase)
//PubBase_anEnum_set(PubBase, ???)


// Class ProcBase
try
// Constructor is propected and must not be defined here
  ProcBase = new_ProctectedBase("allprotected_ProcBase");
  swigtesterror();
catch
end

checkequal(ProcBase_EnumVal1_get(), 0, "ProcBase_EnumVal1_get()");
checkequal(ProcBase_EnumVal2_get(), 1, "ProcBase_EnumVal2_get()");

try
  delete_Klass(klass);
catch
  swigtesterror();
end
try
  delete_PubBase(pubBase);
catch
  swigtesterror();
end

exec("swigtest.quit", -1);

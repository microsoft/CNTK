using System;
using special_variable_macrosNamespace;

public class runme {
  static void Main() {
    Name name = new Name();
    if (special_variable_macros.testFred(name) != "none")
      throw new Exception("test failed");
    if (special_variable_macros.testJack(name) != "$specialname")
      throw new Exception("test failed");
    if (special_variable_macros.testJill(name) != "jilly")
      throw new Exception("test failed");
    if (special_variable_macros.testMary(name) != "SWIGTYPE_p_NameWrap")
      throw new Exception("test failed");
    if (special_variable_macros.testJames(name) != "SWIGTYPE_Name")
      throw new Exception("test failed");
    if (special_variable_macros.testJim(name) != "multiname num")
      throw new Exception("test failed");
    if (special_variable_macros.testJohn(new PairIntBool(10, false)) != 123)
      throw new Exception("test failed");
    NewName newName = NewName.factory("factoryname");
    name = newName.getStoredName();
  }
}

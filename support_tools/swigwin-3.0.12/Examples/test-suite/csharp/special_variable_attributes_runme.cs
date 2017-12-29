
using System;
using special_variable_attributesNamespace;

public class special_variable_attributes_runme {

  public static void Main() {
    if (special_variable_attributes.getNumber1() != 111)
      throw new ApplicationException("getNumber1 failed");
    if (special_variable_attributes.getNumber2() != 222)
      throw new ApplicationException("getNumber2 failed");
    if (special_variable_attributes.getNumber3() != 333)
      throw new ApplicationException("getNumber3 failed");

    if (special_variable_attributes.bounceNumber1(10) != 110)
      throw new ApplicationException("bounceNumber1 failed");
    if (special_variable_attributes.bounceNumber2(10) != 220)
      throw new ApplicationException("bounceNumber2 failed");
    if (special_variable_attributes.bounceNumber3(10) != 330)
      throw new ApplicationException("bounceNumber3 failed");

    if (special_variable_attributes.multi1(12.34) != 12+34)
      throw new ApplicationException("multi1 failed");
    if (special_variable_attributes.multi2(12.34) != 12+34+55)
      throw new ApplicationException("multi2 failed");
    if (special_variable_attributes.multi3(12.34) != 12+34+77)
      throw new ApplicationException("multi3 failed");
  }

}

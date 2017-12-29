using System;
using System.Collections.Generic;
using multiple_inheritance_interfacesNamespace;

public class multiple_inheritance_interfaces_runme {

  static string SortArrayToString(string[] types) {
    Array.Sort<string>(types);
    return string.Join(" ", types);
  }

  static string SortArrayToString(Type[] types) {
    List<string> stypes = new List<string>();
    foreach (Type t in types)
      stypes.Add(t.Name);
    return SortArrayToString(stypes.ToArray());
  }


  private static void checkBaseAndInterfaces(Type cls, bool interfaceExpected, string baseClass, string[] interfaces) {
    string[] expectedInterfaces = new string[interfaces.Length + (interfaceExpected ? 0 : 1)];
    for (int i=0; i<interfaces.Length; ++i)
      expectedInterfaces[i] = interfaces[i];
    if (!interfaceExpected)
      expectedInterfaces[interfaces.Length] = "IDisposable";
    Type[] actualInterfaces = cls.GetInterfaces();
    string expectedInterfacesString = SortArrayToString(expectedInterfaces);
    string actualInterfacesString = SortArrayToString(actualInterfaces);
    if (expectedInterfacesString != actualInterfacesString)
      throw new Exception("Expected interfaces for " + cls.Name + ": \n" + expectedInterfacesString + "\n" + "Actual interfaces: \n" + actualInterfacesString);

    string expectedBaseString = null;
    if (interfaceExpected) {
      // expecting an interface
      if (!cls.IsInterface)
        throw new Exception(cls.Name + " should be an interface but is not");
      expectedBaseString = string.IsNullOrEmpty(baseClass) ? "" : "multiple_inheritance_interfacesNamespace." + baseClass;
    } else {
      // expecting a class
      if (cls.IsInterface)
        throw new Exception(cls.Name + " is an interface but it should not be");
      expectedBaseString = string.IsNullOrEmpty(baseClass) ? "Object" : baseClass;
    }

    string actualBaseString = cls.BaseType == null ? "" : cls.BaseType.Name;
    if (expectedBaseString != actualBaseString)
      throw new Exception("Expected base for " + cls.Name + ": [" + expectedBaseString + "]" + " Actual base: [" + actualBaseString + "]");
  }

  public static void Main() {
    // Note that we can't get just the immediate interface
    // Type.GetInterfaces() returns all interfaces up the inheritance hierarchy
    checkBaseAndInterfaces(typeof(IA), true, "", new string[] {});
    checkBaseAndInterfaces(typeof(IB), true, "", new string[] {});
    checkBaseAndInterfaces(typeof(IC), true, "", new string[] {"IA", "IB"});
    checkBaseAndInterfaces(typeof(A), false, "", new string[] {"IA"});
    checkBaseAndInterfaces(typeof(B), false, "", new string[] {"IB"});
    checkBaseAndInterfaces(typeof(C), false, "", new string[] {"IA", "IB", "IC"});
    checkBaseAndInterfaces(typeof(D), false, "", new string[] {"IA", "IB", "IC"});
    checkBaseAndInterfaces(typeof(E), false, "D", new string[] {"IA", "IB", "IC"});

    checkBaseAndInterfaces(typeof(IJ), true, "", new string[] {});
    checkBaseAndInterfaces(typeof(IK), true, "", new string[] {"IJ"});
    checkBaseAndInterfaces(typeof(IL), true, "", new string[] {"IJ", "IK"});
    checkBaseAndInterfaces(typeof(J), false, "", new string[] {"IJ"});
    checkBaseAndInterfaces(typeof(K), false, "", new string[] {"IJ", "IK"});
    checkBaseAndInterfaces(typeof(L), false, "", new string[] {"IJ", "IK", "IL"});
    checkBaseAndInterfaces(typeof(M), false, "", new string[] {"IJ", "IK", "IL"});

    checkBaseAndInterfaces(typeof(P), false, "", new string[] {});
    checkBaseAndInterfaces(typeof(IQ), true, "", new string[] {});
    checkBaseAndInterfaces(typeof(Q), false, "", new string[] {"IQ"});
    checkBaseAndInterfaces(typeof(R), false, "P", new string[] {"IQ"});
    checkBaseAndInterfaces(typeof(S), false, "P", new string[] {"IQ"});
    checkBaseAndInterfaces(typeof(T), false, "", new string[] {"IQ"});
    checkBaseAndInterfaces(typeof(U), false, "R", new string[] {"IQ"});
    checkBaseAndInterfaces(typeof(V), false, "S", new string[] {"IQ"});
    checkBaseAndInterfaces(typeof(W), false, "T", new string[] {"IQ"});

    // overloaded methods check
    D d = new D();
    d.ia();
    d.ia(10);
    d.ia("bye");
    d.ia("bye", false);
  }
}

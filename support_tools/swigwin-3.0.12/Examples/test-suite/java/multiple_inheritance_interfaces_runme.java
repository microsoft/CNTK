import multiple_inheritance_interfaces.*;
import java.util.Arrays;

public class multiple_inheritance_interfaces_runme {

  static {
    try {
      System.loadLibrary("multiple_inheritance_interfaces");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  private static void checkBaseAndInterfaces(Class cls, boolean interfaceExpected, String base, String[] interfaces) {
    String[] expectedInterfaces = new String[interfaces.length];
    for (int i=0; i<interfaces.length; ++i)
      expectedInterfaces[i] = "interface multiple_inheritance_interfaces." + interfaces[i];
    Class[] actualInterfaces = cls.getInterfaces();
    String expectedInterfacesString = Arrays.toString(expectedInterfaces);
    String actualInterfacesString = Arrays.toString(actualInterfaces);
    if (!expectedInterfacesString.equals(actualInterfacesString))
      throw new RuntimeException("Expected interfaces for " + cls.getName() + ": \n" + expectedInterfacesString + "\n" + "Actual interfaces: \n" + actualInterfacesString);

    String expectedBaseString = null;
    if (interfaceExpected) {
      // expecting an interface
      if (!cls.isInterface())
        throw new RuntimeException(cls.getName() + " should be an interface but is not");
      expectedBaseString = base.isEmpty() ? "" : "multiple_inheritance_interfaces." + base;
    } else {
      // expecting a class
      if (cls.isInterface())
        throw new RuntimeException(cls.getName() + " is an interface but it should not be");
      expectedBaseString = base.isEmpty() ? "java.lang.Object" : "multiple_inheritance_interfaces." + base;
    }

    String actualBaseString = cls.getSuperclass() == null ? "" : cls.getSuperclass().getName();
    if (!expectedBaseString.equals(actualBaseString))
      throw new RuntimeException("Expected base for " + cls.getName() + ": [" + expectedBaseString + "]" + " Actual base: [" + actualBaseString + "]");
  }

  public static void main(String argv[]) {
    checkBaseAndInterfaces(IA.class, true, "", new String[] {});
    checkBaseAndInterfaces(IB.class, true, "", new String[] {});
    checkBaseAndInterfaces(IC.class, true, "", new String[] {"IA", "IB"});
    checkBaseAndInterfaces(A.class, false, "", new String[] {"IA"});
    checkBaseAndInterfaces(B.class, false, "", new String[] {"IB"});
    checkBaseAndInterfaces(C.class, false, "", new String[] {"IA", "IB", "IC"});
    checkBaseAndInterfaces(D.class, false, "", new String[] {"IA", "IB", "IC"});
    checkBaseAndInterfaces(E.class, false, "D", new String[] {});

    checkBaseAndInterfaces(IJ.class, true, "", new String[] {});
    checkBaseAndInterfaces(IK.class, true, "", new String[] {"IJ"});
    checkBaseAndInterfaces(IL.class, true, "", new String[] {"IK"});
    checkBaseAndInterfaces(J.class, false, "", new String[] {"IJ"});
    checkBaseAndInterfaces(K.class, false, "", new String[] {"IJ", "IK"});
    checkBaseAndInterfaces(L.class, false, "", new String[] {"IJ", "IK", "IL"});
    checkBaseAndInterfaces(M.class, false, "", new String[] {"IJ", "IK", "IL"});

    checkBaseAndInterfaces(P.class, false, "", new String[] {});
    checkBaseAndInterfaces(IQ.class, true, "", new String[] {});
    checkBaseAndInterfaces(Q.class, false, "", new String[] {"IQ"});
    checkBaseAndInterfaces(R.class, false, "P", new String[] {"IQ"});
    checkBaseAndInterfaces(S.class, false, "P", new String[] {"IQ"});
    checkBaseAndInterfaces(T.class, false, "", new String[] {"IQ"});
    checkBaseAndInterfaces(U.class, false, "R", new String[] {});
    checkBaseAndInterfaces(V.class, false, "S", new String[] {});
    checkBaseAndInterfaces(W.class, false, "T", new String[] {});

    // overloaded methods check
    D d = new D();
    d.ia();
    d.ia(10);
    d.ia("bye");
    d.ia("bye", false);
  }
}

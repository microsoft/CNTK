import allprotected.*;

public class allprotected_runme {
  static {
    try {
      System.loadLibrary("allprotected");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    MyProtectedBase mpb = new MyProtectedBase("MyProtectedBase");
    mpb.accessProtected();
  }
}

class MyProtectedBase extends ProtectedBase
{
  MyProtectedBase(String name) {
    super(name);
  }
  void accessProtected() {
    String s = virtualMethod();
    if (!s.equals("ProtectedBase"))
      throw new RuntimeException("Failed");

    Klass k = instanceMethod(new Klass("xyz"));
    if (!k.getName().equals("xyz")) 
      throw new RuntimeException("Failed");

    k = instanceOverloaded(new Klass("xyz"));
    if (!k.getName().equals("xyz")) 
      throw new RuntimeException("Failed");

    k = instanceOverloaded(new Klass("xyz"), "abc");
    if (!k.getName().equals("abc")) 
      throw new RuntimeException("Failed");

    k = ProtectedBase.staticMethod(new Klass("abc"));
    if (!k.getName().equals("abc")) 
      throw new RuntimeException("Failed");

    k = ProtectedBase.staticOverloaded(new Klass("xyz"));
    if (!k.getName().equals("xyz")) 
      throw new RuntimeException("Failed");

    k = ProtectedBase.staticOverloaded(new Klass("xyz"), "abc");
    if (!k.getName().equals("abc")) 
      throw new RuntimeException("Failed");

    setInstanceMemberVariable(30);
    int i = getInstanceMemberVariable();
    if (i != 30)
      throw new RuntimeException("Failed");

    setStaticMemberVariable(40);
    i = getStaticMemberVariable();
    if (i != 40)
      throw new RuntimeException("Failed");

    i = staticConstMemberVariable;
    if (i != 20)
      throw new RuntimeException("Failed");

    setAnEnum(ProtectedBase.AnEnum.EnumVal1);
    ProtectedBase.AnEnum ae = getAnEnum();
    if (ae != ProtectedBase.AnEnum.EnumVal1)
      throw new RuntimeException("Failed");
  }
}


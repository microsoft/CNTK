// This tests changes the package name from nspace to nspacePackage as javac can't seem to resolve classes and packages having the same name
public class nspace_runme {

  static {
    try {
	System.loadLibrary("nspace");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    // constructors and destructors
    nspacePackage.Outer.Inner1.Color color1 = new nspacePackage.Outer.Inner1.Color();
    nspacePackage.Outer.Inner1.Color color = new nspacePackage.Outer.Inner1.Color(color1);
    color1.delete();
    color1 = null;

    // class methods
    color.colorInstanceMethod(20.0);
    nspacePackage.Outer.Inner1.Color.colorStaticMethod(20.0);
    nspacePackage.Outer.Inner1.Color created = nspacePackage.Outer.Inner1.Color.create();

    // class enums
    nspacePackage.Outer.SomeClass someClass = new nspacePackage.Outer.SomeClass();
    nspacePackage.Outer.Inner1.Color.Channel channel = someClass.GetInner1ColorChannel();
    if (channel != nspacePackage.Outer.Inner1.Color.Channel.Transmission)
      throw new RuntimeException("Transmission wrong");

    // class anonymous enums
    int val1 = nspacePackage.Outer.Inner1.Color.ColorEnumVal1;
    int val2 = nspacePackage.Outer.Inner1.Color.ColorEnumVal2;
    if (val1 != 0 || val2 != 0x22)
      throw new RuntimeException("ColorEnumVal wrong");

    // instance member variables
    color.setInstanceMemberVariable(123);
    if (color.getInstanceMemberVariable() != 123)
      throw new RuntimeException("instance member variable failed");

    // static member variables
    nspacePackage.Outer.Inner1.Color.setStaticMemberVariable(789);
    if (nspacePackage.Outer.Inner1.Color.getStaticMemberVariable() != 789)
      throw new RuntimeException("static member variable failed");
    if (nspacePackage.Outer.Inner1.Color.staticConstMemberVariable != 222)
      throw new RuntimeException("static const member variable failed");
    if (nspacePackage.Outer.Inner1.Color.staticConstEnumMemberVariable != nspacePackage.Outer.Inner1.Color.Channel.Transmission)
      throw new RuntimeException("static const enum member variable failed");

    // Same class different namespaces
    nspacePackage.Outer.Inner1.Color col1 = new nspacePackage.Outer.Inner1.Color();
    nspacePackage.Outer.Inner2.Color col2 = nspacePackage.Outer.Inner2.Color.create();
    col2.colors(col1, col1, col2, col2, col2);

    // check globals in a namespace don't get mangled with the nspacePackage option
    nspacePackage.nspace.namespaceFunction(color);
    nspacePackage.nspace.setNamespaceVar(111);
    if (nspacePackage.nspace.getNamespaceVar() != 111)
      throw new RuntimeException("global var failed");

    // global enums
    nspacePackage.Outer.Inner1.Channel outerChannel1 = someClass.GetInner1Channel();
    if (outerChannel1 != nspacePackage.Outer.Inner1.Channel.Transmission1)
      throw new RuntimeException("Transmission1 wrong");
    nspacePackage.Outer.Inner2.Channel outerChannel2 = someClass.GetInner2Channel();
    if (outerChannel2 != nspacePackage.Outer.Inner2.Channel.Transmission2)
      throw new RuntimeException("Transmission2 wrong");

    // turn feature off / ignoring
    nspacePackage.Outer.namespce ns = new nspacePackage.Outer.namespce();
    nspacePackage.NoNSpacePlease nons = new nspacePackage.NoNSpacePlease();

    // Derived class
    nspacePackage.Outer.Inner3.Blue blue3 = new nspacePackage.Outer.Inner3.Blue();
    blue3.blueInstanceMethod();
    nspacePackage.Outer.Inner4.Blue blue4 = new nspacePackage.Outer.Inner4.Blue();
    blue4.blueInstanceMethod();
  }
}

// This tests changes the package name from nspace to nspacePackage as javac can't seem to resolve classes and packages having the same name
public class nspace_extend_runme {

  static {
    try {
	System.loadLibrary("nspace_extend");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    {
      // constructors and destructors
      nspace_extendPackage.Outer.Inner1.Color color1 = new nspace_extendPackage.Outer.Inner1.Color();
      nspace_extendPackage.Outer.Inner1.Color color = new nspace_extendPackage.Outer.Inner1.Color(color1);
      color1.delete();
      color1 = null;

      // class methods
      color.colorInstanceMethod(20.0);
      nspace_extendPackage.Outer.Inner1.Color.colorStaticMethod(20.0);
      nspace_extendPackage.Outer.Inner1.Color created = nspace_extendPackage.Outer.Inner1.Color.create();
    }
    {
      // constructors and destructors
      nspace_extendPackage.Outer.Inner2.Color color2 = new nspace_extendPackage.Outer.Inner2.Color();
      nspace_extendPackage.Outer.Inner2.Color color = new nspace_extendPackage.Outer.Inner2.Color(color2);
      color2.delete();
      color2 = null;

      // class methods
      color.colorInstanceMethod(20.0);
      nspace_extendPackage.Outer.Inner2.Color.colorStaticMethod(20.0);
      nspace_extendPackage.Outer.Inner2.Color created = nspace_extendPackage.Outer.Inner2.Color.create();

      // Same class different namespaces
      nspace_extendPackage.Outer.Inner1.Color col1 = new nspace_extendPackage.Outer.Inner1.Color();
      nspace_extendPackage.Outer.Inner2.Color col2 = nspace_extendPackage.Outer.Inner2.Color.create();
      col2.colors(col1, col1, col2, col2, col2);
    }
  }
}

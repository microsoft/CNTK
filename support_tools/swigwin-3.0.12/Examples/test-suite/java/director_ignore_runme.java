import director_ignore.*;

public class director_ignore_runme {
  static {
    try {
        System.loadLibrary("director_ignore");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    // Just check the classes can be instantiated and other methods work as expected
    DIgnoresDerived a = new DIgnoresDerived();
    if (a.Triple(5) != 15)
      throw new RuntimeException("Triple failed");
    DAbstractIgnoresDerived b = new DAbstractIgnoresDerived();
    if (b.Quadruple(5) != 20)
      throw new RuntimeException("Quadruple failed");
  }
}

class DIgnoresDerived extends DIgnores
{
  public DIgnoresDerived()
  {
    super();
  }
}

class DAbstractIgnoresDerived extends DAbstractIgnores
{
  public DAbstractIgnoresDerived()
  {
    super();
  }
}


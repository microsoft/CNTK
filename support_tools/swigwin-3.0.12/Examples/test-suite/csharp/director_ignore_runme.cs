using System;

namespace director_ignoreNamespace {

public class runme
{
  static void Main() 
  {
    runme r = new runme();
    r.run();
  }

  void run()
  {
    // Just check the classes can be instantiated and other methods work as expected
    DIgnoresDerived a = new DIgnoresDerived();
    if (a.Triple(5) != 15)
      throw new Exception("Triple failed");
    DAbstractIgnoresDerived b = new DAbstractIgnoresDerived();
    if (b.Quadruple(5) != 20)
      throw new Exception("Quadruple failed");
  }
}

class DIgnoresDerived : DIgnores
{
  public DIgnoresDerived() : base()
  {
  }

  // These will give a warning if the %ignore is not working
  public virtual int OverloadedMethod(int n, int xoffset, int yoffset) { return 0; }
  public virtual int OverloadedMethod(int n, int xoffset) { return 0; }
  public virtual int OverloadedMethod(int n) { return 0; }

  public virtual int OverloadedProtectedMethod(int n, int xoffset, int yoffset) { return 0; }
  public virtual int OverloadedProtectedMethod(int n, int xoffset) { return 0; }
  public virtual int OverloadedProtectedMethod(int n) { return 0; }
}

class DAbstractIgnoresDerived : DAbstractIgnores
{
  public DAbstractIgnoresDerived() : base()
  {
  }

  // These will give a warning if the %ignore is not working
  public virtual int OverloadedMethod(int n, int xoffset, int yoffset) { return 0; }
  public virtual int OverloadedMethod(int n, int xoffset) { return 0; }
  public virtual int OverloadedMethod(int n) { return 0; }

  public virtual int OverloadedProtectedMethod(int n, int xoffset, int yoffset) { return 0; }
  public virtual int OverloadedProtectedMethod(int n, int xoffset) { return 0; }
  public virtual int OverloadedProtectedMethod(int n) { return 0; }
}

}

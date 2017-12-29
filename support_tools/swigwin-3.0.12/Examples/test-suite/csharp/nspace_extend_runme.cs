using System;

public class runme
{
  static void Main() 
  {
    {
      // constructors and destructors
      nspace_extendNamespace.Outer.Inner1.Color color1 = new nspace_extendNamespace.Outer.Inner1.Color();
      nspace_extendNamespace.Outer.Inner1.Color color = new nspace_extendNamespace.Outer.Inner1.Color(color1);
      color1.Dispose();
      color1 = null;

      // class methods
      color.colorInstanceMethod(20.0);
      nspace_extendNamespace.Outer.Inner1.Color.colorStaticMethod(20.0);
      nspace_extendNamespace.Outer.Inner1.Color created = nspace_extendNamespace.Outer.Inner1.Color.create();
      created.Dispose();
    }
    {
      // constructors and destructors
      nspace_extendNamespace.Outer.Inner2.Color color2 = new nspace_extendNamespace.Outer.Inner2.Color();
      nspace_extendNamespace.Outer.Inner2.Color color = new nspace_extendNamespace.Outer.Inner2.Color(color2);
      color2.Dispose();
      color2 = null;

      // class methods
      color.colorInstanceMethod(20.0);
      nspace_extendNamespace.Outer.Inner2.Color.colorStaticMethod(20.0);
      nspace_extendNamespace.Outer.Inner2.Color created = nspace_extendNamespace.Outer.Inner2.Color.create();
      created.Dispose();

      // Same class different namespaces
      nspace_extendNamespace.Outer.Inner1.Color col1 = new nspace_extendNamespace.Outer.Inner1.Color();
      nspace_extendNamespace.Outer.Inner2.Color col2 = nspace_extendNamespace.Outer.Inner2.Color.create();
      col2.colors(col1, col1, col2, col2, col2);
    }
  }
}

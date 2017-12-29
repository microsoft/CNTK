using System;

public class runme
{
    static void Main()
    {
      // constructors and destructors
      nspaceNamespace.Outer.Inner1.Color color1 = new nspaceNamespace.Outer.Inner1.Color();
      nspaceNamespace.Outer.Inner1.Color color = new nspaceNamespace.Outer.Inner1.Color(color1);
      color1.Dispose();
      color1 = null;

      // class methods
      color.colorInstanceMethod(20.0);
      nspaceNamespace.Outer.Inner1.Color.colorStaticMethod(20.0);
      nspaceNamespace.Outer.Inner1.Color created = nspaceNamespace.Outer.Inner1.Color.create();
      created.Dispose();

      // class enums
      nspaceNamespace.Outer.SomeClass someClass = new nspaceNamespace.Outer.SomeClass();
      nspaceNamespace.Outer.Inner1.Color.Channel channel = someClass.GetInner1ColorChannel();
      if (channel != nspaceNamespace.Outer.Inner1.Color.Channel.Transmission)
        throw new ApplicationException("Transmission wrong");

      // class anonymous enums
      int val1 = nspaceNamespace.Outer.Inner1.Color.ColorEnumVal1;
      int val2 = nspaceNamespace.Outer.Inner1.Color.ColorEnumVal2;
      if (val1 != 0 || val2 != 0x22)
        throw new ApplicationException("ColorEnumVal wrong");

      // instance member variables
      color.instanceMemberVariable = 123;
      if (color.instanceMemberVariable != 123)
        throw new ApplicationException("instance member variable failed");

      // static member variables
      nspaceNamespace.Outer.Inner1.Color.staticMemberVariable = 789;
      if (nspaceNamespace.Outer.Inner1.Color.staticMemberVariable != 789)
        throw new ApplicationException("static member variable failed");
      if (nspaceNamespace.Outer.Inner1.Color.staticConstMemberVariable != 222)
        throw new ApplicationException("static const member variable failed");
      if (nspaceNamespace.Outer.Inner1.Color.staticConstEnumMemberVariable != nspaceNamespace.Outer.Inner1.Color.Channel.Transmission)
        throw new ApplicationException("static const enum member variable failed");

      // check globals in a namespace don't get mangled with the nspaceNamespace option
      nspaceNamespace.nspace.namespaceFunction(color);
      nspaceNamespace.nspace.namespaceVar = 111;
      if (nspaceNamespace.nspace.namespaceVar != 111)
        throw new ApplicationException("global var failed");

      // Same class different namespaces
      nspaceNamespace.Outer.Inner1.Color col1 = new nspaceNamespace.Outer.Inner1.Color();
      nspaceNamespace.Outer.Inner2.Color col2 = nspaceNamespace.Outer.Inner2.Color.create();
      col2.colors(col1, col1, col2, col2, col2);

      // global enums
      nspaceNamespace.Outer.Inner1.Channel outerChannel1 = someClass.GetInner1Channel();
      if (outerChannel1 != nspaceNamespace.Outer.Inner1.Channel.Transmission1)
        throw new ApplicationException("Transmission1 wrong");
      nspaceNamespace.Outer.Inner2.Channel outerChannel2 = someClass.GetInner2Channel();
      if (outerChannel2 != nspaceNamespace.Outer.Inner2.Channel.Transmission2)
        throw new ApplicationException("Transmission2 wrong");

      // turn feature off / ignoring
      nspaceNamespace.Outer.namespce ns = new nspaceNamespace.Outer.namespce();
      ns.Dispose();
      nspaceNamespace.NoNSpacePlease nons = new nspaceNamespace.NoNSpacePlease();
      nons.Dispose();

      // Derived class
      nspaceNamespace.Outer.Inner3.Blue blue3 = new nspaceNamespace.Outer.Inner3.Blue();
      blue3.blueInstanceMethod();
      nspaceNamespace.Outer.Inner4.Blue blue4 = new nspaceNamespace.Outer.Inner4.Blue();
      blue4.blueInstanceMethod();
    }
}

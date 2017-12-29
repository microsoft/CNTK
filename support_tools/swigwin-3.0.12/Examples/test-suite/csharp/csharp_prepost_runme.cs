
using System;
using System.Reflection;
using csharp_prepostNamespace;

public class csharp_prepost_runme {

  class PrePost3_Derived : PrePost3
  {
    public PrePost3_Derived(){}
    public override void method(ref double[] vpre, DoubleVector vpost)
    {
      Assert(vpre[0], 1.0);
      vpre[0] = 2.0;
      Assert(vpost.Count, 2);
      vpost.Add(1.0);
    }
    public override int methodint(ref double[] vpre, DoubleVector vpost)
    {
      method(ref vpre, vpost);
      return vpost.Count;
    }
  }
  public static void Main() {
    {
      double[] v;
      csharp_prepost.globalfunction(out v);
      Assert(v.Length, 3);
      Assert(v[0], 0.0);
      Assert(v[1], 1.1);
      Assert(v[2], 2.2);
    }
    {
      double[] v;
      new PrePostTest(out v);
      Assert(v.Length, 2);
      Assert(v[0], 3.3);
      Assert(v[1], 4.4);
    }
    {
      double[] v;
      PrePostTest p = new PrePostTest();
      p.method(out v);
      Assert(v.Length, 2);
      Assert(v[0], 5.5);
      Assert(v[1], 6.6);
    }
    {
      double[] v;
      PrePostTest.staticmethod(out v);
      Assert(v.Length, 2);
      Assert(v[0], 7.7);
      Assert(v[1], 8.8);
    }

    {
      PrePost3_Derived p = new PrePost3_Derived();
      double[] vpre = new double[] { 1.0 };
      DoubleVector vpost = new DoubleVector();
      vpost.Add(3.0);
      vpost.Add(4.0);
      p.method(ref vpre, vpost);
      Assert(vpre[0], 2.0);
      Assert(vpost.Count, 3);
    }
    {
      PrePost3_Derived p = new PrePost3_Derived();
      double[] vpre = new double[] { 1.0 };
      DoubleVector vpost = new DoubleVector();
      vpost.Add(3.0);
      vpost.Add(4.0);
      int size = p.methodint(ref vpre, vpost);
      Assert(vpre[0], 2.0);
      Assert(vpost.Count, 3);
      Assert(size, 3);
    }

    // Check attributes are generated for the constructor helper function
    {
      CsinAttributes c = new CsinAttributes(5);
      Assert(c.getVal(), 500);

      Type type = typeof(CsinAttributes);
      {
        MethodInfo member = (MethodInfo)type.GetMember("SwigConstructCsinAttributes", BindingFlags.NonPublic | BindingFlags.Static)[0];
        if (Attribute.GetCustomAttribute(member, typeof(CustomIntPtrAttribute)) == null)
          throw new Exception("No CustomIntPtr attribute for " + member.Name);
        ParameterInfo parameter = member.GetParameters()[0]; // expecting one parameter
        if (parameter.Name != "val")
          throw new Exception("Incorrect parameter name");
        Attribute attribute = Attribute.GetCustomAttributes(parameter)[0];
        if (attribute.GetType() != typeof(CustomIntAttribute))
          throw new Exception("Expecting CustomInt attribute");
      }
    }
    // Dates
    {
      // pre post typemap attributes example
      System.DateTime dateIn = new System.DateTime(2011, 4, 13);
      System.DateTime dateOut = new System.DateTime();

      // Note in calls below, dateIn remains unchanged and dateOut 
      // is set to a new value by the C++ call
      csharp_prepostNamespace.Action action = new csharp_prepostNamespace.Action(dateIn, out dateOut);
      if (dateOut != dateIn)
        throw new Exception("dates wrong");

      dateIn = new System.DateTime(2012, 7, 14);
      action.doSomething(dateIn, out dateOut);
      if (dateOut != dateIn)
        throw new Exception("dates wrong");

      System.DateTime refDate = new System.DateTime(1999, 12, 31);
      if (csharp_prepost.ImportantDate != refDate)
        throw new Exception("dates wrong");

      refDate = new System.DateTime(1999, 12, 31);
      csharp_prepost.ImportantDate = refDate;
      System.DateTime importantDate = csharp_prepost.ImportantDate;
      if (importantDate != refDate)
        throw new Exception("dates wrong");

      System.DateTime christmasEve = new System.DateTime(2000, 12, 24);
      csharp_prepost.addYears(ref christmasEve, 10);
      if (christmasEve != new System.DateTime(2010, 12, 24))
        throw new Exception("dates wrong");

      Person person = new Person();
      person.Birthday = christmasEve;
      if (person.Birthday != christmasEve)
        throw new Exception("dates wrong");
    }
  }
  private static void Assert(double d1, double d2) {
    if (d1 != d2)
      throw new Exception("assertion failure. " + d1 + " != " + d2);
  }
}

// Custom attribute classes
[AttributeUsage(AttributeTargets.All, Inherited = false, AllowMultiple = true)]
public class CustomIntAttribute : Attribute {}

[AttributeUsage(AttributeTargets.All, Inherited = false, AllowMultiple = true)]
public class CustomIntPtrAttribute : Attribute {}

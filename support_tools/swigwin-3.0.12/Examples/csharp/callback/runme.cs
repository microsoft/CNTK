using System;

public class runme
{
  static void Main() 
  {
    Console.WriteLine("Adding and calling a normal C++ callback");
    Console.WriteLine("----------------------------------------");

    Caller caller = new Caller();
    using (Callback callback = new Callback())
    {
      caller.setCallback(callback);
      caller.call();
      caller.resetCallback();
    }

    Console.WriteLine();
    Console.WriteLine("Adding and calling a C# callback");
    Console.WriteLine("------------------------------------");

    using (Callback callback = new CSharpCallback())
    {
      caller.setCallback(callback);
      caller.call();
      caller.resetCallback();
    }

    Console.WriteLine();
    Console.WriteLine("C# exit");
  }
}

public class CSharpCallback : Callback
{
  public CSharpCallback()
    : base()
  {
  }

  public override void run()
  {
    Console.WriteLine("CSharpCallback.run()");
  }
}


// This file illustrates the cross language polymorphism using directors.

using System;

// CEO class, which overrides Employee::getPosition().

class CEO : Manager {
  public CEO(String name) : base(name) {
  }
  public override String getPosition() {
    return "CEO";
  }
  // Public method to stop the SWIG proxy base class from thinking it owns the underlying C++ memory.
  public void disownMemory() {
    swigCMemOwn = false; 
  } 
}


public class runme
{
  static void Main() 
  {
    // Create an instance of CEO, a class derived from the C# proxy of the 
    // underlying C++ class. The calls to getName() and getPosition() are standard,
    // the call to getTitle() uses the director wrappers to call CEO.getPosition().

    CEO e = new CEO("Alice");
    Console.WriteLine( e.getName() + " is a " + e.getPosition() );
    Console.WriteLine( "Just call her \"" + e.getTitle() + "\"" );
    Console.WriteLine( "----------------------" );

    // Create a new EmployeeList instance.  This class does not have a C++
    // director wrapper, but can be used freely with other classes that do.

    using (EmployeeList list = new EmployeeList()) {

    // EmployeeList owns its items, so we must surrender ownership of objects we add.
    e.disownMemory();
    list.addEmployee(e);
    Console.WriteLine( "----------------------" );

    // Now we access the first four items in list (three are C++ objects that
    // EmployeeList's constructor adds, the last is our CEO). The virtual
    // methods of all these instances are treated the same. For items 0, 1, and
    // 2, all methods resolve in C++. For item 3, our CEO, getTitle calls
    // getPosition which resolves in C#. The call to getPosition is
    // slightly different, however, because of the overridden getPosition() call, since
    // now the object reference has been "laundered" by passing through
    // EmployeeList as an Employee*. Previously, C# resolved the call
    // immediately in CEO, but now C# thinks the object is an instance of
    // class Employee. So the call passes through the
    // Employee proxy class and on to the C wrappers and C++ director,
    // eventually ending up back at the C# CEO implementation of getPosition().
    // The call to getTitle() for item 3 runs the C++ Employee::getTitle()
    // method, which in turn calls getPosition(). This virtual method call
    // passes down through the C++ director class to the C# implementation
    // in CEO. All this routing takes place transparently.

    Console.WriteLine( "(position, title) for items 0-3:" );

    Console.WriteLine( "  " + list.get_item(0).getPosition() + ", \"" + list.get_item(0).getTitle() + "\"" );
    Console.WriteLine( "  " + list.get_item(1).getPosition() + ", \"" + list.get_item(1).getTitle() + "\"" );
    Console.WriteLine( "  " + list.get_item(2).getPosition() + ", \"" + list.get_item(2).getTitle() + "\"" );
    Console.WriteLine( "  " + list.get_item(3).getPosition() + ", \"" + list.get_item(3).getTitle() + "\"" );
    Console.WriteLine( "----------------------" );

    // The using statement ensures the EmployeeList.Dispose() will be called, which will delete all the Employee*
    // items it contains. The last item is our CEO, which gets destroyed as well.
    }
    Console.WriteLine( "----------------------" );

    // All done.

    Console.WriteLine( "C# exit" );
  }
}

// This file illustrates the cross language polymorphism using directors.


// CEO class, which overrides Employee::getPosition().

class CEO extends Manager {
  public CEO(String name) {
    super(name);
    }
  public String getPosition() {
    return "CEO";
    }
  // Public method to stop the SWIG proxy base class from thinking it owns the underlying C++ memory.
  public void disownMemory() {
    swigCMemOwn = false; 
  } 
}


public class runme {
  static {
    try {
        System.loadLibrary("example");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {

    // Create an instance of CEO, a class derived from the Java proxy of the 
    // underlying C++ class. The calls to getName() and getPosition() are standard,
    // the call to getTitle() uses the director wrappers to call CEO.getPosition().

    CEO e = new CEO("Alice");
    System.out.println( e.getName() + " is a " + e.getPosition() );
    System.out.println( "Just call her \"" + e.getTitle() + "\"" );
    System.out.println( "----------------------" );


    // Create a new EmployeeList instance.  This class does not have a C++
    // director wrapper, but can be used freely with other classes that do.

    EmployeeList list = new EmployeeList();

    // EmployeeList owns its items, so we must surrender ownership of objects we add.
    e.disownMemory();
    list.addEmployee(e);
    System.out.println( "----------------------" );

    // Now we access the first four items in list (three are C++ objects that
    // EmployeeList's constructor adds, the last is our CEO). The virtual
    // methods of all these instances are treated the same. For items 0, 1, and
    // 2, all methods resolve in C++. For item 3, our CEO, getTitle calls
    // getPosition which resolves in Java. The call to getPosition is
    // slightly different, however, because of the overridden getPosition() call, since
    // now the object reference has been "laundered" by passing through
    // EmployeeList as an Employee*. Previously, Java resolved the call
    // immediately in CEO, but now Java thinks the object is an instance of
    // class Employee. So the call passes through the
    // Employee proxy class and on to the C wrappers and C++ director,
    // eventually ending up back at the Java CEO implementation of getPosition().
    // The call to getTitle() for item 3 runs the C++ Employee::getTitle()
    // method, which in turn calls getPosition(). This virtual method call
    // passes down through the C++ director class to the Java implementation
    // in CEO. All this routing takes place transparently.

    System.out.println( "(position, title) for items 0-3:" );

    System.out.println( "  " + list.get_item(0).getPosition() + ", \"" + list.get_item(0).getTitle() + "\"" );
    System.out.println( "  " + list.get_item(1).getPosition() + ", \"" + list.get_item(1).getTitle() + "\"" );
    System.out.println( "  " + list.get_item(2).getPosition() + ", \"" + list.get_item(2).getTitle() + "\"" );
    System.out.println( "  " + list.get_item(3).getPosition() + ", \"" + list.get_item(3).getTitle() + "\"" );
    System.out.println( "----------------------" );

    // Time to delete the EmployeeList, which will delete all the Employee*
    // items it contains. The last item is our CEO, which gets destroyed as well.
    list.delete();
    System.out.println( "----------------------" );

    // All done.

    System.out.println( "java exit" );

  }
}

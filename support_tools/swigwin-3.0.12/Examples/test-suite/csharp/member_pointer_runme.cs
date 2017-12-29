using System;
using member_pointerNamespace;

public class runme {
  public static SWIGTYPE_m_Shape__f_void__double memberPtr = null;
  static void Main() {
    // Get the pointers

    SWIGTYPE_m_Shape__f_void__double area_pt = member_pointer.areapt();
    SWIGTYPE_m_Shape__f_void__double perim_pt = member_pointer.perimeterpt();

    // Create some objects

    Square s = new Square(10);

    // Do some calculations

    check( "Square area ", 100.0, member_pointer.do_op(s,area_pt) );
    check( "Square perim", 40.0, member_pointer.do_op(s,perim_pt) );

    memberPtr = member_pointer.areavar;
    memberPtr = member_pointer.perimetervar;

    // Try the variables
    check( "Square area ", 100.0, member_pointer.do_op(s,member_pointer.areavar) );
    check( "Square perim", 40.0, member_pointer.do_op(s,member_pointer.perimetervar) );

    // Modify one of the variables
    member_pointer.areavar = perim_pt;

    check( "Square perimeter", 40.0, member_pointer.do_op(s,member_pointer.areavar) );

    // Try the constants

    memberPtr = member_pointer.AREAPT;
    memberPtr = member_pointer.PERIMPT;
    memberPtr = member_pointer.NULLPT;

    check( "Square area ", 100.0, member_pointer.do_op(s,member_pointer.AREAPT) );
    check( "Square perim", 40.0, member_pointer.do_op(s,member_pointer.PERIMPT) );

  }
  private static void check(string what, double expected, double actual) {
    if (expected != actual)
      throw new ApplicationException("Failed: " + what + " Expected: " + expected + " Actual: " + actual);
  }
}

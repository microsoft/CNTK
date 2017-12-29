using System;
using rename_pcre_enumNamespace;

public class runme {
  static void Main() {
    Foo foo = Foo.First;
    if ( foo == Foo.Second )
      throw new Exception("Enum values should be different");

    // Check that Foo_Max enum element was ignored.
    int numFooEnumElements = Enum.GetValues(typeof(Foo)).Length;
    if ( numFooEnumElements != 2 )
      throw new Exception(String.Format("Enum should have 2 elements, not {0}",
                                        numFooEnumElements));

    BoundaryCondition bc = BoundaryCondition.MaxMax;
    if ( (int)bc != 2 )
      throw new Exception("Wrong enum value");

    Colour c = Colour.red;
    if ( c == Colour.blue )
      throw new Exception("Enum values should be different");
  }
}

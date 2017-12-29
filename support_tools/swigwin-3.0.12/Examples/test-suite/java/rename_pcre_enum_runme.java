import rename_pcre_enum.*;

public class rename_pcre_enum_runme {
  static { System.loadLibrary("rename_pcre_enum"); }

  public static void main(String argv[])
  {
    Foo foo = Foo.First;
    if ( foo == Foo.Second )
      throw new RuntimeException("Enum values should be different");

    // Check that Foo_Max enum element was ignored.
    int numFooEnumElements = Foo.values().length;
    if ( numFooEnumElements != 2 )
      throw new RuntimeException(String.format("Enum should have 2 elements, not %d",
                                        numFooEnumElements));

    BoundaryCondition bc = BoundaryCondition.MaxMax;
    if ( bc.ordinal() != 2 )
      throw new RuntimeException("Wrong enum value");

    Colour c = Colour.red;
    if ( c == Colour.blue )
        throw new RuntimeException("Enum values should be different");
  }
}

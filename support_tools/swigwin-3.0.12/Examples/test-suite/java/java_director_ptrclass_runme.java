
import java_director_ptrclass.*;

public class java_director_ptrclass_runme {

  static {
    try {
      System.loadLibrary("java_director_ptrclass");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    Foo f = new Foo();
    Foo ft = new TouchingFoo(); 
    Baz b = new Baz();
    if (b.GetTouched()) {
      throw new RuntimeException ( "Baz should not have been touched yet." );
    }

    Baz b2 = f.FinalMaybeTouch(b);

    if (b2.GetTouched() || b.GetTouched()) {
      throw new RuntimeException ( "Baz should not have been touched by Foo." );
    }

    Baz b3 = ft.FinalMaybeTouch(b);

    if (!b.GetTouched() || !b3.GetTouched() || !b2.GetTouched()) {
      throw new RuntimeException ( "Baz was not touched by TouchingFoo. This" +
                                   " might mean the directorin typemap is not" +
                                   " parsing the typemap(jstype, Bar) in its" +
                                   " 'descriptor' kwarg correctly." );
    }
  }
}

class TouchingFoo extends Foo {
  @Override
  public Baz MaybeTouch(Baz baz_ptr) { 
    baz_ptr.SetTouched();
    return baz_ptr;
  }
}


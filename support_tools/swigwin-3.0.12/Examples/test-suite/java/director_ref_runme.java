
import director_ref.*;

public class director_ref_runme {

  static {
    try {
      System.loadLibrary("director_ref");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    director_ref_MyFoo a = new director_ref_MyFoo();
    if (a.GetRefCount() != 1) {
      throw new RuntimeException ( "Refcount test 1 failed." );
    }
    
    // Make sure director logic still works.
    if (!a.GetMsg().equals("director_ref_MyFoo-default")) {
      throw new RuntimeException ( "Test 1 failed" );
    }
    if (!a.GetMsg("boo").equals("director_ref_MyFoo-boo")) {
      throw new RuntimeException ( "Test 2 failed" );
    }

    a.delete();  // should delete the object.
    if (a.cppDeleted != true) {
      throw new RuntimeException ( "Unref test 1 failed." );
    }

    a = new director_ref_MyFoo();
    FooPtr p = new FooPtr(a);
    if (a.GetRefCount() != 2) {
      throw new RuntimeException ( "Refcount test 2 failed." );
    }
    a.delete();  // Shouldn't actually delete the underlying object
    if (a.cppDeleted) {
      throw new RuntimeException ( "Unref test 2 failed." );
    }
    if (p.GetOwnedRefCount() != 1) {
      throw new RuntimeException ( "Unref test 3 failed." );
    }
    p.Reset();  // Now it should be deleted on the cpp side.
    // We can't check cppDeleted because the director will stop
    // working after a delete() call.
    if (p.GetOwnedRefCount() != 0) {
      throw new RuntimeException ( "Unref test 4 failed." );
    }
  }
}

class director_ref_MyFoo extends Foo {
    public director_ref_MyFoo() {
      super();
    }
    public director_ref_MyFoo(int i) {
      super(i);
    }
    public String Msg(String msg) { 
      return "director_ref_MyFoo-" + msg; 
    }
    public void OnDelete() {
      cppDeleted = true;
    }

    public boolean cppDeleted = false;
}


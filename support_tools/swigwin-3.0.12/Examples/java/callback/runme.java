public class runme
{
  static {
    try {
        System.loadLibrary("example");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String[] args)
  {
    System.out.println("Adding and calling a normal C++ callback");
    System.out.println("----------------------------------------");

    Caller              caller = new Caller();
    Callback            callback = new Callback();
    
    caller.setCallback(callback);
    caller.call();
    caller.delCallback();

    callback = new JavaCallback();

    System.out.println();
    System.out.println("Adding and calling a Java callback");
    System.out.println("------------------------------------");

    caller.setCallback(callback);
    caller.call();
    caller.delCallback();

    // Test that a double delete does not occur as the object has already been deleted from the C++ layer.
    // Note that the garbage collector can also call the delete() method via the finalizer (callback.finalize())
    // at any point after here.
    callback.delete();

    System.out.println();
    System.out.println("java exit");
  }
}

class JavaCallback extends Callback
{
  public JavaCallback()
  {
    super();
  }

  public void run()
  {
    System.out.println("JavaCallback.run()");
  }
}


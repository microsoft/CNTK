// Mainly tests that directors are finalized correctly

import java_director.*;

public class java_director_runme {

  static {
    try {
      System.loadLibrary("java_director");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  private static void WaitForGC()
  {
    System.gc();
    System.runFinalization();
    try {
      java.lang.Thread.sleep(10);
    } catch (java.lang.InterruptedException e) {
    }
  }

  public static void main(String argv[]) {
    QuuxContainer qc = createContainer();

    int instances = Quux.instances();
    if (instances != 4)
      throw new RuntimeException("Quux instances should be 4, actually " + instances);

    for (int i = 0; i < qc.size(); ++i) {
      Quux q = qc.get(i);

      if (!q.director_method().equals(qc.invoke(i))) {
        throw new RuntimeException ( "q.director_method()/qv.invoke(" + i + ")");
      }
    }

    qc = null;
    /* Watch qc get reaped, which causes the C++ object to delete
       objects from the internal vector */
    {
      int countdown = 500;
      int expectedCount = 0;
      while (true) {
        WaitForGC();
        if (--countdown == 0)
          break;
        if (Quux.instances() == expectedCount)
          break;
      };
      int actualCount = Quux.instances();
      if (actualCount != expectedCount)
        System.err.println("GC failed to run (java_director). Expected count: " + expectedCount + " Actual count: " + actualCount); // Finalizers are not guaranteed to be run and sometimes they just don't
    }

    /* Test Quux1's director disconnect method rename */
    Quux1 quux1 = new Quux1("quux1");
    if (quux1.disconnectMethodCalled)
      throw new RuntimeException("Oops");
    quux1.delete();
    if (!quux1.disconnectMethodCalled)
      throw new RuntimeException("disconnect method not called");
  }

  public static QuuxContainer createContainer() {
    QuuxContainer qc = new QuuxContainer();

    qc.push(new Quux("element 1"));
    qc.push(new java_director_MyQuux("element 2"));
    qc.push(new java_director_MyQuux("element 3"));
    qc.push(new Quux("element 4"));

    return qc;
  }
}

class java_director_MyQuux extends Quux {
  public java_director_MyQuux(String arg) {
    super(arg);
  }

  public String director_method() {
    return "java_director_MyQuux:" + member();
  }
}

class java_director_JavaExceptionTest extends JavaExceptionTest {
  public java_director_JavaExceptionTest() {
    super();
  }

  public void etest() throws Exception {
    super.etest();
  }
}


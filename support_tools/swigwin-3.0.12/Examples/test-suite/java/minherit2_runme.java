
import minherit2.*;
import java.lang.reflect.*;

public class minherit2_runme {

  static {
    try {
        System.loadLibrary("minherit2");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    try {

      Method method = IRemoteAsyncIO.class.getDeclaredMethod("asyncmethod", (java.lang.Class[])null);
      if ( !Modifier.isAbstract(method.getModifiers()) )
        throw new RuntimeException("asyncmethod should be abstract" );

      method = IRemoteSyncIO.class.getDeclaredMethod("syncmethod", (java.lang.Class[])null);
      if ( !Modifier.isAbstract(method.getModifiers()) )
        throw new RuntimeException("syncmethod should be abstract" );

    } catch (NoSuchMethodException e) {
      throw new RuntimeException(e);
    }
  }
}

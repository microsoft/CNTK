
import friends_template.*;

public class friends_template_runme {

  static {
    try {
	System.loadLibrary("friends_template");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    friends_template.OperatorOutputDouble(1.1, new MyClassDouble());
    friends_template.OperatorInputDouble(1.1, new MyClassDouble());
    friends_template.funk_hidden(1.1, new MyClassDouble());
    friends_template.funk_seen(1.1, new MyClassDouble());

    friends_template.TemplateFriendHiddenInt(0);
    friends_template.TemplateFriendSeenInt(0, 0);

    SWIGTYPE_p_MyClassT_int_t myClassInt = friends_template.makeMyClassInt();
    friends_template.OperatorInputInt(1, myClassInt);
    friends_template.OperatorFunkSeenInt(1.1, myClassInt);
  }
}

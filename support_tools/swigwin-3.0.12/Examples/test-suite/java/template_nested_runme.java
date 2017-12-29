
import template_nested.*;

public class template_nested_runme {

  static {
    try {
	System.loadLibrary("template_nested");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    new T_NormalTemplateNormalClass().tmethod(new NormalClass());
    new OuterClass().T_OuterTMethodNormalClass(new NormalClass());

    TemplateFuncs tf = new TemplateFuncs();
    if (tf.T_TemplateFuncs1Int(-10) != -10)
      throw new RuntimeException("it failed");
    if (tf.T_TemplateFuncs2Double(-12.3) != -12.3)
      throw new RuntimeException("it failed");

    T_NestedOuterTemplateDouble tn = new T_NestedOuterTemplateDouble();
    if (tn.hohum(-12.3) != -12.3)
      throw new RuntimeException("it failed");
    OuterClass.T_OuterClassInner1Int inner1 = new OuterClass().useInner1(new OuterClass.T_OuterClassInner1Int());
    OuterClass.T_OuterClassInner2NormalClass inner2 = new OuterClass.T_OuterClassInner2NormalClass();
    inner2.setEmbeddedVar(2);
    OuterClass.T_OuterClassInner2NormalClass inner22 = new OuterClass().useInner2Again(inner2);
    OuterClass.T_OuterClassInner1Double inner3 = new OuterClass.T_OuterClassInner1Double();
  }
}


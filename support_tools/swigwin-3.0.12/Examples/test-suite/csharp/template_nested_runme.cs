using System;
using template_nestedNamespace;
#pragma warning disable 219

public class runme {
  static void Main() {
    new T_NormalTemplateNormalClass().tmethod(new NormalClass());
    new OuterClass().T_OuterTMethodNormalClass(new NormalClass());

    TemplateFuncs tf = new TemplateFuncs();
    if (tf.T_TemplateFuncs1Int(-10) != -10)
      throw new Exception("it failed");
    if (tf.T_TemplateFuncs2Double(-12.3) != -12.3)
      throw new Exception("it failed");

    T_NestedOuterTemplateDouble tn = new T_NestedOuterTemplateDouble();
    if (tn.hohum(-12.3) != -12.3)
      throw new Exception("it failed");
    OuterClass.T_OuterClassInner1Int inner1 = new OuterClass().useInner1(new OuterClass.T_OuterClassInner1Int());
    OuterClass.T_OuterClassInner2NormalClass inner2 = new OuterClass.T_OuterClassInner2NormalClass();
    inner2.embeddedVar = 2;
    OuterClass.T_OuterClassInner2NormalClass inner22 = new OuterClass().useInner2Again(inner2);
  }
}


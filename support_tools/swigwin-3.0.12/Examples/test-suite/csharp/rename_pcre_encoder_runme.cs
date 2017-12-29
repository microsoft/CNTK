using System;
using rename_pcre_encoderNamespace;

public class runme {
  static void Main() {
    SomeWidget w = new SomeWidget();
    w.put_borderWidth(17);
    if ( w.get_borderWidth() != 17 )
      throw new Exception(String.Format("Border with should be 17, not {0}",
                                        w.get_borderWidth()));

    if ( rename_pcre_encoder.StartINSAneAndUNSAvoryTraNSAtlanticRaNSAck() != 42 )
      throw new Exception("Unexpected result of renamed function call");
  }
}

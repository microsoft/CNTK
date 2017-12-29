import rename_pcre_encoder.*;

public class rename_pcre_encoder_runme {
  static { System.loadLibrary("rename_pcre_encoder"); }

  public static void main(String argv[])
  {
    SomeWidget w = new SomeWidget();
    w.put_borderWidth(17);
    if ( w.get_borderWidth() != 17 )
      throw new RuntimeException(String.format("Border with should be 17, not %d",
                                               w.get_borderWidth()));
    if ( rename_pcre_encoder.StartINSAneAndUNSAvoryTraNSAtlanticRaNSAck() != 42 )
      throw new RuntimeException("Unexpected result of renamed function call");
  }
}


import rename_wildcard.*;

public class rename_wildcard_runme {

  static {
    try {
	System.loadLibrary("rename_wildcard");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    // Wildcard check
    {
      new GlobalWildStruct().mm1();
      new GlobalWildTemplateStructInt().mm1();
      new SpaceWildStruct().mm1();
      new SpaceWildTemplateStructInt().mm1();
    }
    // No declaration
    {
      new GlobalWildStruct().mm2a();
      new GlobalWildTemplateStructInt().mm2b();
      new SpaceWildStruct().mm2c();
      new SpaceWildTemplateStructInt().mm2d();

      new GlobalWildTemplateStructInt().tt2b();
      new SpaceWildTemplateStructInt().tt2d();
    }
    // With declaration
    {
      new GlobalWildStruct().mm3a();
      new GlobalWildTemplateStructInt().mm3b();
      new SpaceWildStruct().mm3c();
      new SpaceWildTemplateStructInt().mm3d();

      new GlobalWildTemplateStructInt().tt3b();
      new SpaceWildTemplateStructInt().tt3d();
    }
    // Global override too
    {
      new GlobalWildStruct().mm4a();
      new GlobalWildTemplateStructInt().mm4b();
      new SpaceWildStruct().mm4c();
      new SpaceWildTemplateStructInt().mm4d();

      new GlobalWildTemplateStructInt().tt4b();
      new SpaceWildTemplateStructInt().tt4d();
    }
    // %extend renames
    {
      new GlobalWildStruct().mm5a();
      new GlobalWildTemplateStructInt().mm5b();
      new SpaceWildStruct().mm5c();
      new SpaceWildTemplateStructInt().mm5d();

      new GlobalWildTemplateStructInt().tt5b();
      new SpaceWildTemplateStructInt().tt5d();
    }
    // operators
    {
      new GlobalWildStruct().opinta();
      new GlobalWildTemplateStructInt().opintb();
      new SpaceWildStruct().opintc();
      new SpaceWildTemplateStructInt().opintd();

      new GlobalWildTemplateStructInt().opdoubleb();
      new SpaceWildTemplateStructInt().opdoubled();
    }
    // Wildcard renames expected for these
    {
      new NoChangeStruct().mm1();
      new NoChangeStruct().mm2();
      new NoChangeStruct().mm3();
      new NoChangeStruct().mm4();
      new NoChangeStruct().mm5();
      new NoChangeStruct().opint();
      new SpaceNoChangeStruct().mm1();
      new SpaceNoChangeStruct().mm2();
      new SpaceNoChangeStruct().mm3();
      new SpaceNoChangeStruct().mm4();
      new SpaceNoChangeStruct().mm5();
      new SpaceNoChangeStruct().opint();
    }
  }
}


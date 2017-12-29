
import special_variables.*;

public class special_variables_runme {

  static {
    try {
        System.loadLibrary("special_variables");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    verify(special_variables.ExceptionVars(1.0, 2.0),
           "result = Space::exceptionvars(arg1,arg2);  Space::exceptionvars  ExceptionVars   Java_special_1variables_special_1variablesJNI_ExceptionVars  ");

    verify(special_variables.overloadedmethod(),
           "result = Space::overloadedmethod();  Space::overloadedmethod  overloadedmethod  __SWIG_1 Java_special_1variables_special_1variablesJNI_overloadedmethod_1_1SWIG_11  ");

    verify(special_variables.overloadedmethod(10.0),
          "result = Space::overloadedmethod(arg1);  Space::overloadedmethod  overloadedmethod  __SWIG_0 Java_special_1variables_special_1variablesJNI_overloadedmethod_1_1SWIG_10  ");

    ABC a = new ABC(0, 0.0);
    verify(special_variables.getDeclaration(), "SpaceNamespace::ABC::ABC(int,double) SpaceNamespace::ABC::ABC(int,double)");
    a = new ABC();
    verify(special_variables.getDeclaration(), "SpaceNamespace::ABC::ABC() SpaceNamespace::ABC::ABC()");
    a.instancemethod(1);
    verify(special_variables.getDeclaration(), "short * SpaceNamespace::ABC::instancemethod(int) SpaceNamespace::ABC::instancemethod(int)");
    a.instancemethod(1, false);
    verify(special_variables.getDeclaration(), "short * SpaceNamespace::ABC::instancemethod(int,bool) SpaceNamespace::ABC::instancemethod(int,bool)");
    a.constmethod(1);
    verify(special_variables.getDeclaration(), "short * SpaceNamespace::ABC::constmethod(int) const SpaceNamespace::ABC::constmethod(int) const");
    ABC.staticmethod(0, false);
    verify(special_variables.getDeclaration(), "short * SpaceNamespace::ABC::staticmethod(int,bool) SpaceNamespace::ABC::staticmethod(int,bool)");
    a.delete();
    verify(special_variables.getDeclaration(), "SpaceNamespace::ABC::~ABC() SpaceNamespace::ABC::~ABC()");
    TemplateABC abc = new TemplateABC();
    verify(special_variables.getDeclaration(), "SpaceNamespace::Template< SpaceNamespace::ABC >::Template() SpaceNamespace::Template< SpaceNamespace::ABC >::Template()");
    abc.tmethod(new ABC());
    verify(special_variables.getDeclaration(), "std::string SpaceNamespace::Template< SpaceNamespace::ABC >::tmethod(SpaceNamespace::ABC) SpaceNamespace::Template< SpaceNamespace::ABC >::tmethod(SpaceNamespace::ABC)");
    abc.delete();
    verify(special_variables.getDeclaration(), "SpaceNamespace::Template< SpaceNamespace::ABC >::~Template() SpaceNamespace::Template< SpaceNamespace::ABC >::~Template()");
    special_variables.globtemplate(new TemplateABC());
    verify(special_variables.getDeclaration(), "void SpaceNamespace::globtemplate(SpaceNamespace::Template< SpaceNamespace::ABC >) SpaceNamespace::globtemplate(SpaceNamespace::Template< SpaceNamespace::ABC >)");
  }
  static void verify(String received, String expected) {
    if (!received.equals(expected))
      throw new RuntimeException("Incorrect, received: " + received);
  }
}

%module extend_special_variables

%rename(ForExtensionNewName) ForExtension;
%rename(extended_renamed) ForExtension::extended;

%extend ForExtension {
  ForExtension() {
    return new ForExtension();
  }
  const char* extended() {
    return "name:$name symname:$symname wrapname:$wrapname overname:$overname decl:$decl fulldecl:$fulldecl parentclasssymname:$parentclasssymname parentclassname:$parentclassname";
  }
  const char* extended(int) {
    return "name:$name symname:$symname wrapname:$wrapname overname:$overname decl:$decl fulldecl:$fulldecl parentclasssymname:$parentclasssymname parentclassname:$parentclassname";
  }
}

%inline %{
struct ForExtension {
};
%}

%inline %{
namespace Space {
  template <class T> class ExtendTemplate {};
}
%}

%extend Space::ExtendTemplate
{
 void extending() {
   $parentclassname tmp;
   (void)tmp;
  }
}

%template(ExtendTemplateInt) Space::ExtendTemplate<int>;

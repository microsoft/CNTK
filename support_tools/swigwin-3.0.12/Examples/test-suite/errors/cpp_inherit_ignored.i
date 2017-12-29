%module xxx

// Only non-ignored classes should warn about Ignored base classes
%ignore ActualClass;
%ignore ActualClassNoTemplates;

%{
struct BaseClassNoTemplates {};
%}
%inline %{
template<typename T>
class TemplateClass {};

class ActualClass : public TemplateClass<int> {};
class AktuelKlass : public TemplateClass<int> {};

class ActualClassNoTemplates : public BaseClassNoTemplates {};
class AktuelKlassNoTemplates : public BaseClassNoTemplates {};
%}

// Tests directors and allprotected option when the class does not have the "director" feature
// Was previously crashing and/or generating uncompilable code.

%module(directors="1", allprotected="1") allprotected_not

//%feature("director") AllProtectedNot;
%feature("director") AllProtectedNot::ProtectedMethod;
%feature("director") AllProtectedNot::StaticNonVirtualProtectedMethod;
%feature("director") AllProtectedNot::NonVirtualProtectedMethod;
%feature("director") AllProtectedNot::ProtectedVariable;
%feature("director") AllProtectedNot::StaticProtectedVariable;
%feature("director") AllProtectedNot::PublicMethod;

%inline %{
class AllProtectedNot {
public:
	virtual ~AllProtectedNot() {}
	virtual void PublicMethod() {}
protected:
	virtual void ProtectedMethod() {}
	static void StaticNonVirtualProtectedMethod() {}
	void NonVirtualProtectedMethod() {}
        int ProtectedVariable;
        static int StaticProtectedVariable;
};
int AllProtectedNot::StaticProtectedVariable = 0;
%}

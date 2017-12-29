%module rename_rstrip_encoder

// strip the Cls suffix from all identifiers
%rename("%(rstrip:[Cls])s") ""; 

%inline %{

class SomeThingCls {
};

struct AnotherThingCls {
    void DoClsXCls() {}
};

%}

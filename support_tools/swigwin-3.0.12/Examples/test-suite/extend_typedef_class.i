%module extend_typedef_class

%warnfilter(SWIGWARN_PARSE_EXTEND_NAME) tagCClass;
%warnfilter(SWIGWARN_PARSE_EXTEND_NAME) tagCStruct;

// classes in global namespace
%inline %{
  typedef struct tagAClass {
    int membervar;
  } AClass;

  struct BClass {
    int membervar;
  };

  typedef struct tagCClass {
    int membervar;
  } CClass;

  // Unnamed struct
  typedef struct {
    int membervar;
  } DClass;
%}

%extend tagAClass {
  int getvar() { return $self->membervar; }
}

%extend BClass {
  int getvar() { return $self->membervar; }
}

%extend CClass {
  int getvar() { return $self->membervar; }
}

%extend DClass {
  int getvar() { return $self->membervar; }
}


// classes in a namespace
%inline %{
namespace Space {
  typedef struct tagAStruct {
    int membervar;
  } AStruct;

  struct BStruct {
    int membervar;
  };

  typedef struct tagCStruct {
    int membervar;
  } CStruct;

  // Unnamed struct
  typedef struct {
    int membervar;
  } DStruct;
}
%}

namespace Space {

%extend tagAStruct {
  int getvar() { return $self->membervar; }
}

%extend BStruct {
  int getvar() { return $self->membervar; }
}

%extend CStruct {
  int getvar() { return $self->membervar; }
}

%extend DStruct {
  int getvar() { return $self->membervar; }
}

}


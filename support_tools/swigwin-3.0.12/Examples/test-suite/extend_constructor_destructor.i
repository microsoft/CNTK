%module extend_constructor_destructor

%warnfilter(SWIGWARN_PARSE_EXTEND_NAME) Space::tagCStruct;
%warnfilter(SWIGWARN_PARSE_EXTEND_NAME) tagEStruct;
%warnfilter(SWIGWARN_LANG_EXTEND_CONSTRUCTOR) Space::tagCStruct::CStruct;
%warnfilter(SWIGWARN_LANG_EXTEND_DESTRUCTOR) Space::tagCStruct::~CStruct;
%warnfilter(SWIGWARN_LANG_EXTEND_CONSTRUCTOR) tagEStruct::EStruct;
%warnfilter(SWIGWARN_LANG_EXTEND_DESTRUCTOR) tagEStruct::~EStruct;

%inline %{
int globalVar = 0;

namespace Space {
  typedef struct tagAStruct {
    int ivar;
  } AStruct;

  struct BStruct {
    int ivar;
  };

  typedef struct tagCStruct {
    int ivar;
  } CStruct;

  // Unnamed struct
  typedef struct {
    int ivar;
  } DStruct;

}

typedef struct tagEStruct {
  int ivar;
} EStruct;

namespace Space {
  template<typename T>
  struct FFStruct {
    int ivar;
  };
}
%}

namespace Space {

%extend tagAStruct {
  tagAStruct(int ivar0) {
    Space::AStruct *s = new Space::AStruct();
    s->ivar = ivar0;
    globalVar = ivar0;
    return s;
  }
  ~tagAStruct() {
    globalVar = -$self->ivar;
    delete $self;
  }
}

%extend BStruct {
  BStruct(int ivar0) {
    Space::BStruct *s = new Space::BStruct();
    s->ivar = ivar0;
    globalVar = ivar0;
    return s;
  }
  ~BStruct() {
    globalVar = -$self->ivar;
    delete $self;
  }
}

%extend CStruct {
  CStruct(int ivar0) {
    Space::CStruct *s = new Space::CStruct();
    s->ivar = ivar0;
    globalVar = ivar0;
    return s;
  }
  ~CStruct() {
    globalVar = -$self->ivar;
    delete $self;
  }
}

%extend DStruct {
  DStruct(int ivar0) {
    Space::DStruct *s = new Space::DStruct();
    s->ivar = ivar0;
    globalVar = ivar0;
    return s;
  }
  ~DStruct() {
    globalVar = -$self->ivar;
    delete $self;
  }
}

}

%extend EStruct {
  EStruct(int ivar0) {
    EStruct *s = new EStruct();
    s->ivar = ivar0;
    globalVar = ivar0;
    return s;
  }
  ~EStruct() {
    globalVar = -$self->ivar;
    delete $self;
  }
}

namespace Space {
%extend FFStruct {
  FFStruct(int ivar0) {
    Space::FFStruct<T> *s = new Space::FFStruct<T>();
    s->ivar = ivar0;
    globalVar = ivar0;
    return s;
  }
  ~FFStruct() {
    globalVar = -$self->ivar;
    delete $self;
  }
}

}

%template(FStruct) Space::FFStruct<long>;
%template(GStruct) Space::FFStruct<char>;


%module xxx

/////////////////////////////
%extend AStruct {
  ~AStruct() {}
}
struct AStruct {
  ~AStruct() {}
};

/////////////////////////////
struct BStruct {
  ~BStruct() {}
  ~BStruct() {}
};

/////////////////////////////
struct CStruct {
};

%extend CStruct {
  ~NOT_CStruct() {
    delete $self;
  }
}

%extend DStruct {
  ~NOT_DStruct() {
    delete $self;
  }
}

struct DStruct {
};

/////////////////////////////
struct EStruct {
  ~EStruct() {}
};

%extend EStruct {
  ~NOT_EStruct() {
    delete $self;
  }
}

%extend FStruct {
  ~NOT_FStruct() {
    delete $self;
  }
}

struct FStruct {
  ~FStruct() {}
};

/////////////////////////////
struct GStruct {
};

%extend GStruct {
  ~GStruct() {}
  ~NOT_GStruct() {
    delete $self;
  }
}

%extend HStruct {
  ~HStruct() {}
  ~NOT_HStruct() {
    delete $self;
  }
}

struct HStruct {
};

/////////////////////////////
struct IStruct {
  ~IStruct() {}
  ~NOT_IStruct() {}
};

struct JStruct {
  ~JStruct() {}
  ~NOT_JStruct() {}
  ~JStruct() {}
};

/////////////////////////////
struct KStruct {
  ~NOT_KStruct() {}
};

/////////////////////////////
template<typename T>
struct LStruct {
  ~LStruct() {}
  ~NOT_LStruct() {}
  ~LStruct() {}
};
%template(LStructInt) LStruct<int>;
%template(LStructShort) LStruct<short>;


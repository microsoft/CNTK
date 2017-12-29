%module xxx

typedef struct {
  int myint;
} StructA;

typedef struct StructBName {
  int myint;
} StructB;

typedef struct StructC {
  int myint;
} StructC;

%extend StructA {
  void method() {}
}

%extend StructB {
  void method() {}
}

%extend StructC {
  void method() {}
}

struct StructD {
  int myint;
};
typedef struct StructD StructDName;

%extend StructDName {
  void method() {}
}


typedef struct stru_struct {
    int bar;
} stru;
typedef union uni_union {
    int un1;
    double un2;
} uni;

%extend stru {
    stru() {
        stru* s = (stru*)malloc(sizeof(stru));
        s->bar = 11;
        return s;
    }
    ~stru() {
      free($self);
    }
}

%extend uni {
  uni() { return 0; }
  ~uni() { free($self); }
}


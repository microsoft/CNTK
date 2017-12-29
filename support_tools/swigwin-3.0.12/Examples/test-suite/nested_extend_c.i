%module nested_extend_c

#if defined(SWIG_JAVASCRIPT_V8)

%inline %{
#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
/* for nested C class wrappers compiled as C++ code */
/* dereferencing type-punned pointer will break strict-aliasing rules [-Werror=strict-aliasing] */
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
%}

#endif

#if !defined(SWIGOCTAVE) && !defined(SWIG_JAVASCRIPT_V8)
%extend hiA {
  hiA() {
   union hiA *self = (union hiA *)malloc(sizeof(union hiA));
   self->c = 'h';
   return self;
  }
  char hi_extend() {
    return $self->c;
  }
  static const long swig_size = sizeof(union hiA);
}
%extend lowA {
  lowA() {
    struct lowA *self = (struct lowA *)malloc(sizeof(struct lowA));
    self->name = 0;
    self->num = 99;
    return self;
  }
  int low_extend() {
    return $self->num;
  }
  static const long swig_size = sizeof(struct lowA);
}

%extend hiB {
  hiB() {
   union hiB *self = (union hiB *)malloc(sizeof(union hiB));
   self->c = 'h';
   return self;
  }
  char hi_extend() {
    return $self->c;
  }
  static const long swig_size = sizeof(union hiB);
}
%extend lowB {
  lowB() {
    struct lowB *self = (struct lowB *)malloc(sizeof(struct lowB));
    self->name = 0;
    self->num = 99;
    return self;
  }
  int low_extend() {
    return $self->num;
  }
  static const long swig_size = sizeof(struct lowB);
}

%extend FOO_bar {
    void bar_extend()	{
        $self->d = 1;
    }
};
%extend NestedA {
  static const long swig_size = sizeof(struct NestedA);
}

#endif

%inline %{
typedef struct NestedA {
    int a;
    union hiA {
        char c;
        int d;
    } hiA_instance;

    struct lowA {
        char *name;
        int num;
    } lowA_instance;
} NestedA;

typedef struct {
    int a;
    union hiB {
        char c;
        int d;
    } hiB_instance;

    struct lowB {
        char *name;
        int num;
    } lowB_instance;
} NestedB;

typedef struct {
    int a;
    union {
        char c;
        int d;
    } bar;
} FOO;

static struct {
   int i;
} THING;

int useThing() {
  return THING.i;
}
%}


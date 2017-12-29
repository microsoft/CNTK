%module(directors="1") virtual_poly

%warnfilter(SWIGWARN_JAVA_COVARIANT_RET, SWIGWARN_CSHARP_COVARIANT_RET) copy; /* Java, C# covariant return types */
%warnfilter(SWIGWARN_JAVA_COVARIANT_RET, SWIGWARN_CSHARP_COVARIANT_RET) ref_this; /* Java, C# covariant return types */
%warnfilter(SWIGWARN_JAVA_COVARIANT_RET, SWIGWARN_CSHARP_COVARIANT_RET) covariant; /* Java, C# covariant return types */
%warnfilter(SWIGWARN_JAVA_COVARIANT_RET, SWIGWARN_CSHARP_COVARIANT_RET) covariant2; /* Java, C# covariant return types */
%warnfilter(SWIGWARN_JAVA_COVARIANT_RET, SWIGWARN_CSHARP_COVARIANT_RET) covariant3; /* Java, C# covariant return types */
%warnfilter(SWIGWARN_JAVA_COVARIANT_RET, SWIGWARN_CSHARP_COVARIANT_RET) covariant4; /* Java, C# covariant return types */

%{
#if defined(__SUNPRO_CC)
#pragma error_messages (off, wbadasg) /* Assigning extern "C" ... */
#endif
%}

//
// Check this example with directors wherever possible.
//
//%feature("director");

// This shouldn't get used.
// %newobject *::copy();

%newobject *::copy() const;


%inline %{  
  struct NNumber
  {
    virtual ~NNumber() {};
    virtual NNumber* copy() const = 0;
    virtual NNumber& ref_this() 
    {
      return *this;
    }
    
    
    NNumber* nnumber() 
    {
      return this;
    }
    

  };
  
  /* 
     NInt and NDouble are both NNumber derived classes, but they
     have more different than common attributes.
     
     In particular the function 'get', that is type dependent, can't
     be included in the NNumber abstract interface.

     For this reason, the virtual 'copy' method has a polymorphic (covariant)
     return type, since in most of the cases we don't want to lose the
     original object type, which is very very important.

     Using the polymorphic return type reduced greatly the need of
     using 'dynamic_cast' at the C++ side, and at the target languages
     that support it.
   */
  struct NInt : NNumber
  {
    NInt(int v) : val(v) 
    {
    }
    
    int get() const
    {
      return val;
    }
    
    virtual NInt* copy() const
    {
      return new NInt(val);
    }

    virtual NInt& ref_this() 
    {
      return *this;
    }

    /* See below */
    static NInt* narrow(NNumber* nn);

    
  private:
    int val;
  };

  inline NInt& incr(NInt& i) {
    i = i.get() + 1;
    return i;
  }

  struct NDouble : NNumber
  {
    NDouble(double v) : val(v) 
    {
    }
    
    double get() const
    {
      return val;
    }
    
    virtual NDouble* copy() const
    {
      return new NDouble(val);
    }

    virtual NDouble& ref_this() 
    {
      return *this;
    }

    /* See below */
    static NDouble* narrow(NNumber* nn);
    
  private:
    double val;
  };

  /*
     Java and C# do not support the polymorphic (covariant) return types used
     in the copy method. So, they just emit 'plain' copy functions as if this is
     being wrapped instead:
    
      NNumber* NNumber::copy() const;
      NNumber* NInt::copy() const;  
      NNumber* NDouble::copy() const;
    
     However, since the objects provide their own downcasting
     mechanism, the narrow methods similar to the CORBA mechanism,
     could be used, otherwise use the Java/C# downcasts.
  */
  inline NInt* NInt::narrow(NNumber* n) {
    // this is just a plain C++ dynamic_cast, but in theory the user
    // could use whatever he wants.
    return dynamic_cast<NInt*>(n);
  }  
  
  inline NDouble* NDouble::narrow(NNumber* n) {
    return dynamic_cast<NDouble*>(n);
  }
%}

%inline %{

// These three classes test covariant return types and whether swig accurately matches
// polymorphic methods (mainly for C# override keyword). Also tests methods which hide
// the base class' method (for C#, new keyword required on method declaration).

typedef int* IntegerPtr;
typedef double Double;

template<typename T> struct Base {
  T t;
  virtual IntegerPtr method() const = 0;
  virtual IntegerPtr foxy() const = 0;
  virtual IntegerPtr foxy(int a) const = 0;
  virtual int * foxy(int*& a) { return 0; }
  virtual double afunction() = 0;
  virtual IntegerPtr defaultargs(double d, int * a = 0) = 0;
  static void StaticHidden() {}
  void AmIAmINotVirtual() {}
  IntegerPtr NotVirtual(IntegerPtr i) { return 0; }
  virtual Base * covariant(int a = 0, int * i = 0) { return 0; }
  typedef Base * BasePtr;
  virtual BasePtr covariant2() { return 0; }
  virtual BasePtr covariant3() { return 0; }
  virtual ~Base() {}
};

template<typename T> struct Derived : Base<T> {
  int * method() const { return 0; }
  IntegerPtr foxy() const { return 0; }
  int * foxy(int a) const { return 0; }
  virtual int * foxy(int*& a) { return 0; }
  Double afunction() { return 0; }
  int * defaultargs(Double d, IntegerPtr a = 0) { return 0; }
  void AmIAmINotVirtual() {}
  int * NotVirtual(int *i) { return 0; }
  typedef Derived * DerivedPtr;
  DerivedPtr covariant(int a = 0, IntegerPtr i = 0) { return 0; }
  DerivedPtr covariant2() { return 0; }
  Derived<T> * covariant3() { return 0; }
  virtual Derived<T> * covariant4(double d) { return 0; }
  virtual int IsVirtual() { return 0; }
};

template<typename T> struct Bottom : Derived<T> {
  int * method() const { return 0; }
  static void StaticHidden() {}
  void AmIAmINotVirtual() {}
  IntegerPtr NotVirtual(IntegerPtr i) { return 0; }
  void (*funcptr)(int a, bool b);
  Bottom<T> * covariant(int a = 0, IntegerPtr i = 0) { return 0; }
  Derived<T> * covariant2() { return 0; }
  Bottom<T> * covariant3() { return 0; }
  Bottom<T> * covariant4(double d) { return 0; }
  int IsVirtual() { return 0; }
};
%}


%template(BaseInt) Base<int>;
%template(DerivedInt) Derived<int>;
%template(BottomInt) Bottom<int>;



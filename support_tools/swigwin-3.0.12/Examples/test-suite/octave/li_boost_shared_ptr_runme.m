# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

1;
li_boost_shared_ptr;

function verifyValue(expected, got)
    if (expected ~= got)
      error("verify value failed.");% Expected: ", expected, " Got: ", got)
    end
endfunction

function verifyCount(expected, k)
    got = use_count(k);
    if (expected ~= got)
      error("verify use_count failed. Expected: %d  Got: %d ", expected, got);
    end
endfunction

function runtest()
    li_boost_shared_ptr; # KTTODO this needs to be here at present. Global module failure?
    # simple shared_ptr usage - created in C++
    k = Klass("me oh my");
    val = k.getValue();
    verifyValue("me oh my", val)
    verifyCount(1, k)

    # simple shared_ptr usage - not created in C++
    k = factorycreate();
    val = k.getValue();
    verifyValue("factorycreate", val)
    verifyCount(1, k)

    # pass by shared_ptr
    k = Klass("me oh my");
    kret = smartpointertest(k);
    val = kret.getValue();
    verifyValue("me oh my smartpointertest", val)
    verifyCount(2, k)
    verifyCount(2, kret)

    # pass by shared_ptr pointer
    k = Klass("me oh my");
    kret = smartpointerpointertest(k);
    val = kret.getValue();
    verifyValue("me oh my smartpointerpointertest", val)
    verifyCount(2, k)
    verifyCount(2, kret)

    # pass by shared_ptr reference
    k = Klass("me oh my");
    kret = smartpointerreftest(k);
    val = kret.getValue();
    verifyValue("me oh my smartpointerreftest", val)
    verifyCount(2, k)
    verifyCount(2, kret)

    # pass by shared_ptr pointer reference
    k = Klass("me oh my");
    kret = smartpointerpointerreftest(k);
    val = kret.getValue();
    verifyValue("me oh my smartpointerpointerreftest", val)
    verifyCount(2, k)
    verifyCount(2, kret)

    # const pass by shared_ptr
    k = Klass("me oh my");
    kret = constsmartpointertest(k);
    val = kret.getValue();
    verifyValue("me oh my", val)
    verifyCount(2, k)
    verifyCount(2, kret)

    # const pass by shared_ptr pointer
    k = Klass("me oh my");
    kret = constsmartpointerpointertest(k);
    val = kret.getValue();
    verifyValue("me oh my", val)
    verifyCount(2, k)
    verifyCount(2, kret)

    # const pass by shared_ptr reference
    k = Klass("me oh my");
    kret = constsmartpointerreftest(k);
    val = kret.getValue();
    verifyValue("me oh my", val)
    verifyCount(2, k)
    verifyCount(2, kret)

    # pass by value
    k = Klass("me oh my");
    kret = valuetest(k);
    val = kret.getValue();
    verifyValue("me oh my valuetest", val)
    verifyCount(1, k)
    verifyCount(1, kret)

    # pass by pointer
    k = Klass("me oh my");
    kret = pointertest(k);
    val = kret.getValue();
    verifyValue("me oh my pointertest", val)
    verifyCount(1, k)
    verifyCount(1, kret)

    # pass by reference
    k = Klass("me oh my");
    kret = reftest(k);
    val = kret.getValue();
    verifyValue("me oh my reftest", val)
    verifyCount(1, k)
    verifyCount(1, kret)

    # pass by pointer reference
    k = Klass("me oh my");
    kret = pointerreftest(k);
    val = kret.getValue();
    verifyValue("me oh my pointerreftest", val)
    verifyCount(1, k)
    verifyCount(1, kret)

    # null tests
    #KTODO None not defined
    # k = None;

    # if (smartpointertest(k) ~= None)
    #   error("return was not null")
    # end

    # if (smartpointerpointertest(k) ~= None)
    #   error("return was not null")
    # end

    # if (smartpointerreftest(k) ~= None)
    #   error("return was not null")
    # end

    # if (smartpointerpointerreftest(k) ~= None)
    #   error("return was not null")
    # end

    # if (nullsmartpointerpointertest(None) ~= "null pointer")
    #   error("not null smartpointer pointer")
    # end

    # # try:
    # #   valuetest(k)
    # #   error("Failed to catch null pointer")
    # # except ValueError:
    # #   pass

    # if (pointertest(k) ~= None)
    #   error("return was not null")
    # end

    # # try:
    # #   reftest(k)
    # #   error("Failed to catch null pointer")
    # # except ValueError:
    # #   pass

    # $owner
    k = pointerownertest();
    val = k.getValue();
    verifyValue("pointerownertest", val)
    verifyCount(1, k)
    k = smartpointerpointerownertest();
    val = k.getValue();
    verifyValue("smartpointerpointerownertest", val)
    verifyCount(1, k)

    # //////////////////////////////// Derived class ////////////////////////////////////////
    # derived pass by shared_ptr
    k = KlassDerived("me oh my");
    kret = derivedsmartptrtest(k);
    val = kret.getValue();
    verifyValue("me oh my derivedsmartptrtest-Derived", val)
    verifyCount(2, k)
    verifyCount(2, kret)

    # derived pass by shared_ptr pointer
    k = KlassDerived("me oh my");
    kret = derivedsmartptrpointertest(k);
    val = kret.getValue();
    verifyValue("me oh my derivedsmartptrpointertest-Derived", val)
    verifyCount(2, k)
    verifyCount(2, kret)

    # derived pass by shared_ptr ref
    k = KlassDerived("me oh my");
    kret = derivedsmartptrreftest(k);
    val = kret.getValue();
    verifyValue("me oh my derivedsmartptrreftest-Derived", val)
    verifyCount(2, k)
    verifyCount(2, kret)

    # derived pass by shared_ptr pointer ref
    k = KlassDerived("me oh my");
    kret = derivedsmartptrpointerreftest(k);
    val = kret.getValue();
    verifyValue("me oh my derivedsmartptrpointerreftest-Derived", val)
    verifyCount(2, k)
    verifyCount(2, kret)

    # derived pass by pointer
    k = KlassDerived("me oh my");
    kret = derivedpointertest(k);
    val = kret.getValue();
    verifyValue("me oh my derivedpointertest-Derived", val)
    verifyCount(1, k)
    verifyCount(1, kret)

    # derived pass by ref
    k = KlassDerived("me oh my");
    kret = derivedreftest(k);
    val = kret.getValue();
    verifyValue("me oh my derivedreftest-Derived", val)
    verifyCount(1, k)
    verifyCount(1, kret)

     # //////////////////////////////// Derived and base class mixed ////////////////////////////////////////
     # pass by shared_ptr (mixed)
     k = KlassDerived("me oh my");
     kret = smartpointertest(k);
     val = kret.getValue();
     verifyValue("me oh my smartpointertest-Derived", val)
     verifyCount(2, k)
     verifyCount(2, kret)

     # pass by shared_ptr pointer (mixed)
     k = KlassDerived("me oh my");
     kret = smartpointerpointertest(k);
     val = kret.getValue();
     verifyValue("me oh my smartpointerpointertest-Derived", val)
     verifyCount(2, k)
     verifyCount(2, kret)

     # pass by shared_ptr reference (mixed)
     k = KlassDerived("me oh my");
     kret = smartpointerreftest(k);
     val = kret.getValue();
     verifyValue("me oh my smartpointerreftest-Derived", val)
     verifyCount(2, k)
     verifyCount(2, kret)

     # pass by shared_ptr pointer reference (mixed)
     k = KlassDerived("me oh my");
     kret = smartpointerpointerreftest(k);
     val = kret.getValue();
     verifyValue("me oh my smartpointerpointerreftest-Derived", val)
     verifyCount(2, k)
     verifyCount(2, kret)

     # pass by value (mixed)
     k = KlassDerived("me oh my");
     kret = valuetest(k);
     val = kret.getValue();
     verifyValue("me oh my valuetest", val) # note slicing
     verifyCount(1, k)
     verifyCount(1, kret)

     # pass by pointer (mixed)
     k = KlassDerived("me oh my");
     kret = pointertest(k);
     val = kret.getValue();
     verifyValue("me oh my pointertest-Derived", val)
     verifyCount(1, k)
     verifyCount(1, kret)
     # pass by ref (mixed)
     k = KlassDerived("me oh my");
     kret = reftest(k);
     val = kret.getValue();
     verifyValue("me oh my reftest-Derived", val)
     verifyCount(1, k)
     verifyCount(1, kret)

    # //////////////////////////////// Overloading tests ////////////////////////////////////////
    # Base class
    k = Klass("me oh my");
    verifyValue(overload_rawbyval(k), "rawbyval")
    verifyValue(overload_rawbyref(k), "rawbyref")
    verifyValue(overload_rawbyptr(k), "rawbyptr")
    verifyValue(overload_rawbyptrref(k), "rawbyptrref")

    verifyValue(overload_smartbyval(k), "smartbyval")
    verifyValue(overload_smartbyref(k), "smartbyref")
    verifyValue(overload_smartbyptr(k), "smartbyptr")
    verifyValue(overload_smartbyptrref(k), "smartbyptrref")

    # Derived class
    k = KlassDerived("me oh my");
    verifyValue(overload_rawbyval(k), "rawbyval")
    verifyValue(overload_rawbyref(k), "rawbyref")
    verifyValue(overload_rawbyptr(k), "rawbyptr")
    verifyValue(overload_rawbyptrref(k), "rawbyptrref")

    verifyValue(overload_smartbyval(k), "smartbyval")
    verifyValue(overload_smartbyref(k), "smartbyref")
    verifyValue(overload_smartbyptr(k), "smartbyptr")
    verifyValue(overload_smartbyptrref(k), "smartbyptrref")

    # 3rd derived class
    k = Klass3rdDerived("me oh my");
    val = k.getValue();
    verifyValue("me oh my-3rdDerived", val)
    verifyCount(1, k)

    val = test3rdupcast(k);
    verifyValue("me oh my-3rdDerived", val)
    verifyCount(1, k)

    # //////////////////////////////// Member variables ////////////////////////////////////////
    # smart pointer by value
    m = MemberVariables();
    k = Klass("smart member value");
    m.SmartMemberValue = k;
    val = k.getValue();
    verifyValue("smart member value", val)
    verifyCount(2, k)

    kmember = m.SmartMemberValue;
    val = kmember.getValue();
    verifyValue("smart member value", val)
    verifyCount(3, kmember)
    verifyCount(3, k)

    clear m
    verifyCount(2, kmember)
    verifyCount(2, k)

    # smart pointer by pointer
    m = MemberVariables();
    k = Klass("smart member pointer");
    m.SmartMemberPointer = k;
    val = k.getValue();
    verifyValue("smart member pointer", val)
    verifyCount(1, k)

    kmember = m.SmartMemberPointer;
    val = kmember.getValue();
    verifyValue("smart member pointer", val)
    verifyCount(2, kmember)
    verifyCount(2, k)

    clear m
    verifyCount(2, kmember)
    verifyCount(2, k)

    # smart pointer by reference
    m = MemberVariables();
    k = Klass("smart member reference");
    m.SmartMemberReference = k;
    val = k.getValue();
    verifyValue("smart member reference", val)
    verifyCount(2, k)

    kmember = m.SmartMemberReference;
    val = kmember.getValue();
    verifyValue("smart member reference", val)
    verifyCount(3, kmember)
    verifyCount(3, k)

    # The C++ reference refers to SmartMemberValue...
    kmemberVal = m.SmartMemberValue;
    val = kmember.getValue();
    verifyValue("smart member reference", val)
    verifyCount(4, kmemberVal)
    verifyCount(4, kmember)
    verifyCount(4, k)

    clear m
    verifyCount(3, kmemberVal)
    verifyCount(3, kmember)
    verifyCount(3, k)

    # plain by value
    m = MemberVariables();
    k = Klass("plain member value");
    m.MemberValue = k;
    val = k.getValue();
    verifyValue("plain member value", val)
    verifyCount(1, k)

    kmember = m.MemberValue;
    val = kmember.getValue();
    verifyValue("plain member value", val)
    verifyCount(1, kmember)
    verifyCount(1, k)

    clear m
    verifyCount(1, kmember)
    verifyCount(1, k)

    # plain by pointer
    m = MemberVariables();
    k = Klass("plain member pointer");
    m.MemberPointer = k;
    val = k.getValue();
    verifyValue("plain member pointer", val)
    verifyCount(1, k)

    kmember = m.MemberPointer;
    val = kmember.getValue();
    verifyValue("plain member pointer", val)
    verifyCount(1, kmember)
    verifyCount(1, k)

    clear m
    verifyCount(1, kmember)
    verifyCount(1, k)

    # plain by reference
    m = MemberVariables();
    k = Klass("plain member reference");
    m.MemberReference = k;
    val = k.getValue();
    verifyValue("plain member reference", val)
    verifyCount(1, k)

    kmember = m.MemberReference;
    val = kmember.getValue();
    verifyValue("plain member reference", val)
    verifyCount(1, kmember)
    verifyCount(1, k)

    clear m
    verifyCount(1, kmember)
    verifyCount(1, k)

    # null member variables
    m = MemberVariables();

    # shared_ptr by value
    k = m.SmartMemberValue;
    #KTODO None not defined
    # if (k ~= None)
    #   error("expected null")
    # end

    # m.SmartMemberValue = None
    # k = m.SmartMemberValue
    # if (k ~= None)
    #   error("expected null")
    # end
    # verifyCount(0, k)

    # # plain by value
    # # try:
    # #   m.MemberValue = None
    # #   error("Failed to catch null pointer")
    # # except ValueError:
    # #   pass

    # # ////////////////////////////////// Global variables ////////////////////////////////////////
    # # smart pointer
    # kglobal = cvar.GlobalSmartValue
    # if (kglobal ~= None)
    #   error("expected null")
    # end

    k = Klass("smart global value");
    cvar.GlobalSmartValue = k;
    verifyCount(2, k)

    kglobal = cvar.GlobalSmartValue;
    val = kglobal.getValue();
    verifyValue("smart global value", val)
    verifyCount(3, kglobal)
    verifyCount(3, k)
    verifyValue("smart global value", cvar.GlobalSmartValue.getValue())
    #KTTODO cvar.GlobalSmartValue = None

    # plain value
    k = Klass("global value");
    cvar.GlobalValue = k;
    verifyCount(1, k)

    kglobal = cvar.GlobalValue;
    val = kglobal.getValue();
    verifyValue("global value", val)
    verifyCount(1, kglobal)
    verifyCount(1, k)
    verifyValue("global value", cvar.GlobalValue.getValue())

    # try:
    #   cvar.GlobalValue = None
    #   error("Failed to catch null pointer")
    # except ValueError:
    #   pass

    # plain pointer
    kglobal = cvar.GlobalPointer;
    #KTODO if (kglobal ~= None)
    #KTODO   error("expected null")
    #KTODO end

    k = Klass("global pointer");
    cvar.GlobalPointer = k;
    verifyCount(1, k)

    kglobal = cvar.GlobalPointer;
    val = kglobal.getValue();
    verifyValue("global pointer", val)
    verifyCount(1, kglobal)
    verifyCount(1, k)
    #KTODO cvar.GlobalPointer = None

    # plain reference
    k = Klass("global reference");
    cvar.GlobalReference = k;
    verifyCount(1, k)

    kglobal = cvar.GlobalReference;
    val = kglobal.getValue();
    verifyValue("global reference", val)
    verifyCount(1, kglobal)
    verifyCount(1, k)

    # try:
    #   cvar.GlobalReference = None 
    #   error("Failed to catch null pointer")
    # except ValueError:
    #   pass

    # ////////////////////////////////// Templates ////////////////////////////////////////
    pid = PairIntDouble(10, 20.2);
    if (pid.baseVal1 ~= 20 || pid.baseVal2 ~= 40.4)
      error("Base values wrong")
    end
    if (pid.val1 ~= 10 || pid.val2 ~= 20.2)
      error("Derived Values wrong")
    end

endfunction

debug = false;%true;

    if (debug)
      fprintf( "Started\n" )
    end

    cvar.debug_shared = debug;

    # Change loop count to run for a long time to monitor memory
    loopCount = 1; #5000
    for i=0:loopCount
      runtest()
    end

    # Expect 1 instance - the one global variable (GlobalValue)
    #KTTODO next fails, possibly because we commented GlobalSmartValue=None
    #if (Klass.getTotal_count() ~= 1)
    #  error("Klass.total_count=%d", Klass.getTotal_count())
    #end

    wrapper_count = shared_ptr_wrapper_count() ;
    #KTTODO next fails as NOT_COUNTING not in octave name space, so we hard-wire it here
    #if (wrapper_count ~= NOT_COUNTING)
    if (wrapper_count ~= -123456)
      # Expect 1 instance - the one global variable (GlobalSmartValue)
      if (wrapper_count ~= 1)
        error("shared_ptr wrapper count=%s", wrapper_count)
      end
    end

    if (debug)
      fprintf( "Finished\n" )
    end

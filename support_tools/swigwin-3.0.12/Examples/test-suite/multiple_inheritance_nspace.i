// This is a copy of the multiple_inheritance_abstract test
%module  multiple_inheritance_nspace

%warnfilter(SWIGWARN_RUBY_MULTIPLE_INHERITANCE,
	    SWIGWARN_D_MULTIPLE_INHERITANCE,
	    SWIGWARN_PHP_MULTIPLE_INHERITANCE); /* languages not supporting multiple inheritance or %interface */

// nspace feature only supported by these languages
#if defined(SWIGJAVA) || defined(SWIGCSHARP) || defined(SWIGD) || defined(SWIGLUA) || defined(SWIGJAVASCRIPT)
%nspace;
#endif

#if defined(SWIGJAVA) || defined(SWIGCSHARP)
%include "swiginterface.i"
%interface(Space::ABase1)
%interface(Space::CBase1)
%interface(Space::CBase2)
#endif

#if defined(SWIGD)
// Missing multiple inheritance support results in incorrect use of override
%ignore CBase1;
%ignore CBase2;
#endif

#if defined(SWIGJAVA)
SWIG_JAVABODY_PROXY(public, public, SWIGTYPE)
#endif

%inline %{
namespace Space {
  struct CBase1 {
    virtual void cbase1x() {
      return;
    } 
    virtual int cbase1y() {
      return 1;
    }
    int cbase1z() {
      return 10;
    }
    virtual ~CBase1() {
    }
  };

  struct CBase2 {
    virtual int cbase2() {
      return 2;
    }
    virtual ~CBase2() {
    }
  };

  struct ABase1 {
    virtual int abase1() = 0;
    virtual ~ABase1() {
    }
  };

  struct Derived1 : CBase2, CBase1 {
    virtual void cbase1x() {
      return;
    }
    virtual int cbase1y() {
      return 3;
    }
    virtual int cbase2() {
      return 4;
    }
    virtual CBase2 *cloneit() {
      return new Derived1(*this);
    }
    void derived1() {
    }
  };

  struct Derived2 : CBase1, ABase1 {
    virtual void cbase1x() {
      return;
    }
    virtual int cbase1y() {
      return 6;
    }
    virtual int abase1() {
      return 5;
    }
    virtual CBase1 *cloneit() {
      return new Derived2(*this);
    }
    void derived2() {
    }
  };

  struct Derived3 : ABase1, CBase1, CBase2 {
    virtual int cbase1y() {
      return 7;
    }
    virtual int cbase2() {
      return 8;
    }
    virtual int abase1() {
      return 9;
    }
    virtual void cbase1x() {
    }
    virtual ABase1 *cloneit() {
      return new Derived3(*this);
    }
    void derived3() {
    }
  };

  struct Bottom1 : Derived1 {
    virtual void cbase1x() {
      return;
    }
    virtual int cbase1y() {
      return 103;
    }
    virtual int cbase2() {
      return 104;
    }
  };

  struct Bottom2 : Derived2 {
    virtual int cbase1y() {
      return 206;
    }
    virtual int abase1() {
      return 205;
    }
  };

  struct Bottom3 : Derived3 {
    virtual int cbase1y() {
      return 307;
    }
    virtual int cbase2() {
      return 308;
    }
    virtual int abase1() {
      return 309;
    }
  };

  // Base classes as input
  int InputValCBase1(CBase1 cb1) {
    return cb1.cbase1y();
  }
  int InputValCBase2(CBase2 cb2) {
    return cb2.cbase2();
  }

  int InputPtrABase1(ABase1 *pab1) {
    return pab1->abase1();
  }
  int InputPtrCBase1(CBase1 *pcb1) {
    return pcb1->cbase1y();
  }
  int InputPtrCBase2(CBase2 *pcb2) {
    return pcb2->cbase2();
  }

  int InputRefABase1(ABase1 &rab1) {
    return rab1.abase1();
  }
  int InputRefCBase1(CBase1 &rcb1) {
    return rcb1.cbase1y();
  }
  int InputRefCBase2(CBase2 &rcb2) {
    return rcb2.cbase2();
  }

  int InputCPtrRefABase1(ABase1 *const& pab1) {
    return pab1->abase1();
  }
  int InputCPtrRefCBase1(CBase1 *const& pcb1) {
    return pcb1->cbase1y();
  }
  int InputCPtrRefCBase2(CBase2 *const& pcb2) {
    return pcb2->cbase2();
  }

  // Derived classes as input
  int InputValDerived1(Derived1 d) {
    return d.cbase1y() + d.cbase2();
  }
  int InputValDerived2(Derived2 d) {
    return d.cbase1y() + d.abase1();
  }
  int InputValDerived3(Derived3 d) {
    return d.cbase1y() + d.cbase2() + d.abase1();
  }

  int InputRefDerived1(Derived1 &d) {
    return d.cbase1y() + d.cbase2();
  }
  int InputRefDerived2(Derived2 &d) {
    return d.cbase1y() + d.abase1();
  }
  int InputRefDerived3(Derived3 &d) {
    return d.cbase1y() + d.cbase2() + d.abase1();
  }

  int InputPtrDerived1(Derived1 *d) {
    return d->cbase1y() + d->cbase2();
  }
  int InputPtrDerived2(Derived2 *d) {
    return d->cbase1y() + d->abase1();
  }
  int InputPtrDerived3(Derived3 *d) {
    return d->cbase1y() + d->cbase2() + d->abase1();
  }

  int InputCPtrRefDerived1(Derived1 *const& d) {
    return d->cbase1y() + d->cbase2();
  }
  int InputCPtrRefDerived2(Derived2 *const& d) {
    return d->cbase1y() + d->abase1();
  }
  int InputCPtrRefDerived3(Derived3 *const& d) {
    return d->cbase1y() + d->cbase2() + d->abase1();
  }

  // Bottom classes as input
  int InputValBottom1(Bottom1 d) {
    return d.cbase1y() + d.cbase2();
  }
  int InputValBottom2(Bottom2 d) {
    return d.cbase1y() + d.abase1();
  }
  int InputValBottom3(Bottom3 d) {
    return d.cbase1y() + d.cbase2() + d.abase1();
  }

  int InputRefBottom1(Bottom1 &d) {
    return d.cbase1y() + d.cbase2();
  }
  int InputRefBottom2(Bottom2 &d) {
    return d.cbase1y() + d.abase1();
  }
  int InputRefBottom3(Bottom3 &d) {
    return d.cbase1y() + d.cbase2() + d.abase1();
  }

  int InputPtrBottom1(Bottom1 *d) {
    return d->cbase1y() + d->cbase2();
  }
  int InputPtrBottom2(Bottom2 *d) {
    return d->cbase1y() + d->abase1();
  }
  int InputPtrBottom3(Bottom3 *d) {
    return d->cbase1y() + d->cbase2() + d->abase1();
  }

  int InputCPtrRefBottom1(Bottom1 *const& d) {
    return d->cbase1y() + d->cbase2();
  }
  int InputCPtrRefBottom2(Bottom2 *const& d) {
    return d->cbase1y() + d->abase1();
  }
  int InputCPtrRefBottom3(Bottom3 *const& d) {
    return d->cbase1y() + d->cbase2() + d->abase1();
  }

  // Return pointers
  CBase1 *MakePtrDerived1_CBase1() {
    return new Derived1();
  }
  CBase2 *MakePtrDerived1_CBase2() {
    return new Derived1();
  }
  CBase1 *MakePtrDerived2_CBase1() {
    return new Derived2();
  }
  ABase1 *MakePtrDerived2_ABase1() {
    return new Derived2();
  }
  ABase1 *MakePtrDerived3_ABase1() {
    return new Derived3();
  }
  CBase1 *MakePtrDerived3_CBase1() {
    return new Derived3();
  }
  CBase2 *MakePtrDerived3_CBase2() {
    return new Derived3();
  }

  // Return references
  CBase1 &MakeRefDerived1_CBase1() {
    static Derived1 d;
    return d;
  }
  CBase2 &MakeRefDerived1_CBase2() {
    static Derived1 d;
    return d;
  }
  CBase1 &MakeRefDerived2_CBase1() {
    static Derived2 d;
    return d;
  }
  ABase1 &MakeRefDerived2_ABase1() {
    static Derived2 d;
    return d;
  }
  ABase1 &MakeRefDerived3_ABase1() {
    static Derived3 d;
    return d;
  }
  CBase1 &MakeRefDerived3_CBase1() {
    static Derived3 d;
    return d;
  }
  CBase2 &MakeRefDerived3_CBase2() {
    static Derived3 d;
    return d;
  }

  // Return by value (sliced objects)
  CBase1 MakeValDerived1_CBase1() {
    return Derived1();
  }
  CBase2 MakeValDerived1_CBase2() {
    return Derived1();
  }
  CBase1 MakeValDerived2_CBase1() {
    return Derived2();
  }
  CBase1 MakeValDerived3_CBase1() {
    return Derived3();
  }
  CBase2 MakeValDerived3_CBase2() {
    return Derived3();
  }
}

%}


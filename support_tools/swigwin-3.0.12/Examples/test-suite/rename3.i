// Test %rename directive within %extend
%module rename3

namespace Space {
  // Note no namespace nor class qualification
  %extend XYZ {
    %rename(opIntPtrA) operator NotXYZ<int>*() const;
    %rename(opIntPtrB) operator XYZ<int>*() const;
  }

  %extend XYZ {
    // Note use of type T
    %rename(opAnother1) operator Another() const;
    %rename(tMethod1) templateT(T i);
    %rename(tMethodNotXYZ1) templateNotXYZ(NotXYZ<T>);
    %rename(tMethodXYZ1) templateXYZ(XYZ<T>);
    %rename(opT1) operator T();
    %rename(opNotXYZ1) operator NotXYZ<T>() const;
    %rename(opXYZ1) operator XYZ<T>() const;
  }

  %extend XYZ<int> {
    %rename(opAnother2) operator Another() const;
    %rename(tMethod2) templateT(int i);
    %rename(tMethodNotXYZ2) templateNotXYZ(NotXYZ<int>);
    %rename(tMethodXYZ2) templateXYZ(XYZ<int>);
    %rename(opT2) operator int();
    %rename(opNotXYZ2) operator NotXYZ<int>() const;
    %rename(opXYZ2) operator XYZ<int>() const;
  }

  %extend XYZ<Space::Klass> {
    %rename(opAnother3) operator Another() const;
    %rename(tMethod3) templateT(Space::Klass i);
    %rename(tMethodNotXYZ3) templateNotXYZ(NotXYZ<Space::Klass>);
    %rename(tMethodXYZ3) templateXYZ(XYZ<Space::Klass>);
    %rename(opT3) operator Space::Klass();
    %rename(opNotXYZ3) operator NotXYZ<Space::Klass>() const;
    %rename(opXYZ3) operator XYZ<Space::Klass>() const;
  }

  %extend XYZ<Space::Enu> {
    %rename(opAnother4 )operator Another() const;
    %rename(tMethod4) templateT(Space::Enu i);
    %rename(tMethodNotXYZ4) templateNotXYZ(NotXYZ<Space::Enu>);
    %rename(tMethodXYZ4) templateXYZ(XYZ<Space::Enu>);
    %rename(opT4) operator Space::Enu();
    %rename(opNotXYZ4) operator NotXYZ<Space::Enu>() const;
    %rename(opXYZ4) operator XYZ<Space::Enu>() const;
  }


  %extend ABC {
    %rename(methodABC) method(ABC a) const;
    %rename(opABC) operator ABC() const;
    %rename(methodKlass) method(Klass k) const;
    %rename(opKlass) operator Klass() const;
  }
}

%{
#include "rename.h"
%}
%include "rename.h"

%template(XYZInt) Space::XYZ<int>;
%template(XYZDouble) Space::XYZ<double>;
%template(XYZKlass) Space::XYZ<Space::Klass>;
%template(XYZEnu) Space::XYZ<Space::Enu>;

%template(NotXYZInt) Space::NotXYZ<int>;
%template(NotXYZDouble) Space::NotXYZ<double>;
%template(NotXYZKlass) Space::NotXYZ<Space::Klass>;
%template(NotXYZEnu) Space::NotXYZ<Space::Enu>;


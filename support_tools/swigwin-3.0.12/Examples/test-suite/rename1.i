// Test %rename directive in global namespace by fully qualifying types
%module rename1

// Note: Space:: qualifier
%rename(opIntPtrA) Space::XYZ::operator NotXYZ<int>*() const;
%rename(opIntPtrB) Space::XYZ::operator XYZ<int>*() const;

%rename(opAnother1) Space::XYZ::operator Another() const;
%rename(opAnother2) Space::XYZ<int>::operator Another() const;
%rename(opAnother3) Space::XYZ<Space::Klass>::operator Another() const;
%rename(opAnother4) Space::XYZ<Space::Enu>::operator Another() const;

%rename(tMethod1) Space::XYZ::templateT(T i);
%rename(tMethod2) Space::XYZ<int>::templateT(int i);
%rename(tMethod3) Space::XYZ<Space::Klass>::templateT(Space::Klass i);
%rename(tMethod4) Space::XYZ<Space::Enu>::templateT(Space::Enu i);

%rename(tMethodNotXYZ1) Space::XYZ::templateNotXYZ(NotXYZ<T>);
%rename(tMethodNotXYZ2) Space::XYZ<int>::templateNotXYZ(NotXYZ<int>);
%rename(tMethodNotXYZ3) Space::XYZ<Space::Klass>::templateNotXYZ(NotXYZ<Space::Klass>);
%rename(tMethodNotXYZ4) Space::XYZ<Space::Enu>::templateNotXYZ(NotXYZ<Space::Enu>);

%rename(tMethodXYZ1) Space::XYZ::templateXYZ(XYZ<T>);
%rename(tMethodXYZ2) Space::XYZ<int>::templateXYZ(XYZ<int>);
%rename(tMethodXYZ3) Space::XYZ<Space::Klass>::templateXYZ(XYZ<Space::Klass>);
%rename(tMethodXYZ4) Space::XYZ<Space::Enu>::templateXYZ(XYZ<Space::Enu>);

%rename(opT1) Space::XYZ::operator T();
%rename(opT2) Space::XYZ<int>::operator int();
%rename(opT3) Space::XYZ<Space::Klass>::operator Space::Klass();
%rename(opT4) Space::XYZ<Space::Enu>::operator Space::Enu();

%rename(opNotXYZ1) Space::XYZ::operator NotXYZ<T>() const;
%rename(opNotXYZ2) Space::XYZ<int>::operator NotXYZ<int>() const;
%rename(opNotXYZ3) Space::XYZ<Space::Klass>::operator NotXYZ<Space::Klass>() const;
%rename(opNotXYZ4) Space::XYZ<Space::Enu>::operator NotXYZ<Space::Enu>() const;

%rename(opXYZ1) Space::XYZ::operator XYZ<T>() const;
%rename(opXYZ2) Space::XYZ<int>::operator XYZ<int>() const;
%rename(opXYZ3) Space::XYZ<Space::Klass>::operator XYZ<Space::Klass>() const;
%rename(opXYZ4) Space::XYZ<Space::Enu>::operator XYZ<Space::Enu>() const;


%rename(methodABC) Space::ABC::method(ABC a) const;
%rename(opABC) Space::ABC::operator ABC() const;
%rename(methodKlass) Space::ABC::method(Klass k) const;
%rename(opKlass) Space::ABC::operator Klass() const;

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


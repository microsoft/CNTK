// Test %rename directive in the Space namespace
%module rename2

namespace Space {
// Note: no Space:: qualifier
%rename(opIntPtrA) XYZ::operator NotXYZ<int>*() const;
%rename(opIntPtrB) XYZ::operator XYZ<int>*() const;

%rename(opAnother1) XYZ::operator Another() const;
%rename(opAnother2) XYZ<int>::operator Another() const;
%rename(opAnother3) XYZ<Space::Klass>::operator Another() const;
%rename(opAnother4) XYZ<Space::Enu>::operator Another() const;

%rename(tMethod1) XYZ::templateT(T i);
%rename(tMethod2) XYZ<int>::templateT(int i);
%rename(tMethod3) XYZ<Space::Klass>::templateT(Space::Klass i);
%rename(tMethod4) XYZ<Space::Enu>::templateT(Space::Enu i);

%rename(tMethodNotXYZ1) XYZ::templateNotXYZ(NotXYZ<T>);
%rename(tMethodNotXYZ2) XYZ<int>::templateNotXYZ(NotXYZ<int>);
%rename(tMethodNotXYZ3) XYZ<Space::Klass>::templateNotXYZ(NotXYZ<Space::Klass>);
%rename(tMethodNotXYZ4) XYZ<Space::Enu>::templateNotXYZ(NotXYZ<Space::Enu>);

%rename(tMethodXYZ1) XYZ::templateXYZ(XYZ<T>);
%rename(tMethodXYZ2) XYZ<int>::templateXYZ(XYZ<int>);
%rename(tMethodXYZ3) XYZ<Space::Klass>::templateXYZ(XYZ<Space::Klass>);
%rename(tMethodXYZ4) XYZ<Space::Enu>::templateXYZ(XYZ<Space::Enu>);

%rename(opT1) XYZ::operator T();
%rename(opT2) XYZ<int>::operator int();
%rename(opT3) XYZ<Space::Klass>::operator Space::Klass();
%rename(opT4) XYZ<Space::Enu>::operator Space::Enu();

%rename(opNotXYZ1) XYZ::operator NotXYZ<T>() const;
%rename(opNotXYZ2) XYZ<int>::operator NotXYZ<int>() const;
%rename(opNotXYZ3) XYZ<Space::Klass>::operator NotXYZ<Space::Klass>() const;
%rename(opNotXYZ4) XYZ<Space::Enu>::operator NotXYZ<Space::Enu>() const;

%rename(opXYZ1) XYZ::operator XYZ<T>() const;
%rename(opXYZ2) XYZ<int>::operator XYZ<int>() const;
%rename(opXYZ3) XYZ<Space::Klass>::operator XYZ<Space::Klass>() const;
%rename(opXYZ4) XYZ<Space::Enu>::operator XYZ<Space::Enu>() const;


%rename(methodABC) ABC::method(ABC a) const;
%rename(opABC) ABC::operator ABC() const;
%rename(methodKlass) ABC::method(Klass k) const;
%rename(opKlass) ABC::operator Klass() const;
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


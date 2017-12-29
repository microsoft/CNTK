%module inout

%include "typemaps.i"
%include "std_pair.i"

%{
  inline void AddOne3(double* a, double* b, double* c) {
    *a += 1;
    *b += 1;
    *c += 1;
  }

  inline void AddOne1(double* a) {
    *a += 1;
  } 

  inline void AddOne1p(std::pair<double, double>* p) {
    p->first += 1;
    p->second += 1;
  } 

  inline void AddOne2p(std::pair<double, double>* p,double* a) {
    *a += 1;
    p->first += 1;
    p->second += 1;
  } 

  inline void AddOne3p(double* a, std::pair<double, double>* p,double* b) {
    *a += 1;
    *b += 1;
    p->first += 1;
    p->second += 1;
  } 

  inline void AddOne1r(double& a) {
    a += 1;
  } 

%}

%template() std::pair<double, double>;

void AddOne1(double* INOUT);
void AddOne3(double* INOUT, double* INOUT, double* INOUT);
void AddOne1p(std::pair<double, double>* INOUT);
void AddOne2p(std::pair<double, double>* INOUT, double* INOUT);
void AddOne3p(double* INOUT, std::pair<double, double>* INOUT, double* INOUT);
void AddOne1r(double& INOUT);

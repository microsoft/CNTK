/* File : example.h */
#include <math.h>

class ComplexVal {
private:
  double rpart, ipart;
public:
  ComplexVal(double r = 0, double i = 0) : rpart(r), ipart(i) { }
  ComplexVal(const ComplexVal &c) : rpart(c.rpart), ipart(c.ipart) { }
  ComplexVal &operator=(const ComplexVal &c) {
    rpart = c.rpart;
    ipart = c.ipart;
    return *this;
  }
  ComplexVal operator+(const ComplexVal &c) const {
    return ComplexVal(rpart+c.rpart, ipart+c.ipart);
  }
  ComplexVal operator-(const ComplexVal &c) const {
    return ComplexVal(rpart-c.rpart, ipart-c.ipart);
  }
  ComplexVal operator*(const ComplexVal &c) const {
    return ComplexVal(rpart*c.rpart - ipart*c.ipart,
		   rpart*c.ipart + c.rpart*ipart);
  }
  ComplexVal operator-() const {
    return ComplexVal(-rpart, -ipart);
  }

  double re() const { return rpart; }
  double im() const { return ipart; }
};

ComplexVal operator*(const double &s, const ComplexVal &c) {
  return ComplexVal(s*c.re(), s*c.im());
}

/* File : example.h */
#include <math.h>

class Complex {
private:
  double rpart, ipart;
public:
  Complex(double r = 0, double i = 0) : rpart(r), ipart(i) { }
  Complex(const Complex &c) : rpart(c.rpart), ipart(c.ipart) { }
  Complex &operator=(const Complex &c) {
    rpart = c.rpart;
    ipart = c.ipart;
    return *this;
  }
  Complex operator+(const Complex &c) const {
    return Complex(rpart+c.rpart, ipart+c.ipart);
  }
  Complex operator-(const Complex &c) const {
    return Complex(rpart-c.rpart, ipart-c.ipart);
  }
  Complex operator*(const Complex &c) const {
    return Complex(rpart*c.rpart - ipart*c.ipart,
		   rpart*c.ipart + c.rpart*ipart);
  }
  Complex operator-() const {
    return Complex(-rpart, -ipart);
  }
  
  double re() const { return rpart; }
  double im() const { return ipart; }
};




  

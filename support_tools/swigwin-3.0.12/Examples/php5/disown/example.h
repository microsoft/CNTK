/* File : example.h */

#include <vector>

class Shape {
public:
  Shape() {
    nshapes++;
  }
  virtual ~Shape() {
    nshapes--;
  }
  double  x, y;   
  void    move(double dx, double dy);
  virtual double area(void) = 0;
  virtual double perimeter(void) = 0;
  static  int nshapes;
  static  int get_nshapes();
};

class Circle : public Shape {
private:
  double radius;
public:
  Circle(double r) : radius(r) { }
  ~Circle() { }
  void set_radius( double r );
  virtual double area(void);
  virtual double perimeter(void);
};

class Square : public Shape {
private:
  double width;
public:
  Square(double w) : width(w) { }
  ~Square() { }
  virtual double area(void);
  virtual double perimeter(void);
};

class ShapeContainer {
private:
  typedef std::vector<Shape*>::iterator iterator;
  std::vector<Shape*> shapes;
public:
  ShapeContainer() : shapes() {}
  ~ShapeContainer();
  void addShape( Shape *s );
};

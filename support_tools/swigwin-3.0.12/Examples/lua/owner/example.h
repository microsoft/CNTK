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
};

class Circle : public Shape {
private:
  double radius;
public:
  Circle(double r) : radius(r) { }
  virtual double area(void);
  virtual double perimeter(void);
};

class Square : public Shape {
private:
  double width;
public:
  Square(double w) : width(w) { }
  virtual double area(void);
  virtual double perimeter(void);
};


Circle* createCircle(double w); // this method creates a new object
Square* createSquare(double w); // this method creates a new object

class ShapeOwner {
private:
    std::vector<Shape*> shapes;
    ShapeOwner(const ShapeOwner&); // no copying
    ShapeOwner& operator=(const ShapeOwner&); // no copying
public:
    ShapeOwner();
    ~ShapeOwner();
    void add(Shape* ptr); // this method takes ownership of the object
    Shape* get(int idx); // this pointer is still owned by the class (assessor)
    Shape* remove(int idx); // this method returns memory which must be deleted
};

  

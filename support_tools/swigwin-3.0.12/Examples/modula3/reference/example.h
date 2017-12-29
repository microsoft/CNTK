/* File : example.h */

struct Vector {
private:
  double x,y,z;
public:
  Vector() : x(0), y(0), z(0) { }
  Vector(double x, double y, double z) : x(x), y(y), z(z) { }
  Vector operator+(const Vector &b) const;
  char *print();
};

struct VectorArray {
private:
  Vector *items;
  int     maxsize;
public:
  VectorArray(int maxsize);
  ~VectorArray();
  Vector &operator[](int);
  int size();
};

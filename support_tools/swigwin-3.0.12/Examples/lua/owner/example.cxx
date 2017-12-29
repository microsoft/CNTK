/* File : example.c */

#include "example.h"
#include <stdio.h>

#define M_PI 3.14159265358979323846

/* Move the shape to a new location */
void Shape::move(double dx, double dy) {
  x += dx;
  y += dy;
}

int Shape::nshapes = 0;

double Circle::area(void) {
  return M_PI*radius*radius;
}

double Circle::perimeter(void) {
  return 2*M_PI*radius;
}

double Square::area(void) {
  return width*width;
}

double Square::perimeter(void) {
  return 4*width;
}

Circle* createCircle(double w)
{
    return new Circle(w);
}

Square* createSquare(double w)
{
    return new Square(w);
}

ShapeOwner::ShapeOwner() {
  printf("  ShapeOwner(%p)\n", (void *)this);
}

ShapeOwner::~ShapeOwner()
{
  printf("  ~ShapeOwner(%p)\n", (void *)this);
  for(unsigned i=0;i<shapes.size();i++)
    delete shapes[i];
}

void ShapeOwner::add(Shape* ptr) // this method takes ownership of the object
{
    shapes.push_back(ptr);
}

Shape* ShapeOwner::get(int idx) // this pointer is still owned by the class (assessor)
{
    if (idx < 0 || idx >= static_cast<int>(shapes.size()))
        return NULL;
    return shapes[idx];
}

Shape* ShapeOwner::remove(int idx) // this method returns memory which must be deleted
{
    if (idx < 0 || idx >= static_cast<int>(shapes.size()))
        return NULL;
    Shape* ptr=shapes[idx];
    shapes.erase(shapes.begin()+idx);
    return ptr;
}

%module example

%contract Circle::Circle(double radius) {
require:
    radius > 0;
}

%contract Circle::area(void) {
ensure:
    area > 0;
}

%contract Shape::move(double dx, double dy) {
require:
    dx > 0;
}

/* should be no effect, since there is no move() for class Circle */
%contract Circle::move(double dx, double dy) {
require:
    dy > 1;
}

# include must be after contracts
%{
#include "example.h"
%}
%include "example.h"

/* File : equality.i */
/*
 Specific test for operator== overload. Partially overlaps with
 operator_overload.i
*/

%module equality

%warnfilter(SWIGWARN_LANG_IDENTIFIER) operator==;

%inline %{

/* Point has no equality operator */
typedef struct Point
{
  double x;
  double y;
} Point;

static const Point s_zeroPoint = { 0.0, 0.0 };
/* stack version */
Point MakePoint(double x, double y) 
  { Point new_point = {x, y}; return new_point; }

const Point* GetZeroPointPtr() { return &s_zeroPoint; }
Point GetZeroPointCopy() { return s_zeroPoint; }

/* EqualOpDefined has correct equality operator */
class EqualOpDefined {
public:
  EqualOpDefined():
    x(5) {}
  EqualOpDefined(int val):
    x(val) {}

  int x;
};

/* EqualOpWrong has logically incorrect equality operator */
class EqualOpWrong {
public:
  inline static const EqualOpWrong* GetStaticObject();
};
 
static EqualOpWrong s_wrongEqOp;

const EqualOpWrong* EqualOpWrong::GetStaticObject()
  { return &s_wrongEqOp; }


inline bool operator==( const EqualOpDefined& first, const EqualOpDefined& second )
  { return first.x == second.x; }

inline bool operator==( const EqualOpWrong& first, const EqualOpWrong& second )
  { return false; }

%}

/* 
  in order to wrapper this correctly
  we need to extend the class
  to make the friends & non members part of the class
*/
%extend EqualOpDefined {
	bool operator==(const EqualOpDefined& b){return (*$self) == b;}
}
%extend EqualOpWrong {
	bool operator==(const EqualOpWrong& b){return (*$self) == b;}
}

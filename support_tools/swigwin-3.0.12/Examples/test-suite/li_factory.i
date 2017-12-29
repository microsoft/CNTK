%module li_factory
%include factory.i

%newobject Geometry::create;

%newobject Geometry::clone;
%factory(Geometry *Geometry::create, Point, Circle);
%factory(Geometry *Geometry::clone, Point, Circle);
#ifdef SWIGPHP
%rename(clone_) clone;
#endif
%factory(Geometry *Point::clone, Point, Circle);
%factory(Geometry *Circle::clone, Point, Circle);

%inline {
  struct Geometry {
    enum GeomType{
      POINT,
      CIRCLE
    };
    
    virtual ~Geometry() {}    
    virtual int draw() = 0;
    static Geometry *create(GeomType i);
		virtual Geometry *clone() = 0;
  };

  struct Point : Geometry  {
    int draw() { return 1; }
    double width() { return 1.0; }    
		Geometry *clone() { return new Point(); }
  };

  struct Circle : Geometry  {
    int draw() { return 2; }
    double radius() { return 1.5; }      
		Geometry *clone() { return new Circle(); }
  }; 

  Geometry *Geometry::create(GeomType type) {
    switch (type) {
    case POINT: return new Point();
    case CIRCLE: return new Circle(); 
    default: return 0;
    }
  }
}



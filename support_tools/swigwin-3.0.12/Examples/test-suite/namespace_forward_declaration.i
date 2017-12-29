%module namespace_forward_declaration

%inline %{
  namespace Space1 {
    namespace Space2 {
      struct XXX;
      struct YYY;
    }

    struct Space2::YYY {
      int yyy(int h) {
	return h;
      }    
    };
    struct Space1::Space2::XXX {
      int xxx(int h) {
	return h;
      }    
    };

    void testXXX1(Space1::Space2::XXX xx) {
    }
    void testXXX2(Space2::XXX xx) {
    }
    void testXXX3(::Space1::Space2::XXX xx) {
    }
    void testYYY1(Space1::Space2::YYY yy) {
    }
    void testYYY2(Space2::YYY yy) {
    }
    void testYYY3(::Space1::Space2::YYY yy) {
    }
  }
%}


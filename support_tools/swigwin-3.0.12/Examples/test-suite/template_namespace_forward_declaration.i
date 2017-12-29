%module template_namespace_forward_declaration

%inline %{
  namespace Space1 {
    namespace Space2 {
      template<typename T> struct XXX;
      template<typename T> struct YYY;
    }

    template<typename T> struct Space2::YYY {
      T yyy(T h) {
	return h;
      }    
    };
    template<typename T> struct Space1::Space2::XXX {
      T xxx(T h) {
	return h;
      }    
    };

    void testXXX1(Space1::Space2::XXX<int> xx) {
    }
    void testXXX2(Space2::XXX<int> xx) {
    }
    void testXXX3(::Space1::Space2::XXX<int> xx) {
    }
    void testYYY1(Space1::Space2::YYY<int> yy) {
    }
    void testYYY2(Space2::YYY<int> yy) {
    }
    void testYYY3(::Space1::Space2::YYY<int> yy) {
    }
  }
%}

%template(XXXInt) Space1::Space2::XXX<int>;
%template(YYYInt) Space1::Space2::YYY<int>;


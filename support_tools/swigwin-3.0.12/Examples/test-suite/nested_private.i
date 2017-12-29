%module nested_private

// segfault due to private nested class usage

%inline %{
#include <string>
class MotorCar {

  struct DesignOpinion {
    std::string reason;
  };

public:
  struct WindScreen {
    WindScreen(bool opaque) : opaque(opaque) {}
    DesignOpinion Opinion();
  private:
    bool opaque;
  };
  
  std::string WindScreenOpinion() {
    return MotorCar::WindScreen(true).Opinion().reason;
  }
};

MotorCar::DesignOpinion MotorCar::WindScreen::Opinion() {
  DesignOpinion opinion;
  opinion.reason = !opaque ? "great design" : "you can't see out the windscreen";
  return opinion;
}

%}

#include "example.h"

int MotorCar::DesignOpinion::AceDesignCount = 0;
int MotorCar::DesignOpinion::TotalDesignCount = 0;

int MotorCar::DesignOpinion::PercentScore() {
  return AceDesignCount*100/TotalDesignCount;
}

MotorCar::Wheels::Wheels(Shape shape, size_t count) : shape(shape), count(count) {}

MotorCar::WindScreen::WindScreen(bool opaque) : opaque(opaque) {}

MotorCar::MotorCar(const std::string &name, const Wheels &wheels, const WindScreen &windscreen) : name(name), wheels(wheels), windscreen(windscreen) {}

MotorCar MotorCar::DesignFromComponents(const std::string &name, const Wheels &wheels, const WindScreen &windscreen) {
  MotorCar car = MotorCar(name, wheels, windscreen);
  DesignOpinion::TotalDesignCount++;
  if (car.wheels.Opinion().itrocks && car.windscreen.Opinion().itrocks)
    DesignOpinion::AceDesignCount++;
  return car;
}

MotorCar::DesignOpinion MotorCar::Wheels::Opinion() {
  DesignOpinion opinion;
  opinion.itrocks = true;
  if (shape == Square) {
    opinion.itrocks = false;
    opinion.reason = "you'll have a few issues with wheel rotation";
  }
  if (count <= 2) {
    opinion.reason += opinion.itrocks ? "" : " and ";
    opinion.itrocks = false;
    opinion.reason += "a few more wheels are needed for stability";
  }
  if (opinion.itrocks)
    opinion.reason = "your choice of wheels was top notch";

  return opinion;
}

MotorCar::DesignOpinion MotorCar::WindScreen::Opinion() {
  DesignOpinion opinion;
  opinion.itrocks = !opaque;
  opinion.reason = opinion.itrocks ? "the driver will have a commanding view out the window" : "you can't see out the windscreen";
  return opinion;
}

std::string MotorCar::WillItWork() {
  DesignOpinion wh = wheels.Opinion();
  DesignOpinion ws = windscreen.Opinion();
  std::string willit;
  if (wh.itrocks && ws.itrocks) {
    willit = "Great car design because " + wh.reason + " and " + ws.reason;
  } else {
    willit = "You need a rethink because ";
    willit += wh.itrocks ? "" : wh.reason;
    willit += (!wh.itrocks && !ws.itrocks) ? " and " : "";
    willit += ws.itrocks ? "" : ws.reason;
  }
  return willit;
}

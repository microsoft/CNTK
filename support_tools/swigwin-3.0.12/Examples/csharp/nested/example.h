#include <string>

/** Design a motor car from various components */
struct MotorCar {

  /** Information about an opinion of the design of a car component */
  struct DesignOpinion {
    bool itrocks;
    std::string reason;
    static int AceDesignCount;
    static int TotalDesignCount;
    static int PercentScore();
  };

  /** Wheels component */
  struct Wheels {
    enum Shape { Round, Square };
    Wheels(Shape shape, size_t count);
    DesignOpinion Opinion();
  private:
    Shape shape;
    size_t count;
  };

  /** Windscreen component */
  struct WindScreen {
    WindScreen(bool opaque);
    DesignOpinion Opinion();
  private:
    bool opaque;
  };

  /** Factory method for creating a car */
  static MotorCar DesignFromComponents(const std::string &name, const Wheels &wheels, const WindScreen &windscreen);

  std::string Name() {
    return name;
  }

  /** Get an overall opinion on the car design */
  std::string WillItWork();

private:
  MotorCar(const std::string &name, const Wheels &wheels, const WindScreen &windscreen);
  std::string name;
  Wheels wheels;
  WindScreen windscreen;
};

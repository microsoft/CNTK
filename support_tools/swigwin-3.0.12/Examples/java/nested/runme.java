// This example illustrates wrapping of nested C++ classes

public class runme {
  static {
    try {
        System.loadLibrary("example");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[])
  {
    MotorCar car1 = MotorCar.DesignFromComponents("Bumpy", new MotorCar.Wheels(MotorCar.Wheels.Shape.Square, 4), new MotorCar.WindScreen(false));
    MotorCar car2 = MotorCar.DesignFromComponents("Wobbly", new MotorCar.Wheels(MotorCar.Wheels.Shape.Round, 2), new MotorCar.WindScreen(false));
    MotorCar car3 = MotorCar.DesignFromComponents("Batty", new MotorCar.Wheels(MotorCar.Wheels.Shape.Round, 4), new MotorCar.WindScreen(true));
    MotorCar car4 = MotorCar.DesignFromComponents("Spiffing", new MotorCar.Wheels(MotorCar.Wheels.Shape.Round, 4), new MotorCar.WindScreen(false));

    System.out.println("Expert opinion on " + car1.Name() + " : \n  " + car1.WillItWork());
    System.out.println("Expert opinion on " + car2.Name() + " : \n  " + car2.WillItWork());
    System.out.println("Expert opinion on " + car3.Name() + " : \n  " + car3.WillItWork());
    System.out.println("Expert opinion on " + car4.Name() + " : \n  " + car4.WillItWork());

    int count = MotorCar.DesignOpinion.getAceDesignCount();
    int total = MotorCar.DesignOpinion.getTotalDesignCount();
    int percent = MotorCar.DesignOpinion.PercentScore();
    System.out.println("Overall opinion rating on car design is " + count + "/" + total  + " = " + percent + "%");

    System.out.println("Single square wheel thoughts: " + new MotorCar.Wheels(MotorCar.Wheels.Shape.Square, 1).Opinion().getReason());
  }
}

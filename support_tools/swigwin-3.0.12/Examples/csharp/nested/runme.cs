// This example illustrates how C++ classes can be used from C# using SWIG.
// The C# class gets mapped onto the C++ class and behaves as if it is a C# class.

using System;

public class runme
{
  static void Main()
  {
    MotorCar car1 = MotorCar.DesignFromComponents("Bumpy", new MotorCar.Wheels(MotorCar.Wheels.Shape.Square, 4), new MotorCar.WindScreen(false));
    MotorCar car2 = MotorCar.DesignFromComponents("Wobbly", new MotorCar.Wheels(MotorCar.Wheels.Shape.Round, 2), new MotorCar.WindScreen(false));
    MotorCar car3 = MotorCar.DesignFromComponents("Batty", new MotorCar.Wheels(MotorCar.Wheels.Shape.Round, 4), new MotorCar.WindScreen(true));
    MotorCar car4 = MotorCar.DesignFromComponents("Spiffing", new MotorCar.Wheels(MotorCar.Wheels.Shape.Round, 4), new MotorCar.WindScreen(false));

    Console.WriteLine("Expert opinion on " + car1.Name() + " : \n  " + car1.WillItWork());
    Console.WriteLine("Expert opinion on " + car2.Name() + " : \n  " + car2.WillItWork());
    Console.WriteLine("Expert opinion on " + car3.Name() + " : \n  " + car3.WillItWork());
    Console.WriteLine("Expert opinion on " + car4.Name() + " : \n  " + car4.WillItWork());

    int count = MotorCar.DesignOpinion.AceDesignCount;
    int total = MotorCar.DesignOpinion.TotalDesignCount;
    int percent = MotorCar.DesignOpinion.PercentScore();
    Console.WriteLine("Overall opinion rating on car design is " + count + "/" + total  + " = " + percent + "%");

    Console.WriteLine("Single square wheel thoughts: " + new MotorCar.Wheels(MotorCar.Wheels.Shape.Square, 1).Opinion().reason);
  }
}

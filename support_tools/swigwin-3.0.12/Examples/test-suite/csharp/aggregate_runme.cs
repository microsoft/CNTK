using System;
using aggregateNamespace;

public class runme {
    static void Main() {

      // Confirm that move() returns correct results under normal use
      int result = aggregate.move(aggregate.UP);
      if (result != aggregate.UP) throw new Exception("UP failed");

      result = aggregate.move(aggregate.DOWN);
      if (result != aggregate.DOWN) throw new Exception("DOWN failed");

      result = aggregate.move(aggregate.LEFT);
      if (result != aggregate.LEFT) throw new Exception("LEFT failed");

      result = aggregate.move(aggregate.RIGHT);
      if (result != aggregate.RIGHT) throw new Exception("RIGHT failed");

      // Confirm that move() raises an exception when the contract is violated
      try {
        aggregate.move(0);
        throw new Exception("0 test failed");
      }
      catch (ArgumentOutOfRangeException) {
      }
    }
}

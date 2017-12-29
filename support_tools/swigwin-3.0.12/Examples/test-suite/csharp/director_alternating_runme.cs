
using System;
using director_alternatingNamespace;

public class director_alternating_runme {
  public static void Main() {
   if (director_alternating.getBar().id() != director_alternating.idFromGetBar())
     throw new Exception("idFromGetBar failed");
  }
}



// This is the bool runtime testcase. It checks that the C++ bool type works.

using System;
using boolsNamespace;

public class bools_runme {

  public static void Main() {

		bool t = true;
		bool f = false;
		
		check_bo(f);
		check_bo(t);
  }

  public static void check_bo(bool input) {

		for( int i=0; i<1000; i++ ) {
			if( bools.bo(input) != input ) {
				string ErrorMessage = "Runtime test check_bo failed.";
				throw new Exception(ErrorMessage);
			}
		}

  }
}


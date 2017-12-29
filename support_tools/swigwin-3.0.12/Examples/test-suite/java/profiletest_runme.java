import profiletest.*;

public class profiletest_runme {

    System.loadLibrary("profiletest");

    public static void main(String argv[]) {
	
	long a = profiletest.new_A();
	long b = profiletest.new_B();
	for (int i=0; i<1000000; i++) {
	    a = profiletest.B_fn(b, a);
	}
    }
}

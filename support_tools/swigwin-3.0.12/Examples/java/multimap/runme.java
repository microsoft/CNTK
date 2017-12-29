
public class runme {

  static {
    try {
      System.loadLibrary("example");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

    // Call our gcd() function
    int x = 42;
    int y = 105;
    int g = example.gcd(x,y);
    System.out.println("The gcd of " + x + " and " + y + " is " + g);
    
    // Call the gcdmain() function
    String[] args = {"gcdmain","42","105"};
    example.gcdmain(args);
    
    // Call the count function
    System.out.println(example.count("Hello World", 'l'));
    
    // Call the capitalize function
    String[] capitalizeMe = {"hello world"};
    example.capitalize(capitalizeMe);
    System.out.println(capitalizeMe[0]);
  }
}








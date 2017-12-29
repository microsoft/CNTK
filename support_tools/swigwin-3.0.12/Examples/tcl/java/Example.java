public class Example {
    public   int      mPublicInt;
    
    public Example() {
	mPublicInt = 0;
    }
    
    public Example(int IntVal) {
	mPublicInt = IntVal;
    }
    
    
    public int Add(int a, int b) {
	return (a+b);
    }
    
    public float Add(float a, float b) {
	return (a+b);
    }
    
    public String Add(String a, String b) {
	return (a+b);
    }
    
    public Example Add(Example a, Example b) {
	return new Example(a.mPublicInt + b.mPublicInt);
    }
}


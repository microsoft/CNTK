using System;
using throw_exceptionNamespace;

public class runme
{
    static void Main() {
        Foo f = new Foo();
        try {
            f.test_int();
            throw new Exception("Integer exception should have been thrown");
        } catch (System.Exception) {
        }
        try {
            f.test_msg();
            throw new Exception("String exception should have been thrown");
        } catch (System.Exception) {
        }
        try {
            f.test_cls();
            throw new Exception("Class exception should have been thrown");
        } catch (System.Exception) {
        }
    }
}

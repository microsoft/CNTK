using System;
using default_constructorNamespace;

public class runme
{
    static void Main() 
    {
        // calling protected destructor test
        try {
            using (G g = new G()) {
            }
            throw new Exception("Protected destructor exception should have been thrown");
        } catch (MethodAccessException) {
        }
    }
}

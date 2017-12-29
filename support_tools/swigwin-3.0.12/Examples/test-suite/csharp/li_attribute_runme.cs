// Ported from Python li_attribute_runme.py

using System;
using li_attributeNamespace;

public class li_attribute_runme {

  public static void Main() {
		A aa = new A(1,2,3);

		if (aa.a != 1)
			throw new ApplicationException("error");
		aa.a = 3;
		if (aa.a != 3)
			throw new ApplicationException("error");

		if (aa.b != 2)
			throw new ApplicationException("error");
		aa.b = 5;
		if (aa.b != 5)
			throw new ApplicationException("error");

		if (aa.d != aa.b)
			throw new ApplicationException("error");

		if (aa.c != 3)
			throw new ApplicationException("error");
		//aa.c = 5;
		//if (aa.c != 3)
		//  throw new ApplicationException("error");

		Param_i pi = new Param_i(7);
		if (pi.value != 7)
			throw new ApplicationException("error");

		pi.value=3;
		if (pi.value != 3)
			throw new ApplicationException("error");

		B b = new B(aa);

		if (b.a.c != 3)
			throw new ApplicationException("error");

		// class/struct attribute with get/set methods using return/pass by reference
		MyFoo myFoo = new MyFoo();
		myFoo.x = 8;
		MyClass myClass = new MyClass();
		myClass.Foo = myFoo;
		if (myClass.Foo.x != 8)
			throw new ApplicationException("error");

		// class/struct attribute with get/set methods using return/pass by value
		MyClassVal myClassVal = new MyClassVal();
		if (myClassVal.ReadWriteFoo.x != -1)
			throw new ApplicationException("error");
		if (myClassVal.ReadOnlyFoo.x != -1)
			throw new ApplicationException("error");
		myClassVal.ReadWriteFoo = myFoo;
		if (myClassVal.ReadWriteFoo.x != 8)
			throw new ApplicationException("error");
		if (myClassVal.ReadOnlyFoo.x != 8)
			throw new ApplicationException("error");

    // string attribute with get/set methods using return/pass by value
		MyStringyClass myStringClass = new MyStringyClass("initial string");
		if (myStringClass.ReadWriteString != "initial string")
			throw new ApplicationException("error");
		if (myStringClass.ReadOnlyString != "initial string")
			throw new ApplicationException("error");
		myStringClass.ReadWriteString = "changed string";
		if (myStringClass.ReadWriteString != "changed string")
			throw new ApplicationException("error");
		if (myStringClass.ReadOnlyString != "changed string")
			throw new ApplicationException("error");
  }
}


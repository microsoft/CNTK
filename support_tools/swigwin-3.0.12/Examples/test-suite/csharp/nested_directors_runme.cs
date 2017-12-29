using System;
using nested_directorsNamespace;
#pragma warning disable 219

public class CNested : Base.Nest {
	public override bool GetValue() {return true;}
}
public class CSub : Sub {
	protected override bool GetValue() { return base.GetValue(); }
	public bool Test(){ return GetValue(); }
}

public class runme {
  static void Main() {
  	CNested n = new CNested();
		CSub s = new CSub();
		if (!s.Test())
			throw new Exception("Sub.GetValue");
  }
}

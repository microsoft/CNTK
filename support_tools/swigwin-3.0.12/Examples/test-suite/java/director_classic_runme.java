import director_classic.*;

public class director_classic_runme {
  static {
    try {
        System.loadLibrary("director_classic");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    { 
      Person person = new Person();
      check(person, "Person");
      person.delete();
    }
    { 
      Person person = new Child();
      check(person, "Child");
      person.delete();
    }
    { 
      Person person = new GrandChild();
      check(person, "GrandChild"); 
      person.delete();
    }
    { 
      Person person = new TargetLangPerson();
      check(person, "TargetLangPerson"); 
      person.delete();
    }
    { 
      Person person = new TargetLangChild();
      check(person, "TargetLangChild"); 
      person.delete();
    }
    { 
      Person person = new TargetLangGrandChild();
      check(person, "TargetLangGrandChild"); 
      person.delete();
    }

    // Semis - don't override id() in target language
    { 
      Person person = new TargetLangSemiPerson();
      check(person, "Person"); 
      person.delete();
    }
    { 
      Person person = new TargetLangSemiChild();
      check(person, "Child"); 
      person.delete();
    }
    { 
      Person person = new TargetLangSemiGrandChild();
      check(person, "GrandChild"); 
      person.delete();
    }

    // Orphans - don't override id() in C++
    { 
      Person person = new OrphanPerson();
      check(person, "Person"); 
      person.delete();
    }
    { 
      Person person = new OrphanChild();
      check(person, "Child"); 
      person.delete();
    }
    { 
      Person person = new TargetLangOrphanPerson();
      check(person, "TargetLangOrphanPerson"); 
      person.delete();
    }
    { 
      Person person = new TargetLangOrphanChild();
      check(person, "TargetLangOrphanChild"); 
      person.delete();
    }

    // Duals - id() makes an upcall to the base id()
    { 
      Person person = new TargetLangDualPerson();
      check(person, "TargetLangDualPerson + Person"); 
      person.delete();
    }
    { 
      Person person = new TargetLangDualChild();
      check(person, "TargetLangDualChild + Child"); 
      person.delete();
    }
    { 
      Person person = new TargetLangDualGrandChild();
      check(person, "TargetLangDualGrandChild + GrandChild"); 
      person.delete();
    }

    // Mix Orphans and Duals
    { 
      Person person = new TargetLangDualOrphanPerson();
      check(person, "TargetLangDualOrphanPerson + Person"); 
      person.delete();
    }
    { 
      Person person = new TargetLangDualOrphanChild();
      check(person, "TargetLangDualOrphanChild + Child"); 
      person.delete();
    }
  }

  static void check(Person person, String expected) {
    String ret;
    // Normal target language polymorphic call
    ret = person.id();
    if (debug) 
      System.out.println(ret);
    if (!ret.equals(expected))
      throw new RuntimeException("Failed. Received: " + ret + " Expected: " + expected);

    // Polymorphic call from C++
    Caller caller = new Caller();
    caller.setCallback(person);
    ret = caller.call();
    if (debug) 
      System.out.println(ret);
    if (!ret.equals(expected))
      throw new RuntimeException("Failed. Received: " + ret + " Expected: " + expected);

    // Polymorphic call of object created in target language and passed to C++ and back again
    Person baseclass = caller.baseClass();
    ret = baseclass.id();
    if (debug) 
      System.out.println(ret);
    if (!ret.equals(expected))
      throw new RuntimeException("Failed. Received: " + ret + " Expected: " + expected);

    caller.resetCallback();
    if (debug) 
      System.out.println("----------------------------------------");
  }
  static boolean debug = false;
}

class TargetLangPerson extends Person
{
  public TargetLangPerson()
  {
    super();
  }

  public String id()
  {
    String identifier = "TargetLangPerson";
    return identifier;
  }
}

class TargetLangChild extends Child
{
  public TargetLangChild()
  {
    super();
  }

  public String id()
  {
    String identifier = "TargetLangChild";
    return identifier;
  }
}

class TargetLangGrandChild extends GrandChild
{
  public TargetLangGrandChild()
  {
    super();
  }

  public String id()
  {
    String identifier = "TargetLangGrandChild";
    return identifier;
  }
}

// Semis - don't override id() in target language
class TargetLangSemiPerson extends Person
{
  public TargetLangSemiPerson()
  {
    super();
  }
  // No id() override
}

class TargetLangSemiChild extends Child
{
  public TargetLangSemiChild()
  {
    super();
  }
  // No id() override
}

class TargetLangSemiGrandChild extends GrandChild
{
  public TargetLangSemiGrandChild()
  {
    super();
  }
  // No id() override
}

// Orphans - don't override id() in C++
class TargetLangOrphanPerson extends OrphanPerson
{
  public TargetLangOrphanPerson()
  {
    super();
  }

  public String id()
  {
    String identifier = "TargetLangOrphanPerson";
    return identifier;
  }
}

class TargetLangOrphanChild extends OrphanChild
{
  public TargetLangOrphanChild()
  {
    super();
  }

  public String id()
  {
    String identifier = "TargetLangOrphanChild";
    return identifier;
  }
}

// Duals - id() makes an upcall to the base id()
class TargetLangDualPerson extends Person
{
  public TargetLangDualPerson()
  {
    super();
  }

  public String id()
  {
    String identifier = "TargetLangDualPerson + " + super.id();
    return identifier;
  }
}

class TargetLangDualChild extends Child
{
  public TargetLangDualChild()
  {
    super();
  }

  public String id()
  {
    String identifier = "TargetLangDualChild + " + super.id();
    return identifier;
  }
}

class TargetLangDualGrandChild extends GrandChild
{
  public TargetLangDualGrandChild()
  {
    super();
  }

  public String id()
  {
    String identifier = "TargetLangDualGrandChild + " + super.id();
    return identifier;
  }
}

// Mix Orphans and Duals
class TargetLangDualOrphanPerson extends OrphanPerson
{
  public TargetLangDualOrphanPerson()
  {
    super();
  }

  public String id()
  {
    String identifier = "TargetLangDualOrphanPerson + " + super.id();
    return identifier;
  }
}

class TargetLangDualOrphanChild extends OrphanChild
{
  public TargetLangDualOrphanChild()
  {
    super();
  }

  public String id()
  {
    String identifier = "TargetLangDualOrphanChild + " + super.id();
    return identifier;
  }
}


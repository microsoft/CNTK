# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

director_classic

TargetLangPerson=@() subclass(Person(),'id',@(self) "TargetLangPerson");
TargetLangChild=@() subclass(Child(),'id',@(self) "TargetLangChild");
TargetLangGrandChild=@() subclass(GrandChild(),'id',@(self) "TargetLangGrandChild");

# Semis - don't override id() in target language
TargetLangSemiPerson=@() subclass(Person());
TargetLangSemiChild=@() subclass(Child());
TargetLangSemiGrandChild=@() subclass(GrandChild());

# Orphans - don't override id() in C++
TargetLangOrphanPerson=@() subclass(OrphanPerson(),'id',@(self) "TargetLangOrphanPerson");
TargetLangOrphanChild=@() subclass(OrphanChild(),'id',@(self) "TargetLangOrphanChild");


function check(person,expected)
  global Caller;

  # Normal target language polymorphic call
  ret = person.id();
  if (ret != expected)
    raise ("Failed. Received: " + ret + " Expected: " + expected);
  endif

  # Polymorphic call from C++
  caller = Caller();
  caller.setCallback(person);
  ret = caller.call();
  if (ret != expected)
    error ("Failed. Received: " + ret + " Expected: " + expected);
  endif

  # Polymorphic call of object created in target language and passed to C++ and back again
  baseclass = caller.baseClass();
  ret = baseclass.id();
  if (ret != expected)
    error ("Failed. Received: " + ret + " Expected: " + expected);
  endif

  caller.resetCallback();
end


person = Person();
check(person, "Person");
clear person;

person = Child();
check(person, "Child");
clear person;

person = GrandChild();
check(person, "GrandChild"); 
clear person;

person = TargetLangPerson();
check(person, "TargetLangPerson"); 
clear person;

person = TargetLangChild();
check(person, "TargetLangChild"); 
clear person;

person = TargetLangGrandChild();
check(person, "TargetLangGrandChild"); 
clear person;

# Semis - don't override id() in target language
person = TargetLangSemiPerson();
check(person, "Person"); 
clear person;

person = TargetLangSemiChild();
check(person, "Child"); 
clear person;

person = TargetLangSemiGrandChild();
check(person, "GrandChild"); 
clear person;

# Orphans - don't override id() in C++
person = OrphanPerson();
check(person, "Person");
clear person;

person = OrphanChild();
check(person, "Child");
clear person;

person = TargetLangOrphanPerson();
check(person, "TargetLangOrphanPerson"); 
clear person;

person = TargetLangOrphanChild();
check(person, "TargetLangOrphanChild"); 
clear person;


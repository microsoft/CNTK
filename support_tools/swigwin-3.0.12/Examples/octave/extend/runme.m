# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

# This file illustrates the cross language polymorphism using directors.

swigexample


# CEO class, which overrides Employee::getPosition().

CEO=@(name) subclass(swigexample.Manager(name),'getPosition',@(self) "CEO");

# Create an instance of our employee extension class, CEO. The calls to
# getName() and getPosition() are standard, the call to getTitle() uses
# the director wrappers to call CEO.getPosition. e = CEO("Alice")

e = CEO("Alice");
printf("%s is a %s\n",e.getName(),e.getPosition());
printf("Just call her \"%s\"\n",e.getTitle());
printf("----------------------\n");


# Create a new EmployeeList instance.  This class does not have a C++
# director wrapper, but can be used freely with other classes that do.

list = swigexample.EmployeeList();

# EmployeeList owns its items, so we must surrender ownership of objects
# we add. This involves first calling the __disown__ method to tell the
# C++ director to start reference counting. We reassign the resulting
# weakref.proxy to e so that no hard references remain. This can also be
# done when the object is constructed, as in: e =
# CEO("Alice").__disown()

e = e.__disown();
list.addEmployee(e);
printf("----------------------\n");

# Now we access the first four items in list (three are C++ objects that
# EmployeeList's constructor adds, the last is our CEO). The virtual
# methods of all these instances are treated the same. For items 0, 1, and
# 2, both all methods resolve in C++. For item 3, our CEO, getTitle calls
# getPosition which resolves in Octave. The call to getPosition is
# slightly different, however, from the e.getPosition() call above, since
# now the object reference has been "laundered" by passing through
# EmployeeList as an Employee*. Previously, Octave resolved the call
# immediately in CEO, but now Octave thinks the object is an instance of
# class Employee (actually EmployeePtr). So the call passes through the
# Employee proxy class and on to the C wrappers and C++ director,
# eventually ending up back at the CEO implementation of getPosition().
# The call to getTitle() for item 3 runs the C++ Employee::getTitle()
# method, which in turn calls getPosition(). This virtual method call
# passes down through the C++ director class to the Octave implementation
# in CEO. All this routing takes place transparently.

printf("(position, title) for items 0-3:\n");
for i=0:3,
  printf("  %s, \"%s\"\n",list.get_item(i).getPosition(), list.get_item(i).getTitle());
endfor
printf("----------------------\n");

# Time to delete the EmployeeList, which will delete all the Employee*
# items it contains. The last item is our CEO, which gets destroyed as its
# reference count goes to zero. The Octave destructor runs, and is still
# able to call self.getName() since the underlying C++ object still
# exists. After this destructor runs the remaining C++ destructors run as
# usual to destroy the object.

clear list;
printf("----------------------\n");

# All done.

printf("octave exit\n");

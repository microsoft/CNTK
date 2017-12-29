# file: runme.pl

# This file illustrates the cross language polymorphism using directors.

use example;


# CEO class, which overrides Employee::getPosition().

{
  package CEO;
  use base 'example::Manager';
  sub getPosition {
    return "CEO";
  }
}


# Create an instance of our employee extension class, CEO. The calls to
# getName() and getPosition() are standard, the call to getTitle() uses
# the director wrappers to call CEO->getPosition. $e = CEO->new("Alice")

$e = CEO->new("Alice");
print $e->getName(), " is a ", $e->getPosition(), "\n";
printf "Just call her \"%s\"\n", $e->getTitle();
print "----------------------\n";


# Create a new EmployeeList instance.  This class does not have a C++
# director wrapper, but can be used freely with other classes that do.

$list = example::EmployeeList->new();

# EmployeeList owns its items, so we must surrender ownership of objects
# we add. This involves calling the DISOWN method to tell the
# C++ director to start reference counting.

$e->DISOWN();
$list->addEmployee($e);
print "----------------------\n";

# Now we access the first four items in list (three are C++ objects that
# EmployeeList's constructor adds, the last is our CEO). The virtual
# methods of all these instances are treated the same. For items 0, 1, and
# 2, both all methods resolve in C++. For item 3, our CEO, getTitle calls
# getPosition which resolves in Perl. The call to getPosition is
# slightly different, however, from the $e->getPosition() call above, since
# now the object reference has been "laundered" by passing through
# EmployeeList as an Employee*. Previously, Perl resolved the call
# immediately in CEO, but now Perl thinks the object is an instance of
# class Employee (actually EmployeePtr). So the call passes through the
# Employee proxy class and on to the C wrappers and C++ director,
# eventually ending up back at the CEO implementation of getPosition().
# The call to getTitle() for item 3 runs the C++ Employee::getTitle()
# method, which in turn calls getPosition(). This virtual method call
# passes down through the C++ director class to the Perl implementation
# in CEO. All this routing takes place transparently.

print "(position, title) for items 0-3:\n";

printf "  %s, \"%s\"\n", $list->get_item(0)->getPosition(), $list->get_item(0)->getTitle();
printf "  %s, \"%s\"\n", $list->get_item(1)->getPosition(), $list->get_item(1)->getTitle();
printf "  %s, \"%s\"\n", $list->get_item(2)->getPosition(), $list->get_item(2)->getTitle();
printf "  %s, \"%s\"\n", $list->get_item(3)->getPosition(), $list->get_item(3)->getTitle();
print "----------------------\n";

# Time to delete the EmployeeList, which will delete all the Employee*
# items it contains. The last item is our CEO, which gets destroyed as its
# reference count goes to zero. The Perl destructor runs, and is still
# able to call self.getName() since the underlying C++ object still
# exists. After this destructor runs the remaining C++ destructors run as
# usual to destroy the object.

undef $list;
print "----------------------\n";

# All done.

print "perl exit\n";

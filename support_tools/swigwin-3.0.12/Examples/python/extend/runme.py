# file: runme.py

# This file illustrates the cross language polymorphism using directors.

import example


# CEO class, which overrides Employee::getPosition().

class CEO(example.Manager):

    def __init__(self, name):
        example.Manager.__init__(self, name)

    def getPosition(self):
        return "CEO"


# Create an instance of our employee extension class, CEO. The calls to
# getName() and getPosition() are standard, the call to getTitle() uses
# the director wrappers to call CEO.getPosition. e = CEO("Alice")

e = CEO("Alice")
print e.getName(), "is a", e.getPosition()
print "Just call her \"%s\"" % e.getTitle()
print "----------------------"


# Create a new EmployeeList instance.  This class does not have a C++
# director wrapper, but can be used freely with other classes that do.

list = example.EmployeeList()

# EmployeeList owns its items, so we must surrender ownership of objects
# we add. This involves first calling the __disown__ method to tell the
# C++ director to start reference counting. We reassign the resulting
# weakref.proxy to e so that no hard references remain. This can also be
# done when the object is constructed, as in: e =
# CEO("Alice").__disown__()

e = e.__disown__()
list.addEmployee(e)
print "----------------------"

# Now we access the first four items in list (three are C++ objects that
# EmployeeList's constructor adds, the last is our CEO). The virtual
# methods of all these instances are treated the same. For items 0, 1, and
# 2, both all methods resolve in C++. For item 3, our CEO, getTitle calls
# getPosition which resolves in Python. The call to getPosition is
# slightly different, however, from the e.getPosition() call above, since
# now the object reference has been "laundered" by passing through
# EmployeeList as an Employee*. Previously, Python resolved the call
# immediately in CEO, but now Python thinks the object is an instance of
# class Employee (actually EmployeePtr). So the call passes through the
# Employee proxy class and on to the C wrappers and C++ director,
# eventually ending up back at the CEO implementation of getPosition().
# The call to getTitle() for item 3 runs the C++ Employee::getTitle()
# method, which in turn calls getPosition(). This virtual method call
# passes down through the C++ director class to the Python implementation
# in CEO. All this routing takes place transparently.

print "(position, title) for items 0-3:"

print "  %s, \"%s\"" % (list.get_item(0).getPosition(), list.get_item(0).getTitle())
print "  %s, \"%s\"" % (list.get_item(1).getPosition(), list.get_item(1).getTitle())
print "  %s, \"%s\"" % (list.get_item(2).getPosition(), list.get_item(2).getTitle())
print "  %s, \"%s\"" % (list.get_item(3).getPosition(), list.get_item(3).getTitle())
print "----------------------"

# Time to delete the EmployeeList, which will delete all the Employee*
# items it contains. The last item is our CEO, which gets destroyed as its
# reference count goes to zero. The Python destructor runs, and is still
# able to call self.getName() since the underlying C++ object still
# exists. After this destructor runs the remaining C++ destructors run as
# usual to destroy the object.

del list
print "----------------------"

# All done.

print "python exit"

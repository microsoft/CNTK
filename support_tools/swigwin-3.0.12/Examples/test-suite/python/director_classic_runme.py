from director_classic import *


class TargetLangPerson(Person):

    def __init__(self):
        Person.__init__(self)

    def id(self):
        identifier = "TargetLangPerson"
        return identifier


class TargetLangChild(Child):

    def __init__(self):
        Child.__init__(self)

    def id(self):
        identifier = "TargetLangChild"
        return identifier


class TargetLangGrandChild(GrandChild):

    def __init__(self):
        GrandChild.__init__(self)

    def id(self):
        identifier = "TargetLangGrandChild"
        return identifier

# Semis - don't override id() in target language


class TargetLangSemiPerson(Person):

    def __init__(self):
        Person.__init__(self)
        # No id() override


class TargetLangSemiChild(Child):

    def __init__(self):
        Child.__init__(self)
        # No id() override


class TargetLangSemiGrandChild(GrandChild):

    def __init__(self):
        GrandChild.__init__(self)
        # No id() override

# Orphans - don't override id() in C++


class TargetLangOrphanPerson(OrphanPerson):

    def __init__(self):
        OrphanPerson.__init__(self)

    def id(self):
        identifier = "TargetLangOrphanPerson"
        return identifier


class TargetLangOrphanChild(OrphanChild):

    def __init__(self):
        OrphanChild.__init__(self)

    def id(self):
        identifier = "TargetLangOrphanChild"
        return identifier


def check(person, expected):

    debug = 0
    # Normal target language polymorphic call
    ret = person.id()
    if (debug):
        print(ret)
    if (ret != expected):
        raise RuntimeError(
            "Failed. Received: " + str(ret) + " Expected: " + expected)

    # Polymorphic call from C++
    caller = Caller()
    caller.setCallback(person)
    ret = caller.call()
    if (debug):
        print(ret)
    if (ret != expected):
        raise RuntimeError(
            "Failed. Received: " + str(ret) + " Expected: " + expected)

    # Polymorphic call of object created in target language and passed to C++
    # and back again
    baseclass = caller.baseClass()
    ret = baseclass.id()
    if (debug):
        print(ret)
    if (ret != expected):
        raise RuntimeError(
            "Failed. Received: " + str(ret) + " Expected: " + expected)

    caller.resetCallback()
    if (debug):
        print("----------------------------------------")


person = Person()
check(person, "Person")
del person

person = Child()
check(person, "Child")
del person

person = GrandChild()
check(person, "GrandChild")
del person

person = TargetLangPerson()
check(person, "TargetLangPerson")
del person

person = TargetLangChild()
check(person, "TargetLangChild")
del person

person = TargetLangGrandChild()
check(person, "TargetLangGrandChild")
del person

# Semis - don't override id() in target language
person = TargetLangSemiPerson()
check(person, "Person")
del person

person = TargetLangSemiChild()
check(person, "Child")
del person

person = TargetLangSemiGrandChild()
check(person, "GrandChild")
del person

# Orphans - don't override id() in C++
person = OrphanPerson()
check(person, "Person")
del person

person = OrphanChild()
check(person, "Child")
del person

person = TargetLangOrphanPerson()
check(person, "TargetLangOrphanPerson")
del person

person = TargetLangOrphanChild()
check(person, "TargetLangOrphanChild")
del person

# Check usage of template attributes

import li_attribute_template

chell = li_attribute_template.Cintint(1, 2, 3)


def rassert(what, master):
    if what != master:
        print what
        raise RuntimeError

# Testing primitive by value attribute
rassert(chell.a, 1)

chell.a = 3
rassert(chell.a, 3)

# Testing primitive by ref attribute

rassert(chell.b, 2)

chell.b = 5
rassert(chell.b, 5)

# Testing string
chell.str = "abc"
rassert(chell.str, "abc")

# Testing class by value

rassert(chell.d.value, 1)

chell.d = li_attribute_template.Foo(2)
rassert(chell.d.value, 2)

# Testing class by reference

rassert(chell.e.value, 2)

chell.e = li_attribute_template.Foo(3)
rassert(chell.e.value, 3)

chell.e.value = 4
rassert(chell.e.value, 4)

# Testing moderately complex template by value
rassert(chell.f.first, 1)
rassert(chell.f.second, 2)

pair = li_attribute_template.pair_intint(3, 4)
chell.f = pair
rassert(chell.f.first, 3)
rassert(chell.f.second, 4)

# Testing moderately complex template by ref
rassert(chell.g.first, 2)
rassert(chell.g.second, 3)

pair = li_attribute_template.pair_intint(4, 5)
chell.g = pair
rassert(chell.g.first, 4)
rassert(chell.g.second, 5)

chell.g.first = 6
chell.g.second = 7
rassert(chell.g.first, 6)
rassert(chell.g.second, 7)

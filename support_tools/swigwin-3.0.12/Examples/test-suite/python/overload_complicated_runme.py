from overload_complicated import *

pInt = None

# Check the correct constructors are available
p = Pop(pInt)

p = Pop(pInt, 0)

# Check overloaded in const only and pointers/references which target
# languages cannot disambiguate
if p.hip(0) != 701:
    raise RuntimeError, "Test 1 failed"

if p.hip(pInt) != 702:
    raise RuntimeError, "Test 2 failed"

# Reverse the order for the above
if p.hop(pInt) != 805:
    raise RuntimeError, "Test 3 failed"

if p.hop(0) != 801:
    raise RuntimeError, "Test 4 failed"

# Few more variations and order shuffled
if p.pop(0) != 901:
    raise RuntimeError, "Test 5 failed"

if p.pop(pInt) != 902:
    raise RuntimeError, "Test 6 failed"

if p.pop() != 905:
    raise RuntimeError, "Test 7 failed"

# Overload on const only
if p.bop(pInt) != 1001:
    raise RuntimeError, "Test 8 failed"

if p.bip(pInt) != 2001:
    raise RuntimeError, "Test 9 failed"

# Globals
if muzak(0) != 3001:
    raise RuntimeError, "Test 10 failed"

if muzak(pInt) != 3002:
    raise RuntimeError, "Test 11 failed"

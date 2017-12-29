from disown import *

a = A()

tmp = a.thisown

a.thisown = 0
if a.thisown:
    raise RuntimeError

a.thisown = 1
if (not a.thisown):
    raise RuntimeError

a.thisown = tmp
if (a.thisown != tmp):
    raise RuntimeError


b = B()

b.acquire(a)

if a.thisown:
    raise RuntimeError

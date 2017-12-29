from director_protected import *


class FooBar(Bar):

    def ping(self):
        return "FooBar::ping();"


class FooBar2(Bar):

    def ping(self):
        return "FooBar2::ping();"

    def pang(self):
        return "FooBar2::pang();"


class FooBar3(Bar):

    def cheer(self):
        return "FooBar3::cheer();"


b = Bar()
f = b.create()
fb = FooBar()
fb2 = FooBar2()
fb3 = FooBar3()


try:
    s = fb.used()
    if s != "Foo::pang();Bar::pong();Foo::pong();FooBar::ping();":
        raise RuntimeError
    pass
except:
    raise RuntimeError, "bad FooBar::used"

try:
    s = fb2.used()
    if s != "FooBar2::pang();Bar::pong();Foo::pong();FooBar2::ping();":
        raise RuntimeError
    pass
except:
    raise RuntimeError, "bad FooBar2::used"

try:
    s = b.pong()
    if s != "Bar::pong();Foo::pong();Bar::ping();":
        raise RuntimeError
    pass
except:
    raise RuntimeError, "bad Bar::pong"

try:
    s = f.pong()
    if s != "Bar::pong();Foo::pong();Bar::ping();":
        raise RuntimeError
    pass
except:
    raise RuntimeError, " bad Foo::pong"

try:
    s = fb.pong()
    if s != "Bar::pong();Foo::pong();FooBar::ping();":
        raise RuntimeError
    pass
except:
    raise RuntimeError, " bad FooBar::pong"

protected = 1
try:
    b.ping()
    protected = 0
except:
    pass
if not protected:
    raise RuntimeError, "Foo::ping is protected"

protected = 1
try:
    f.ping()
    protected = 0
except:
    pass
if not protected:
    raise RuntimeError, "Foo::ping is protected"


protected = 1
try:
    f.pang()
    protected = 0
except:
    pass
if not protected:
    raise RuntimeError, "FooBar::pang is protected"


protected = 1
try:
    b.cheer()
    protected = 0
except:
    pass
if not protected:
    raise RuntimeError, "Bar::cheer is protected"

protected = 1
try:
    f.cheer()
    protected = 0
except:
    pass
if not protected:
    raise RuntimeError, "Foo::cheer is protected"

if fb3.cheer() != "FooBar3::cheer();":
    raise RuntimeError, "bad fb3::cheer"

if fb2.callping() != "FooBar2::ping();":
    raise RuntimeError, "bad fb2.callping"

if fb2.callcheer() != "FooBar2::pang();Bar::pong();Foo::pong();FooBar2::ping();":
    raise RuntimeError, "bad fb2.callcheer"

if fb3.callping() != "Bar::ping();":
    raise RuntimeError, "bad fb3.callping"

if fb3.callcheer() != "FooBar3::cheer();":
    raise RuntimeError, "bad fb3.callcheer"

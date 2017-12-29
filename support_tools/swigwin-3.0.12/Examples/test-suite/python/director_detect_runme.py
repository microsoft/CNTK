import director_detect


class MyBar(director_detect.Bar):

    def __init__(self, val=2):
        director_detect.Bar.__init__(self)
        self.val = val

    def get_value(self):
        self.val = self.val + 1
        return self.val

    def get_class(self):
        self.val = self.val + 1
        return director_detect.A()

    def just_do_it(self):
        self.val = self.val + 1

    def clone(self):
        return MyBar(self.val)
    pass


b = MyBar()

f = b.baseclass()

v = f.get_value()
a = f.get_class()
f.just_do_it()

c = b.clone()
vc = c.get_value()

if (v != 3) or (b.val != 5) or (vc != 6):
    raise RuntimeError, "Bad virtual detection"

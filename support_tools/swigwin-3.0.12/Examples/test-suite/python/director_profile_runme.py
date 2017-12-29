import director_profile


class MyB(director_profile.B):

    def vfi(self, a):
        return a + 3


a = director_profile.A()
myb = MyB()
b = director_profile.B.get_self(myb)


fi = b.fi
i = 50000
a = 1
while i:
    a = fi(a)  # 1
    a = fi(a)  # 2
    a = fi(a)  # 3
    a = fi(a)  # 4
    a = fi(a)  # 5
    a = fi(a)  # 6
    a = fi(a)  # 7
    a = fi(a)  # 8
    a = fi(a)  # 9
    a = fi(a)  # 10
    a = fi(a)  # 1
    a = fi(a)  # 2
    a = fi(a)  # 3
    a = fi(a)  # 4
    a = fi(a)  # 5
    a = fi(a)  # 6
    a = fi(a)  # 7
    a = fi(a)  # 8
    a = fi(a)  # 9
    a = fi(a)  # 20
    i -= 1

print a

from director_wstring import *


class B(A):

    def __init__(self, string):
        A.__init__(self, string)

    def get_first(self):
        return A.get_first(self) + u" world!"

    def process_text(self, string):
        self.smem = u"hello"


b = B(u"hello")

b.get(0)
if b.get_first() != u"hello world!":
    print b.get_first()
    raise RuntimeError


b.call_process_func()

if b.smem != u"hello":
    print smem
    raise RuntimeError

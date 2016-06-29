
import swig_cntk

class PyCallback(swig_cntk.Callback):

    def __init__(self):
        swig_cntk.Callback.__init__(self)

    def forward(self):
        print("PyCallback.forward()")

    def backward(self):
        print("PyCallback.backward()")



def callback_test():
    
    op = swig_cntk.Caller()
    callback = swig_cntk.Callback()
    callback.thisown = 0
    op.setCallback(callback)
    op.forward()
    op.backward()
    op.delCallback()

if __name__=='__main__':
    callback_test()

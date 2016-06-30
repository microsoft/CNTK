
import swig_cntk

class PyCallback(swig_cntk.Callback):

    def __init__(self):
        swig_cntk.Callback.__init__(self)

    def forward(self):
        print("PyCallback.forward()")

    def backward(self):
        print("PyCallback.backward()")



def callback_test():
    
    op = swig_cntk.FunctionInCNTK()

    cpp_callback = swig_cntk.Callback()
    cpp_callback.thisown = 0

    # C++ callback
    op.setCallback(cpp_callback)
    op.forward()
    op.backward()
    op.delCallback()

    # Python callback
    py_callback = PyCallback()
    op.setCallback(py_callback.__disown__())
    op.forward()
    op.backward()
    op.delCallback()


if __name__=='__main__':
    callback_test()

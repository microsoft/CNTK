
import cntk.cntk_py as cntk_py

class PyCallback(cntk_py.Callback):

    def __init__(self):
        cntk_py.Callback.__init__(self)

    def forward(self):
        print("PyCallback.forward()")
        1/0

    def backward(self):
        print("PyCallback.backward()")



def callback_test():

    op = cntk_py.FunctionInCNTK()

    cpp_callback = cntk_py.Callback()
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
    import time    
    callback_test()
    #cntk_py.exception_tester()

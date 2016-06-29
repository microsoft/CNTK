# distutils: language = c++

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
cimport numpy as cnp
import numpy as np

from libc.stddef cimport wchar_t

cdef extern from "Python.h":
    wchar_t* PyUnicode_AsWideCharString(object, Py_ssize_t *)


cdef extern from r"CNTKLibrary.h" namespace "CNTK":

    cdef extern from "<string>" namespace "std" nogil:
        cdef cppclass wstring:
            wstring() except +
            wstring(wchar_t *) except +
            wstring(wchar_t *, size_t) except +
            wstring(wstring&) except +

    cdef enum _DataType "CNTK::DataType": Unknown, Float, Double

    cdef cppclass _DeviceDescriptor "CNTK::DeviceDescriptor":
        _DeviceDescriptor()
        @staticmethod
        _DeviceDescriptor CPUDevice()
        @staticmethod
        _DeviceDescriptor GPUDevice(unsigned int deviceId)
        @staticmethod
        _DeviceDescriptor DefaultDevice()

    cdef cppclass _NDShape "CNTK::NDShape":
        _NDShape(size_t numAxes, size_t dimension) except +
        _NDShape(const vector[size_t]& dimensions) except +
        size_t TotalSize() except +
        size_t NumAxes() const
        size_t& operator[](size_t axisId) except +

    ctypedef fused float_or_double:
        float
        double

    cdef cppclass _NDArrayView "CNTK::NDArrayView":
        _NDArrayView(_DataType dataType, const _NDShape& viewShape, void* dataBuffer, size_t bufferSizeInBytes, const _DeviceDescriptor& device, bool readOnly = false) except +
        _NDArrayView(_DataType dataType, const _NDShape& viewShape, const _DeviceDescriptor& device) except +
        _NDArrayView(const double& value, const _NDShape& viewShape, const _DeviceDescriptor& device, bool readOnly) except +
        _NDShape Shape() except +
        const double* DataBufferD() const
        #T* DataBuffer[T]()

    cdef cppclass _NDArrayViewPtr "CNTK::NDArrayViewPtr":
        _NDArrayViewPtr(_NDArrayView*) except +
        _NDArrayView& operator*()

    cdef cppclass _Value "CNTK::Value":
        _Value(const _NDArrayViewPtr& data) except +
        #Value(const NDArrayViewPtr& data, const NDMaskPtr& mask)
        _NDArrayViewPtr Data()

    cdef cppclass _ValuePtr "CNTK::ValuePtr":
        _ValuePtr() except +
        _ValuePtr(_Value*) except +
        _Value& operator*()

    cdef cppclass _BackPropState "CNTK::BackPropState":
        pass

    cdef cppclass _BackPropStatePtr "CNTK::BackPropStatePtr":
        pass

    cdef cppclass _Variable "CNTK::Variable":
        _Variable() except +
        _Variable(const _NDShape& shape, bool isSparse, _DataType dataType, bool needsGradient, wstring& name) except + 
        _NDShape Shape()
        #wstring Name() const

    cdef cppclass _SimpleMap "CNTK::_Internal::_SimpleMap":
        pass

    #ctypedef _SimpleMap<_Variable, _ValuePtr>& __VariableValueMap; 

    cdef cppclass _SimpleSet "CNTK::_Internal::_SimpleSet":
        pass

    cdef cppclass _FunctionPtr "CNTK::FunctionPtr":
        _Function* GetPtr() except +
        _Function& operator*() except +
    
    cdef cppclass _Function "CNTK::Function":
        _Variable Output() except +
        vector[_Variable] Outputs() except +

        _BackPropStatePtr Forward(
                unordered_map[_Variable, _ValuePtr]& arguments,
                unordered_map[_Variable, _ValuePtr]& outputs,
                _DeviceDescriptor& computeDevice,
                unordered_set[_Variable]& outputsToRetainBackwardStateFor) except +

        unordered_map[_Variable, _ValuePtr]* createMap()
        #$void Backward(const BackPropStatePtr& state,
                              #_SimpleMap[_Variable, ValuePtr]& rootGradientValues,
                              #_SimpleMap[_Variable, ValuePtr]& backPropagatedGradientValuesForInputs);

    cdef _FunctionPtr Plus(const _Variable& leftOperand, const _Variable& rightOperand) except +#, const std::wstring& name = L"");

cdef class DeviceDescriptor:
    cdef _DeviceDescriptor* thisptr

    def __init__(self, device_type='default', device_id=None):
        '''
        Args:
            device_type (str): 'cpu', 'gpu', or 'default'
        '''
        if device_type == 'cpu':
            thisptr = _DeviceDescriptor.CPUDevice()
        elif device_type == 'gpu':
            thisptr = _DeviceDescriptor.GPUDevice(device_id)
        elif device_type == 'default':
            thisptr = _DeviceDescriptor.DefaultDevice()
        else:
            if device_type != 'gpu':
                raise ValueError('Device type "%s" is not supported. Please choose among "cpu", "gpu", or "default".'%device_type)

cdef class NDShape:
    cdef _NDShape* thisptr

    def __init__(self, shape):
        '''
        Args:
            shape (tuple): shape dimensions
        '''
        cdef vector[size_t] dim_vect = shape
        self.thisptr = new _NDShape(dim_vect)

cdef class NDArrayViewWithValue:
    cdef _NDArrayViewPtr* thisptr

    def __init__(self, shape, value):
        # FIXME: For some reason device cannot be used to initialize dev
        cdef _DeviceDescriptor dev = _DeviceDescriptor.CPUDevice()
        self.thisptr = new _NDArrayViewPtr(new _NDArrayView(value, deref(NDShape(shape).thisptr), dev, 0))

cdef class NDArrayView:
    cdef _NDArrayViewPtr* thisptr

    def __init__(self, shape):
        cdef _DeviceDescriptor dev = _DeviceDescriptor.CPUDevice()
        self.thisptr = new _NDArrayViewPtr(new _NDArrayView(Double, deref(NDShape(shape).thisptr), dev))



cdef class Variable:
    cdef _Variable thisptr

    @staticmethod
    cdef create(_Variable thisptr):
        cdef Variable v = Variable()
        v.thisptr = thisptr
        return v

    def __init__(self, shape=None, is_sparse=False, needs_gradient=True, name=""):
        cdef vector[size_t] dim_vect
        cdef _NDShape* ndshape
        cdef bool cntk_sparse

        cdef Py_ssize_t length
        cdef wchar_t *wchar_cntk_name
        cdef wstring * cntk_name

        if shape is not None:
            dim_vect = shape
            ndshape = new _NDShape(dim_vect)
            cntk_sparse = is_sparse
            wchar_cntk_name = PyUnicode_AsWideCharString(name, &length)
            cntk_name = new wstring(wchar_cntk_name)
            self.thisptr = _Variable(deref(ndshape), cntk_sparse, Double, needs_gradient, deref(cntk_name))

    def get_shape(self):
        # the following is not possible, because _NDShape does not have a default constructor and Cython requires
        # it before it can put stuff on the stack
        #cdef _NDShape* nds = self.thisptr.Shape()[0]
        shape = []
        for i in range(self.thisptr.Shape().NumAxes()):
            shape.append(<int>self.thisptr.Shape()[i])
        return tuple(shape)


def test():
    shape = (2,3)#left.shape
    left_val_shape = shape+(1,1) # seq_len, sample
    right_val_shape = shape+(1,1) # seq_len, sample

    # initializing left and right parameters
    left = Variable(shape)#, name="left")
    print("left shape=%s"%str(left.get_shape()))
    right = Variable(shape)
    print("right shape=%s"%str(right.get_shape()))

    # setting up the Plus operator
    cdef _FunctionPtr plusFunc = Plus(left.thisptr, right.thisptr)

    # initializing outputs
    cdef unordered_map[_Variable, _ValuePtr] outputs  

    cdef _Variable _outputVariable = plusFunc.GetPtr().Output()
    outputVariable = Variable.create(_outputVariable)
    cdef _ValuePtr outputValue = _ValuePtr(new _Value(deref(NDArrayView(outputVariable.get_shape()+(1,1)).thisptr)))
    outputs[_outputVariable] = outputValue


    # setting up values for left and right
    cdef _ValuePtr left_value  = _ValuePtr(new _Value(deref(NDArrayViewWithValue(left_val_shape, <double>2).thisptr)))
    cdef _ValuePtr right_value = _ValuePtr(new _Value(deref(NDArrayViewWithValue(right_val_shape, <double>5).thisptr)))

    cdef unordered_map[_Variable, _ValuePtr] arguments = unordered_map[_Variable, _ValuePtr]()
    cdef pair[_Variable, _ValuePtr] left_pair = pair[_Variable,_ValuePtr](left.thisptr, left_value)
    cdef pair[_Variable, _ValuePtr] right_pair = pair[_Variable,_ValuePtr](right.thisptr, right_value)
    arguments.insert(left_pair)
    arguments.insert(right_pair)

    # retain no outputs
    cdef unordered_set[_Variable] outputsToRetainBackwardStateFor

    # forward pass
    cdef _BackPropStatePtr _bprop
    _bprop = deref(plusFunc.GetPtr()).Forward(arguments, outputs, _DeviceDescriptor.CPUDevice(), outputsToRetainBackwardStateFor)

    # std::unordered_map<Variable, ValuePtr> outputs = { { plusFunc->Output(), outputValue } };
    # auto backPropState = plusFunc->Forward({ { leftInputVar, leftInputValue }, { rightInputVar, rightInputValue } }, outputs, device, { plusFunc->Output() });    

    cdef  double* buf = <double*>deref(deref(outputValue).Data()).DataBufferD()
    #cdef double[:,::1] membuf = buf
    #cdef double* testbuf = [1,2]
    #testbuf[0] = 8
    #testbuf[1] = 9

    num_output_axes = len(outputVariable.get_shape())
    cdef double[::1] blaview = <double[:6]>(buf)

    nda = np.asarray(blaview, dtype=np.float64).reshape(shape)
    #nda[1] = 1111
    #cdef cnp.ndarray[double, ndim=2, mode='c'] nda = buf
    #print(testbuf[0])
    #print(testbuf[1])

    #return nda[:]
    return nda


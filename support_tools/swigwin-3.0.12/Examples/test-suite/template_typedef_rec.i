%module template_typedef_rec

%inline %{
// --- includes required to compile the wrapper code ---
typedef size_t MY_sizeT;
typedef long MY_intT;
typedef double MY_floatT;

class test_Array
{
public:
  typedef MY_intT    intT;
  typedef MY_sizeT   sizeT;
};



template <typename T>
class ArrayIterator_
{
 public:
  typedef test_Array::intT    intT;
};


template <typename T>
class ArrayReverseIterator
{
 public:
  typedef test_Array::intT    intT;
};


template <typename T>
class ArrayPrimitiveT 
  : public test_Array
{
public:
  typedef T  ValueT;
  typedef T  valueT;
  typedef ArrayIterator_<T>       Iterator;
  typedef ArrayIterator_<const T> ConstIterator;  
  typedef ArrayReverseIterator<T>       ReverseIterator;
  typedef ArrayReverseIterator<const T> ConstReverseIterator;
};


template <class T>
class TreeNode
{
public:
  typedef T  ValueT;
  typedef T  valueT;
  typedef MY_intT    intT;
  typedef MY_sizeT   sizeT;
};

template <class T>
struct ArrayPointerT
{
};

template <class T>
class TreeIterator
{
public:  
  typedef MY_intT    intT;
  typedef MY_sizeT   sizeT;
  typedef ArrayPointerT< T* > NodeArrayT;
  
};


template <class T>
class Tree
{
public:
  typedef T  ValueT;
  typedef T  valueT;
  typedef MY_intT    intT;
  typedef MY_sizeT   sizeT;
  typedef TreeNode<T> NodeT;
  typedef ArrayPointerT< NodeT* > NodeArrayT;
  typedef TreeIterator<NodeT> Iterator;
  typedef TreeIterator<NodeT> ConstIterator;
 

};


class ModelNode
{
  typedef MY_intT    intT;
  typedef MY_floatT  floatT;
  typedef MY_sizeT   sizeT;
  
  
};

class Model
{
  typedef MY_intT    intT;
  typedef MY_sizeT   sizeT;
  typedef Tree<ModelNode> TreeT;
  typedef TreeT::NodeT  TreeNodeT;
  typedef TreeT::Iterator TreeIteratorT;
  
};
%}


// --- define ANSI C/C++ declarations to be interfaced ---
%template(ModelTree)  Tree<ModelNode>;

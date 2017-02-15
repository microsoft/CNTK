/* -----------------------------------------------------------------------------
 * std_unordered_set.i
 *
 * SWIG typemaps for std::unordered_set< K >
 *
 * The C# wrapper is made to look and feel like a C# System.Collections.Generic.ICollection<>.
 * 
 * Using this wrapper is fairly simple. For example, to create a unordered_set from integers use:
 *
 *   %include <std_unordered_set.i>
 *   %template(UnorderdSetInt) std::unordered_set<int>
 *
 * Notes:
 * 1) IEnumerable<> is implemented in the proxy class which is useful for using LINQ with 
 *    C++ std::unordered_set wrappers.
 *
 * Warning: heavy macro usage in this file. Use swig -E to get a sane view on the real file contents!
 * ----------------------------------------------------------------------------- */

%{
#include <unordered_set>
#include <algorithm>
#include <stdexcept>
%}

/* T is the C++ value type */
%define SWIG_STD_UNORDERED_SET_INTERNAL(T)

%typemap(csinterfaces) std::unordered_set< T > "global::System.IDisposable,\n   global::System.Collections.IEnumerable,\n    global::System.Collections.Generic.ICollection<$typemap(cstype, T)>\n";
%typemap(cscode) std::unordered_set<T> %{

		public $csclassname(global::System.Collections.IList list) : this() 
		{
			if (list == null) throw new global::System.ArgumentNullException("list");
			foreach ($typemap(cstype, T) elem in list) 
			{
				this.Add(elem);
			}
		}

		public int Count 
		{
			get { return (int)size(); }
		}

		public void Clear()
		{
			clear();
		}

		public bool IsReadOnly 
	    { 
			get { return false; }
	    }

		  public void CopyTo($typemap(cstype, T)[] array)
		  {
			  CopyTo(0, array, 0, this.Count);
		  }

		  public void CopyTo($typemap(cstype, T)[] array, int arrayIndex)
		  {
			  CopyTo(0, array, arrayIndex, this.Count);
		  }

		  public void CopyTo(int index, $typemap(cstype, T)[] array, int arrayIndex, int count)
		  {
			if (array == null) throw new global::System.ArgumentNullException("array");
			if (index < 0) throw new global::System.ArgumentOutOfRangeException("index", "Value is less than zero");
			if (arrayIndex < 0) throw new global::System.ArgumentOutOfRangeException("arrayIndex", "Value is less than zero");
			if (count < 0) throw new global::System.ArgumentOutOfRangeException("count", "Value is less than zero");
			if (array.Rank > 1)  throw new global::System.ArgumentException("Multi dimensional array.", "array");
			if (index+count > this.Count || arrayIndex+count > array.Length) throw new global::System.ArgumentException("Number of elements to copy is too large.");

			int itercount = 0, iterindex = 0;
			foreach ($typemap(cstype, T) val in this)
			{
				if (iterindex >= index)
				{
					array.SetValue(val, arrayIndex+itercount);
					itercount ++;
				}
				if (itercount >= count) break;
				iterindex ++; 
			}			
		  }

		  global::System.Collections.Generic.IEnumerator<$typemap(cstype, T)> global::System.Collections.Generic.IEnumerable<$typemap(cstype, T)>.GetEnumerator() {
			return new $csclassnameEnumerator(this);
		  }

		  global::System.Collections.IEnumerator global::System.Collections.IEnumerable.GetEnumerator() {
			return new $csclassnameEnumerator(this);
		  }

		  public $csclassnameEnumerator GetEnumerator() {
			return new $csclassnameEnumerator(this);
		  }

		  // Type-safe enumerator
		  /// Note that the IEnumerator documentation requires an InvalidOperationException to be thrown
		  /// whenever the collection is modified. This has been done for changes in the size of the
		  /// collection but not when one of the elements of the collection is modified as it is a bit
		  /// tricky to detect unmanaged code that modifies the collection under our feet.
		  public sealed class $csclassnameEnumerator : global::System.Collections.IEnumerator
			, global::System.Collections.Generic.IEnumerator<$typemap(cstype, T)>
		  {
			private $csclassname collectionRef;
			private int currentIndex;
			private object currentObject;
			private int currentSize;
			private System.IntPtr iterator;

			public $csclassnameEnumerator($csclassname collection) {
			  collectionRef = collection;
			  currentIndex = -1;
			  currentObject = null;
			  currentSize = collectionRef.Count;
			   iterator = collectionRef.create_iterator_begin();   
			}

			// Type-safe iterator Current
			public $typemap(cstype, T) Current {
			  get {
				if (currentIndex == -1)
				  throw new global::System.InvalidOperationException("Enumeration not started.");
				if (currentIndex > currentSize - 1)
				  throw new global::System.InvalidOperationException("Enumeration finished.");
				if (currentObject == null)
				  throw new global::System.InvalidOperationException("Collection modified.");
				return ($typemap(cstype, T))currentObject;
			  }
			}

			// Type-unsafe IEnumerator.Current
			object global::System.Collections.IEnumerator.Current {
			  get {
				return Current;
			  }
			}

			public bool MoveNext() {
			  int size = collectionRef.Count;
			  bool moveOkay = (currentIndex + 1 < size) && (size == currentSize);
			  if (moveOkay) {
				currentIndex++;
				currentObject = collectionRef.get_next(iterator);
			  } else {
				currentObject = null;
			  }
			  return moveOkay;
			}

			public void Reset() {
			  currentIndex = -1;
			  currentObject = null;
			  if (collectionRef.Count != currentSize) {
				throw new global::System.InvalidOperationException("Collection modified.");
			  }
			}

			public void Dispose() {
				currentIndex = -1;
				currentObject = null;
				collectionRef.destroy_iterator(iterator);
			}
		  }  
%}

    public:
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;
        typedef T value_type;

        unordered_set();
        unordered_set(const unordered_set<T> &other);
        size_type size() const;
        bool empty() const;
        void clear();

        %extend {
            
            bool Contains(const value_type& val) 
			{
				std::unordered_set<T>::iterator iter = $self->find(val);
				return iter != $self->end();
            }

            void Add(const value_type& val) throw (std::out_of_range) 
			{
				$self->insert(val);
            }

            bool Remove(const value_type& val) 
			{
               std::unordered_set<T>::iterator iter = $self->find(val);
               if (iter != $self->end()) 
			   {
                   $self->erase(iter);
                   return true;
               }                
               return false;
            }

			// create_iterator_begin(), get_next() and destroy_iterator work together to provide a collection of keys to C#
            %apply void *VOID_INT_PTR { std::unordered_set<T>::iterator *create_iterator_begin }
            %apply void *VOID_INT_PTR { std::unordered_set<T>::iterator *swigiterator }

            std::unordered_set<T>::iterator *create_iterator_begin() {
               return new std::unordered_set<T>::iterator($self->begin());
            }

            const value_type& get_next(std::unordered_set<T>::iterator *swigiterator) {
				std::unordered_set<T>::iterator iter = *swigiterator;
				(*swigiterator)++;
                return (*iter);
            }

            void destroy_iterator(std::unordered_set<T>::iterator *swigiterator) {
                delete swigiterator;
            }

        }

%enddef

%csmethodmodifiers std::unordered_set::size "private"
%csmethodmodifiers std::unordered_set::getitem "private"
%csmethodmodifiers std::unordered_set::setitem "private"
%csmethodmodifiers std::unordered_set::create_iterator_begin "private"
%csmethodmodifiers std::unordered_set::get_next "private"
%csmethodmodifiers std::unordered_set::destroy_iterator "private"

// Default implementation
namespace std {   
  template<class T> class unordered_set {
    SWIG_STD_UNORDERED_SET_INTERNAL(T)
  };
}
 
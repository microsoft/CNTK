/* -----------------------------------------------------------------------------
 * std_unordered_map.i
 *
 * SWIG typemaps for std::unordered_map< K, T>
 *
 * The C# wrapper is made to look and feel like a C# System.Collections.Generic.IDictionary<>.
 * 
 * Using this wrapper is fairly simple. For example, to create a unordered_map from integers to doubles use:
 *
 *   %include <std_unordered_map.i>
 *   %template(MapIntDouble) std::unordered_map<int, double>
 *
 * Notes:
 * 1) IEnumerable<> is implemented in the proxy class which is useful for using LINQ with 
 *    C++ std::unordered_map wrappers.
 *
 * Warning: heavy macro usage in this file. Use swig -E to get a sane view on the real file contents!
 * ----------------------------------------------------------------------------- */

%{
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
%}

/* K is the C++ key type, T is the C++ value type */
%define SWIG_STD_UNORDERED_MAP_INTERNAL(K, T)

%typemap(csinterfaces) std::unordered_map< K, T > "global::System.IDisposable \n    , global::System.Collections.Generic.IDictionary<$typemap(cstype, K), $typemap(cstype, T)>\n";
%typemap(cscode) std::unordered_map<K, T> %{

    public $typemap(cstype, T) this[$typemap(cstype, K) key]
    {
        get 
        {
            return getitem(key);
        }

        set 
        {
            setitem(key, value);
        }
    }

    public bool TryGetValue($typemap(cstype, K) key, out $typemap(cstype, T) value) 
    {
        if (this.ContainsKey(key)) 
        {
            value = this[key];
            return true;
        }
        value = default($typemap(cstype, T));
        return false;
    }

    public int Count 
    {
        get 
        {
            return (int)size();
        }
    }

    public bool IsReadOnly 
    {
        get 
        { 
            return false; 
        }
    }

    public global::System.Collections.Generic.ICollection<$typemap(cstype, K)> Keys 
    {
        get 
        {
            global::System.Collections.Generic.ICollection<$typemap(cstype, K)> keys = new global::System.Collections.Generic.List<$typemap(cstype, K)>();
            int size = this.Count;
            if (size > 0) 
            {
                global::System.IntPtr iter = create_iterator_begin();
                for (int i = 0; i < size; i++) 
                {
                    keys.Add(get_next_key(iter));
                }
                destroy_iterator(iter);
            }
            return keys;
        }
    }

    public global::System.Collections.Generic.ICollection<$typemap(cstype, T)> Values 
    {
        get 
        {
            global::System.Collections.Generic.ICollection<$typemap(cstype, T)> vals = new global::System.Collections.Generic.List<$typemap(cstype, T)>();
            foreach (global::System.Collections.Generic.KeyValuePair<$typemap(cstype, K), $typemap(cstype, T)> pair in this) 
            {
                vals.Add(pair.Value);
            }
            return vals;
        }
    }
  
    public void Add(global::System.Collections.Generic.KeyValuePair<$typemap(cstype, K), $typemap(cstype, T)> item) 
    {
        Add(item.Key, item.Value);
    }

    public bool Remove(global::System.Collections.Generic.KeyValuePair<$typemap(cstype, K), $typemap(cstype, T)> item) 
    {
        if (Contains(item)) 
        {
            return Remove(item.Key);
        } 
        else 
        {
            return false;
        }
    }

    public bool Contains(global::System.Collections.Generic.KeyValuePair<$typemap(cstype, K), $typemap(cstype, T)> item) 
    {
        if (this[item.Key] == item.Value) 
        {
            return true;
        } 
        else
        {
            return false;
        }
    }

    public void CopyTo(global::System.Collections.Generic.KeyValuePair<$typemap(cstype, K), $typemap(cstype, T)>[] array) 
    {
        CopyTo(array, 0);
    }

    public void CopyTo(global::System.Collections.Generic.KeyValuePair<$typemap(cstype, K), $typemap(cstype, T)>[] array, int arrayIndex) 
    {
        if (array == null)
        {
            throw new global::System.ArgumentNullException("array");
        }
    
        if (arrayIndex < 0)
        {
            throw new global::System.ArgumentOutOfRangeException("arrayIndex", "Value is less than zero");
        }
        if (array.Rank > 1) 
        {
            throw new global::System.ArgumentException("Multi dimensional array.", "array");
        }
        if (arrayIndex+this.Count > array.Length)
        {
            throw new global::System.ArgumentException("Number of elements to copy is too large.");
        }

        global::System.Collections.Generic.IList<$typemap(cstype, K)> keyList = new global::System.Collections.Generic.List<$typemap(cstype, K)>(this.Keys);
        for (int i = 0; i < keyList.Count; i++)
        {
            $typemap(cstype, K) currentKey = keyList[i];
            array.SetValue(new global::System.Collections.Generic.KeyValuePair<$typemap(cstype, K), $typemap(cstype, T)>(currentKey, this[currentKey]), arrayIndex+i);
        }
    }

    global::System.Collections.Generic.IEnumerator<global::System.Collections.Generic.KeyValuePair<$typemap(cstype, K), $typemap(cstype, T)>> global::System.Collections.Generic.IEnumerable<global::System.Collections.Generic.KeyValuePair<$typemap(cstype, K), $typemap(cstype, T)>>.GetEnumerator() 
    {
        return new $csclassnameEnumerator(this);
    }

    global::System.Collections.IEnumerator global::System.Collections.IEnumerable.GetEnumerator() 
    {
        return new $csclassnameEnumerator(this);
    }

    public $csclassnameEnumerator GetEnumerator() 
    {
        return new $csclassnameEnumerator(this);
    }

    // Type-safe enumerator
    /// Note that the IEnumerator documentation requires an InvalidOperationException to be thrown
    /// whenever the collection is modified. This has been done for changes in the size of the
    /// collection but not when one of the elements of the collection is modified as it is a bit
    /// tricky to detect unmanaged code that modifies the collection under our feet.
    public sealed class $csclassnameEnumerator : global::System.Collections.IEnumerator, 
        global::System.Collections.Generic.IEnumerator<global::System.Collections.Generic.KeyValuePair<$typemap(cstype, K), $typemap(cstype, T)>>
    {
        private $csclassname collectionRef;
        private global::System.Collections.Generic.IList<$typemap(cstype, K)> keyCollection;
        private int currentIndex;
        private object currentObject;
        private int currentSize;

        public $csclassnameEnumerator($csclassname collection) 
        {
            collectionRef = collection;
            keyCollection = new global::System.Collections.Generic.List<$typemap(cstype, K)>(collection.Keys);
            currentIndex = -1;
            currentObject = null;
            currentSize = collectionRef.Count;
        }

        // Type-safe iterator Current
        public global::System.Collections.Generic.KeyValuePair<$typemap(cstype, K), $typemap(cstype, T)> Current 
        {
            get 
            {
                if (currentIndex == -1)
                {
                    throw new global::System.InvalidOperationException("Enumeration not started.");
                }
                if (currentIndex > currentSize - 1)
                {
                    throw new global::System.InvalidOperationException("Enumeration finished.");
                }
                if (currentObject == null)
                {
                    throw new global::System.InvalidOperationException("Collection modified.");
                }
                return (global::System.Collections.Generic.KeyValuePair<$typemap(cstype, K), $typemap(cstype, T)>)currentObject;
            }
        }

        // Type-unsafe IEnumerator.Current
        object global::System.Collections.IEnumerator.Current
        {
            get 
            {
                return Current;
            }
        }

        public bool MoveNext() 
        {
            int size = collectionRef.Count;
            bool moveOkay = (currentIndex+1 < size) && (size == currentSize);
            if (moveOkay) 
            {
                currentIndex++;
                $typemap(cstype, K) currentKey = keyCollection[currentIndex];
                currentObject = new global::System.Collections.Generic.KeyValuePair<$typemap(cstype, K), $typemap(cstype, T)>(currentKey, collectionRef[currentKey]);
            } 
            else 
            {
                currentObject = null;
            }
            return moveOkay;
        }

        public void Reset() 
        {
            currentIndex = -1;
            currentObject = null;
            if (collectionRef.Count != currentSize) 
            {
                throw new global::System.InvalidOperationException("Collection modified.");
            }
        }

        public void Dispose() 
        {
            currentIndex = -1;
            currentObject = null;
        }
    }
  
%}

    public:
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;
        typedef K key_type;
        typedef T mapped_type;

        unordered_map();
        unordered_map(const unordered_map< K, T> &other);
        size_type size() const;
        bool empty() const;
        %rename(Clear) clear;
        void clear();
        %extend {
            const mapped_type& getitem(const key_type& key) throw (std::out_of_range) {
            std::unordered_map< K, T>::iterator iter = $self->find(key);
            if (iter != $self->end())
                return iter->second;
            else
                throw std::out_of_range("key not found");
            }

            void setitem(const key_type& key, const mapped_type& x) {
            (*$self)[key] = x;
            }

            bool ContainsKey(const key_type& key) {
            std::unordered_map< K, T>::iterator iter = $self->find(key);
            return iter != $self->end();
            }

            void Add(const key_type& key, const mapped_type& val) throw (std::out_of_range) {
            std::unordered_map< K, T>::iterator iter = $self->find(key);
            if (iter != $self->end())
                throw std::out_of_range("key already exists");
            $self->insert(std::pair< K, T >(key, val));
            }

            bool Remove(const key_type& key) {
            std::unordered_map< K, T>::iterator iter = $self->find(key);
            if (iter != $self->end()) {
                $self->erase(iter);
                return true;
            }                
            return false;
            }

            // create_iterator_begin(), get_next_key() and destroy_iterator work together to provide a collection of keys to C#
            %apply void *VOID_INT_PTR { std::unordered_map< K, T>::iterator *create_iterator_begin }
            %apply void *VOID_INT_PTR { std::unordered_map< K, T>::iterator *swigiterator }

            std::unordered_map< K, T>::iterator *create_iterator_begin() {
            return new std::unordered_map< K, T>::iterator($self->begin());
            }

            const key_type& get_next_key(std::unordered_map< K, T>::iterator *swigiterator) {
            std::unordered_map< K, T>::iterator iter = *swigiterator;
            (*swigiterator)++;
            return (*iter).first;
            }

            void destroy_iterator(std::unordered_map< K, T>::iterator *swigiterator) {
            delete swigiterator;
            }
        }

%enddef

%csmethodmodifiers std::unordered_map::size "private"
%csmethodmodifiers std::unordered_map::getitem "private"
%csmethodmodifiers std::unordered_map::setitem "private"
%csmethodmodifiers std::unordered_map::create_iterator_begin "private"
%csmethodmodifiers std::unordered_map::get_next_key "private"
%csmethodmodifiers std::unordered_map::destroy_iterator "private"

// Default implementation
namespace std {   
  template<class K, class T> class unordered_map {
    SWIG_STD_UNORDERED_MAP_INTERNAL(K, T)
  };
}
 

// Legacy macros (deprecated)
%define specialize_std_map_on_key(K,CHECK,CONVERT_FROM,CONVERT_TO)
#warning "specialize_std_map_on_key ignored - macro is deprecated and no longer necessary"
%enddef

%define specialize_std_map_on_value(T,CHECK,CONVERT_FROM,CONVERT_TO)
#warning "specialize_std_map_on_value ignored - macro is deprecated and no longer necessary"
%enddef

%define specialize_std_map_on_both(K,CHECK_K,CONVERT_K_FROM,CONVERT_K_TO, T,CHECK_T,CONVERT_T_FROM,CONVERT_T_TO)
#warning "specialize_std_map_on_both ignored - macro is deprecated and no longer necessary"
%enddef


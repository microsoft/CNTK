/* -----------------------------------------------------------------------------
 * std_unordered_map.i
 *
 * SWIG typemaps for std::unordered_map<K, V>
 * ----------------------------------------------------------------------------- */

%{
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <iostream>
%}

%define SWIG_STD_UNORDERED_MAP_INTERNAL(K, V)

%typemap(javaimports) std::unordered_map<K, V> "import java.util.*;"
%typemap(javainterfaces) std::unordered_map<K, V> "java.util.Map<$typemap(jstype, K), $typemap(jstype, V)>";
%typemap(javacode) std::unordered_map<K, V> %{
    public Set<Map.Entry<$typemap(jstype, K), $typemap(jstype, V)>> entrySet() {
        HashSet<Map.Entry<$typemap(jstype, K), $typemap(jstype, V)>> retVal =
            new HashSet<Map.Entry<$typemap(jstype, K), $typemap(jstype, V)>>();
        for ($typemap(jstype, K) k : keySet()) {
            retVal.add(new Entry(k, get(k)));
        }
        return retVal;
    }

    public Set<$typemap(jstype, K)> keySet() {
        Set keySet = new HashSet<$typemap(jstype, K)>(size());
        if (size() > 0) {
            $typemap(jstype, std::unordered_map<K, V>::iterator) iter = create_iterator_begin();
            for (int i = 0; i < size(); i++) {
                keySet.add(get_next_key(iter));
            }
            destroy_iterator(iter);
        }
        return keySet;
    }

    public Collection<$typemap(jstype, V)> values() {
        Set valueSet = new HashSet<$typemap(jstype, K)>(size());
        if (size() > 0) {
            $typemap(jstype, std::unordered_map<K, V>::iterator) iter = create_iterator_begin();
            for (int i = 0; i < size(); i++) {
                valueSet.add(get_next_value(iter));
            }
            destroy_iterator(iter);
        }
        return valueSet;
    }

    public void putAll(Map<? extends $typemap(jstype, K), ? extends $typemap(jstype, V)> m) {
        throw new UnsupportedOperationException();
    }

    public $typemap(jstype, V) remove(Object o) {
        if (o == null) throw new NullPointerException("Key cannot be null.");
        $typemap(jstype, K) castedKey = ($typemap(jstype, K))o;
        if (castedKey == null) throw new ClassCastException("Argument could not be cast to type of key");
        $typemap(jstype, V) old = get(castedKey);
        _Remove(castedKey);
        return old;
    }

    public $typemap(jstype, V) put($typemap(jstype, K) key, $typemap(jstype, V) value) {
        if (key == null) throw new NullPointerException("Key cannot be null.");
        $typemap(jstype, V) old = get(key);
        setitem(key, value);
        return old;
    }

    public $typemap(jstype, V) get(Object key) {
        if (key == null) throw new NullPointerException("Key cannot be null.");
        $typemap(jstype, K) castedKey = ($typemap(jstype, K))key;
        if (castedKey == null) throw new ClassCastException("Argument could not be cast to type of key");
        try {
            return getitem(castedKey);
        } catch (IndexOutOfBoundsException _) {
            return null;
        }
    }

    public boolean containsKey(Object key) {
        $typemap(jstype, K) maybeKey = ($typemap(jstype, K))key;
        if (maybeKey == null) {
            return false;
        } else {
            return _ContainsKey(maybeKey);
        }
    }

    public boolean containsValue(Object value) {
        $typemap(jstype, V) maybeValue = ($typemap(jstype, V))value;
        if (maybeValue == null) {
            return false;
        } else {
            return _ContainsValue(maybeValue);
        }
    }

    public int size() {
        return (int)_Size();
    }

    public static class Entry implements Map.Entry<$typemap(jstype, K), $typemap(jstype, V)> {
        private final $typemap(jstype, K) key;
        private $typemap(jstype, V) value;

        public Entry($typemap(jstype, K) key, $typemap(jstype, V) value) {
            this.key = key;
            this.value = value;
        }

        public boolean equals(Object o) {
            if (o == null) return false;
            if (getClass() != o.getClass()) return false;
            final Entry other = (Entry) o;
            if (!Objects.equals(this.key, other.key)) return false;
            if (!Objects.equals(this.value, other.value)) return false;
            return true;
        }

        public $typemap(jstype, K) getKey() {
            return key;
        }

        public $typemap(jstype, V) getValue() {
            return value;
        }

        public int hashCode() {
            return Objects.hash(this.key, this.value);
        }

        public $typemap(jstype, V) setValue($typemap(jstype, V) value) {
            $typemap(jstype, V) old = this.value;
            this.value = value;
            return old;
        }
    }
%}

    public:
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;
        typedef K key_type;
        typedef V value_type;

        unordered_map();
        unordered_map(const unordered_map<K, V> &other);
        %rename(_Size) size;
        size_type size() const;
        %rename(isEmpty) empty;
        bool empty() const;
        void clear();
        %extend {
            const V& getitem(const key_type& key) throw (std::out_of_range) {
                std::unordered_map<K, V>::iterator iter = $self->find(key);
                if (iter != $self->end())
                    return iter->second;
                else
                    throw std::out_of_range("key not found");
            }

            void setitem(const key_type& key, const V& x) {
                (*$self)[key] = x;
            }

            bool _ContainsKey(const key_type& key) {
                std::unordered_map<K, V>::iterator iter = $self->find(key);
                return iter != $self->end();
            }

            bool _ContainsValue(const V& value) {
                for (auto kv : *$self) {
                    if (kv.second == value) return true;
                }
                return false;
            }

            void add(const key_type& key, const V& val) throw (std::out_of_range) {
                std::unordered_map<K, V>::iterator iter = $self->find(key);
                if (iter != $self->end())
                    throw std::out_of_range("key already exists");
                $self->insert(std::pair<K, V>(key, val));
            }

            bool _Remove(const key_type& key) {
                std::unordered_map<K, V>::iterator iter = $self->find(key);
                if (iter != $self->end()) {
                    $self->erase(iter);
                    return true;
                }
                return false;
            }
            // create_iterator_begin(), get_next_key() and destroy_iterator
            // work together to provide a collection of keys to Java
            %apply void *VOID_INT_PTR { std::unordered_map<K, V>::iterator *create_iterator_begin }
            %apply void *VOID_INT_PTR { std::unordered_map<K, V>::iterator *swigiterator }

            std::unordered_map<K, V>::iterator *create_iterator_begin() {
                return new std::unordered_map<K, V>::iterator($self->begin());
            }

            const key_type& get_next_key(std::unordered_map<K, V>::iterator *swigiterator) {
                std::unordered_map<K, V>::iterator iter = *swigiterator;
                (*swigiterator)++;
                return iter->first;
            }

            const V& get_next_value(std::unordered_map<K, V>::iterator *swigiterator) {
                std::unordered_map<K, V>::iterator iter = *swigiterator;
                (*swigiterator)++;
                return iter->second;
            }

            void destroy_iterator(std::unordered_map<K, V>::iterator *swigiterator) {
                delete swigiterator;
            }
        }


%enddef

%javamethodmodifiers std::unordered_map::getitem "private"
%javamethodmodifiers std::unordered_map::setitem "private"
%javamethodmodifiers std::unordered_map::_Remove "private"
%javamethodmodifiers std::unordered_map::create_iterator_begin "private"
%javamethodmodifiers std::unordered_map::get_next_key "private"
%javamethodmodifiers std::unordered_map::destroy_iterator "private"

namespace std {
  template<class K, class V> class unordered_map {
    SWIG_STD_UNORDERED_MAP_INTERNAL(K, V)
  };
}

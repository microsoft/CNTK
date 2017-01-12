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

namespace std {
  template<class K, class T> class unordered_map {
    %typemap(csinterfaces) std::unordered_map< K, T > "global::System.IDisposable \n    , global::System.Collections.Generic.IDictionary<$typemap(cstype, K), $typemap(cstype, T)>\n";
    %typemap(java) std::unordered_map<K, T> %{%}

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
      };
}

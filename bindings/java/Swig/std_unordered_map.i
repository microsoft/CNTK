/* -----------------------------------------------------------------------------
 * std_unordered_map.i
 *
 * SWIG typemaps for std::unordered_map< K, T>
 * ----------------------------------------------------------------------------- */

%{
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
%}

namespace std {
  template<class K, class T> class unordered_map {
    %typemap(java) std::unordered_map<K, T> %{
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

                bool containsKey(const key_type& key) {
                std::unordered_map< K, T>::iterator iter = $self->find(key);
                return iter != $self->end();
                }

                void add(const key_type& key, const mapped_type& val) throw (std::out_of_range) {
                std::unordered_map< K, T>::iterator iter = $self->find(key);
                if (iter != $self->end())
                    throw std::out_of_range("key already exists");
                $self->insert(std::pair< K, T >(key, val));
                }

                bool remove(const key_type& key) {
                std::unordered_map< K, T>::iterator iter = $self->find(key);
                if (iter != $self->end()) {
                    $self->erase(iter);
                    return true;
                }
                return false;
                }
            }
      };
}

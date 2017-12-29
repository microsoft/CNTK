%module template_default_arg_overloaded

// Github issue #529

%include <std_string.i>

%inline %{
#include <string>

namespace lsst {
namespace daf {
namespace bass {

class PropertyList {
public:
    PropertyList(void) {};

    virtual ~PropertyList(void) {};

    template <typename T> int set(std::string const& name1, T const& value1, bool inPlace1=true) { return 1; }

    int set(std::string const& name2, PropertyList const& value2, bool inPlace2=true) { return 2; }

    template <typename T> int set(std::string const& name3, T const& value3, std::string const& comment3, bool inPlace3=true) { return 3; }
};

}}} // namespace lsst::daf::bass

// As above but in global namespace
class PropertyListGlobal {
public:
    PropertyListGlobal(void) {};

    virtual ~PropertyListGlobal(void) {};

    template <typename T> int set(std::string const& name1, T const& value1, bool inPlace1=true) { return 1; }

    int set(std::string const& name2, PropertyListGlobal const& value2, bool inPlace2=true) { return 2; }

    template <typename T> int set(std::string const& name3, T const& value3, std::string const& comment3, bool inPlace3=true) { return 3; }
};

%}

%template(setInt) lsst::daf::bass::PropertyList::set<int>;
%template(setIntGlobal) PropertyListGlobal::set<int>;


// Global functions
%inline %{
template<typename T> int goopGlobal(T i1, bool b1 = true) { return 1; }
int goopGlobal(short s2 = 0) { return 2; }
template<typename T> int goopGlobal(const char *s3, bool b3 = true) { return 3; }
%}

// Global functions in a namespace
%inline %{
namespace lsst {
template<typename T> int goop(T i1, bool b1 = true) { return 1; }
int goop(short s2 = 0) { return 2; }
template<typename T> int goop(const char *s3, bool b3 = true) { return 3; }
}
%}

%template(GoopIntGlobal) goopGlobal<int>;
%template(GoopInt) lsst::goop<int>;


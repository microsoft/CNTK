%module template_default_arg_overloaded_extend

%inline %{
namespace gaia2 {

struct Filter {};
struct SearchPoint {};
struct DataSet {};

template <typename SearchPointType, typename DataSetType>
class BaseSearchSpace {};

template <typename SearchPointType, typename DataSetType>
class BaseResultSet {
public:
  const char *over(int i = 0) {
    return "over(int)";
  }
};
}
%}

// Specialized template extend
%extend gaia2::BaseResultSet<gaia2::SearchPoint, gaia2::DataSet> {
  int go_get_method(int n, gaia2::SearchPoint, int offset = -1) {
    return offset;
  }
  const char *over(gaia2::SearchPoint, int x = 0) {
    return "over(giai2::SearchPoint, int)";
  }
}

// Generic extend for all template instantiations
%extend gaia2::BaseResultSet {
  int go_get_template(int n, SearchPointType sss, int offset = -2) {
    return offset;
  }
  const char *over(bool b, SearchPointType, int x = 0) {
    return "over(bool, SearchPointType, int)";
  }
}

%template(ResultSet) gaia2::BaseResultSet<gaia2::SearchPoint, gaia2::DataSet>;


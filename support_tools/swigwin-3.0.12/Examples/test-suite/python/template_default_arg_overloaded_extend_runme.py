from template_default_arg_overloaded_extend import *

def check(flag):
  if not flag:
    raise RuntimeError("failed")

rs = ResultSet()

check(rs.go_get_method(0, SearchPoint()) == -1)
check(rs.go_get_method(0, SearchPoint(), 100) == 100)

check(rs.go_get_template(0, SearchPoint()) == -2)
check(rs.go_get_template(0, SearchPoint(), 100) == 100)

check(rs.over() == "over(int)")
check(rs.over(10) == "over(int)")
check(rs.over(SearchPoint()) == "over(giai2::SearchPoint, int)")
check(rs.over(SearchPoint(), 10) == "over(giai2::SearchPoint, int)")
check(rs.over(True, SearchPoint()) == "over(bool, gaia2::SearchPoint, int)")
check(rs.over(True, SearchPoint(), 10) == "over(bool, gaia2::SearchPoint, int)")

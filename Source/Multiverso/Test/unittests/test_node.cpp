#include <boost/test/unit_test.hpp>
#include <multiverso/node.h>

namespace multiverso {
namespace test {

BOOST_AUTO_TEST_SUITE(node)

BOOST_AUTO_TEST_CASE(node_role) {
  BOOST_CHECK(!multiverso::node::is_worker(multiverso::Role::NONE));
  BOOST_CHECK(multiverso::node::is_worker(multiverso::Role::WORKER));
  BOOST_CHECK(!multiverso::node::is_worker(multiverso::Role::SERVER));
  BOOST_CHECK(multiverso::node::is_worker(multiverso::Role::ALL));

  BOOST_CHECK(!multiverso::node::is_server(multiverso::Role::NONE));
  BOOST_CHECK(!multiverso::node::is_server(multiverso::Role::WORKER));
  BOOST_CHECK(multiverso::node::is_server(multiverso::Role::SERVER));
  BOOST_CHECK(multiverso::node::is_server(multiverso::Role::ALL));
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace test
}  // namespace multiverso
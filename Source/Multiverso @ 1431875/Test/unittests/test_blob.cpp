#include <boost/test/unit_test.hpp>
#include <multiverso/blob.h>

namespace multiverso {
namespace test {

BOOST_AUTO_TEST_SUITE(blob)

BOOST_AUTO_TEST_CASE(blob_constructor_test) {
  multiverso::Blob blob;
  BOOST_CHECK_EQUAL(blob.size(), 0);

  multiverso::Blob blob2(4);
  BOOST_CHECK_EQUAL(blob2.size(), 4);

  int a[3];
  multiverso::Blob blob3(a, 3 * sizeof(int));
  BOOST_CHECK_EQUAL(blob3.size(), 3 * sizeof(int));

}

BOOST_AUTO_TEST_CASE(blob_access_test) {
  multiverso::Blob blob(4);
  BOOST_CHECK_EQUAL(blob.size(), 4);

  const int value = 3;
  int* data = reinterpret_cast<int*>(blob.data());
  *data = value;
  BOOST_CHECK_EQUAL(blob.As<int>(), value);

  std::string str("hello, world!");
  multiverso::Blob str_blob(str.c_str(), str.size());
  BOOST_CHECK_EQUAL(str_blob[0], 'h');
  BOOST_CHECK_EQUAL(str_blob[4], 'o');
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace test
}  // namespace multiverso
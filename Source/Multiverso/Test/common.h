#ifndef MULTIVERSO_TEST_END2ENDTEST_COMMON_H_
#define MULTIVERSO_TEST_END2ENDTEST_COMMON_H_

namespace multiverso {
namespace test {

void TestAllreduce(int argc, char* argv[]);

void TestArray(int argc, char* argv[]);

void TestKV(int argc, char* argv[]);

void TestMatrix(int argc, char* argv[]);

void TestNet(int argc, char* argv[]);

}  // namespace test
}  // namespace multiverso

#endif  // MULTIVERSO_TEST_END2ENDTEST_COMMON_H_
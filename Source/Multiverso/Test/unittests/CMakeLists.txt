enable_testing()

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

SET(MULTIVERSO_UNITTEST_SRC test_array.cpp test_blob.cpp test_kv.cpp test_message.cpp test_multiverso.cpp test_node.cpp test_sync.cpp)

LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH})

MESSAGE(${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
MESSAGE(${Boost_INCLUDE_DIRS})

ADD_EXECUTABLE(multiversotests ${MULTIVERSO_UNITTEST_SRC})

TARGET_INCLUDE_DIRECTORIES(multiversotests PRIVATE ${Boost_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(multiversotests multiverso ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

SET_PROPERTY(TARGET multiversotests PROPERTY CXX_STANDARD 11)

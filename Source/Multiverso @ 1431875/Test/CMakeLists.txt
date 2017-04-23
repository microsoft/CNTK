INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/Test)

SET(MULTIVERSO_TEST_SRC test_allreduce.cpp test_array_table.cpp test_kv_table.cpp test_matrix_perf.cpp test_matrix_table.cpp test_net.cpp main.cpp)

SET(CMAKE_CXX_COMPILER mpicxx)

LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH})


MESSAGE(${MPI_LIBRARIES})
MESSAGE(${MPI_CXX_LIBRARIES})

ENABLE_TESTING()

ADD_EXECUTABLE(multiverso.test ${MULTIVERSO_TEST_SRC})

if(USE_HDFS)
	TARGET_LINK_LIBRARIES(multiverso.test multiverso ${MPI_CXX_LIBRARIES} jvm hdfs)
else()
	TARGET_LINK_LIBRARIES(multiverso.test multiverso ${MPI_CXX_LIBRARIES})
endif(USE_HDFS)

SET_PROPERTY(TARGET multiverso.test PROPERTY CXX_STANDARD 11)

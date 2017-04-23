include_directories(${MPI_CXX_INCLUDE_PATH})

if (NOT USE_ZMQ)
ADD_DEFINITIONS(-DMULTIVERSO_USE_MPI)
else()
ADD_DEFINITIONS(-DMULTIVERSO_USE_ZMQ)
endif()

if (NOT USE_ZMQ)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        message("OpenMP found")
    	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

set(MULTIVERSO_SRC actor.cpp communicator.cpp controller.cpp dashboard.cpp multiverso.cpp net.cpp node.cpp server.cpp table.cpp table/array_table.cpp table/matrix_table.cpp table/sparse_matrix_table.cpp table/matrix.cpp timer.cpp  updater/updater.cpp util/configure.cpp io/hdfs_stream.cpp io/io.cpp io/local_stream.cpp util/log.cpp util/net_util.cpp worker.cpp zoo.cpp c_api.cpp util/allocator.cpp table_factory.cpp blob.cpp)

add_library(multiverso SHARED ${MULTIVERSO_SRC})
#add_library(imultiverso ${MULTIVERSO_SRC})
if (NOT USE_ZMQ)
    target_link_libraries(multiverso ${MPI_LIBRARY})
else()
    target_link_libraries(multiverso zmq)
endif()

install (TARGETS multiverso DESTINATION lib)
if (UNIX)
    install(CODE "execute_process(COMMAND ldconfig)")  # run ldconfig. Otherwise ld.so.cache won't be created.
endif()

#ifndef MULTIVERSO_NET_ALLREDUCE_ENGINE_H_
#define MULTIVERSO_NET_ALLREDUCE_ENGINE_H_

#include <vector>

#include "multiverso/net.h"

namespace multiverso {

/*! \brief Reduce function */
typedef void (ReduceFunction)(const char *src, char *dst, int len);

/*! \brief The network structure for all gather */
class BruckMap {
public:
  /*! \brief The communication times for one all gather operation */
  int k;
  /*! \brief in_ranks[i] means the incoming rank on i-th communication */
  std::vector<int> in_ranks;
  /*! \brief out_ranks[i] means the out rank on i-th communication */
  std::vector<int> out_ranks;
  BruckMap();
  BruckMap(int n);
  /*!
  * \brief Create the object of bruck map
  * \param rank rank of this machine
  * \param num_machines The total number of machines
  * \return The object of bruck map
  */
  static BruckMap Construct(int rank, int num_machines);
};

/*!
* \brief node type on recursive halving algorithm
* When number of machines is not power of 2, need group matches into power of 2 group.
* And we can let each group has at most 2 machines.
* if the group only has 1 machine. this machine is the normal node
* if the grou has 2 machines, this group will have two type of nodes, one is the leader.
* leader will represent this group and communication with others.
*/
enum RecursiveHalvingNodeType {
  Normal, //normal node, 1 group only have 1 machine 
  GroupLeader, //leader of group when number of machines in this group is 2.
  Other// non-leader machines in group
};

/*! \brief Network structure for recursive halving algorithm */
class RecursiveHalvingMap {
public:
  /*! \brief Communication times for one recursive halving algorithm  */
  int k;
  /*! \brief Node type */
  RecursiveHalvingNodeType type;
  /*! \brief Neighbor, only used for non-normal node*/
  int neighbor;
  /*! \brief ranks[i] means the machines that will communicate with on i-th communication*/
  std::vector<int> ranks;
  /*! \brief  send_block_start[i] means send block start index at i-th communication*/
  std::vector<int> send_block_start;
  /*! \brief  send_block_start[i] means send block size at i-th communication*/
  std::vector<int> send_block_len;
  /*! \brief  send_block_start[i] means recv block start index at i-th communication*/
  std::vector<int> recv_block_start;
  /*! \brief  send_block_start[i] means recv block size  at i-th communication*/
  std::vector<int> recv_block_len;

  RecursiveHalvingMap();
  RecursiveHalvingMap(RecursiveHalvingNodeType _type, int n);

  /*!
  * \brief Create the object of recursive halving map
  * \param rank rank of this machine
  * \param num_machines The total number of machines
  * \return The object of recursive halving map
  */
  static RecursiveHalvingMap Construct(int rank, int num_machines);
};

/*! \brief A class that contains some collective communication algorithm */
class AllreduceEngine {
public:

  AllreduceEngine();

  /*!
  * \brief Initial
  * \param linkers, the low-level communication methods
  */
  void Init(const NetInterface* linkers);

  ~AllreduceEngine();
  /*! \brief Get rank of this machine */
  inline int rank();
  /*! \brief Get total number of machines */
  inline int num_machines();
  
  /*!
  * \brief Perform all reduce. if data size is small, will call AllreduceByAllGather, else with call ReduceScatter followed allgather
  * \param input Input data
  * \param input_size The size of input data
  * \param type_size The size of one object in the reduce function
  * \param output Output result
  * \param reducer Reduce function
  */
  void Allreduce(char* input, int input_size, int type_size, char* output, ReduceFunction reducer);
  
  /*!
  * \brief Perform all reduce, use all gather. When data is small, can use this to reduce communication times
  * \param input Input data
  * \param input_size The size of input data
  * \param type_size The size of one object in the reduce function
  * \param output Output result
  * \param reducer Reduce function
  */
  void AllreduceByAllGather(char* input, int input_size, int type_size, char* output, ReduceFunction reducer);

  /*!
  * \brief Perform all gather, use bruck algorithm. Communication times is O(log(n)), and communication cost is O(send_size * number_machine)
  * if all machine have same input size, can call this function
  * \param input Input data
  * \param send_size The size of input data
  * \param output Output result
  */
  void Allgather(char* input, int send_size, char* output);
  
  /*!
  * \brief Perform all gather, use bruck algorithm. Communication times is O(log(n)), and communication cost is O(all_size)
  * if all machine have different input size, can call this function
  * \param input Input data
  * \param all_size The size of input data
  * \param block_start The block start for different machines
  * \param block_len The block size for different machines
  * \param output Output result
  */
  void Allgather(char* input, int all_size, int* block_start, int* block_len, char* output);
 
  /*!
  * \brief Perform reduce scatter, use recursive halving algorithm. Communication times is O(log(n)), and communication cost is O(input_size)
  * \param input Input data
  * \param input_size The size of input data
  * \param type_size The size of one object in the reduce function
  * \param block_start The block start for different machines
  * \param block_len The block size for different machines
  * \param output Output result
  * \param reducer Reduce function
  */
  void ReduceScatter(char* input, int input_size, int type_size, int* block_start, int* block_len, char* output, ReduceFunction reducer);

private:
  /*! \brief Number of all machines */
  int num_machines_;
  /*! \brief Rank of local machine */
  int rank_;
  /*! \brief The network interface, provide send/recv functions  */
  const NetInterface* linkers_;
  /*! \brief Bruck map for all gather algorithm*/
  BruckMap bruck_map_;
  /*! \brief Recursive halving map for reduce scatter */
  RecursiveHalvingMap recursive_halving_map_;
  /*! \brief Buffer to store block start index */
  int* block_start_;
  /*! \brief Buffer to store block size */
  int* block_len_;
  /*! \brief Buffer  */
  char* buffer_;
  /*! \brief Size of buffer_ */
  int buffer_size_;
};

inline int AllreduceEngine::rank() {
  return rank_;
}

inline int AllreduceEngine::num_machines() {
  return num_machines_;
}

}


#endif //MULTIVERSO_NET_ALLREDUCE_ENGINE_H_
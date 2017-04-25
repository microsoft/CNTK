#include <vector>

#include "multiverso/net/allreduce_engine.h"

namespace multiverso {

BruckMap::BruckMap() {
  k = 0;
}

BruckMap::BruckMap(int n) {
  k = n;
  // default set to -1
  for (int i = 0; i < n; ++i) {
    in_ranks.push_back(-1);
    out_ranks.push_back(-1);
  }
}

BruckMap BruckMap::Construct(int rank, int num_machines) {
  // distance at k-th communication, distance[k] = 2^k
  std::vector<int> distance;
  int k = 0;
  for (k = 0; (1 << k) < num_machines; k++) {
    distance.push_back(1 << k);
  }
  BruckMap bruckMap(k);
  for (int j = 0; j < k; ++j) {
    // set incoming rank at k-th communication
    const int in_rank = (rank + distance[j]) % num_machines;
    bruckMap.in_ranks[j] = in_rank;
    // set outgoing rank at k-th communication
    const int out_rank = (rank - distance[j] + num_machines) % num_machines;
    bruckMap.out_ranks[j] = out_rank;
  }
  return bruckMap;
}


RecursiveHalvingMap::RecursiveHalvingMap() {
  k = 0;
}
RecursiveHalvingMap::RecursiveHalvingMap(RecursiveHalvingNodeType _type, int n) {
  type = _type;
  k = n;
  if (type != RecursiveHalvingNodeType::Other) {
    for (int i = 0; i < n; ++i) {
      // default set as -1
      ranks.push_back(-1);
      send_block_start.push_back(-1);
      send_block_len.push_back(-1);
      recv_block_start.push_back(-1);
      recv_block_len.push_back(-1);
    }
  }
}

RecursiveHalvingMap RecursiveHalvingMap::Construct(int rank, int num_machines) {
  // construct all recursive halving map for all machines
  int k = 0;
  while ((1 << k) <= num_machines) { ++k; }
  // let 1 << k <= num_machines
  --k;
  // distance of each communication
  std::vector<int> distance;
  for (int i = 0; i < k; ++i) {
    distance.push_back(1 << (k - 1 - i));
  }

  if ((1 << k) == num_machines) {
    RecursiveHalvingMap rec_map(RecursiveHalvingNodeType::Normal, k);
    // if num_machines = 2^k, don't need to group machines
    for (int i = 0; i < k; ++i) {
      // communication direction, %2 == 0 is positive
      const int dir = ((rank / distance[i]) % 2 == 0) ? 1 : -1;
      // neighbor at k-th communication
      const int next_node_idx = rank + dir * distance[i];
      rec_map.ranks[i] = next_node_idx;
      // receive data block at k-th communication
      const int recv_block_start = rank / distance[i];
      rec_map.recv_block_start[i] = recv_block_start * distance[i];
      rec_map.recv_block_len[i] = distance[i];
      // send data block at k-th communication
      const int send_block_start = next_node_idx / distance[i];
      rec_map.send_block_start[i] = send_block_start * distance[i];
      rec_map.send_block_len[i] = distance[i];
    }
    return rec_map;
  }
  else {
    // if num_machines != 2^k, need to group machines

    int lower_power_of_2 = 1 << k;

    int rest = num_machines - lower_power_of_2;

    std::vector<RecursiveHalvingNodeType> node_type;
    for (int i = 0; i < num_machines; ++i) {
      node_type.push_back(RecursiveHalvingNodeType::Normal);
    }
    // group, two machine in one group, total "rest" groups will have 2 machines.
    for (int i = 0; i < rest; ++i) {
      int right = num_machines - i * 2 - 1;
      int left = num_machines - i * 2 - 2;
      // let left machine as group leader
      node_type[left] = RecursiveHalvingNodeType::GroupLeader;
      node_type[right] = RecursiveHalvingNodeType::Other;
    }
    int group_cnt = 0;
    // cache block information for groups, group with 2 machines will have double block size
    std::vector<int> group_block_start(lower_power_of_2);
    std::vector<int> group_block_len(lower_power_of_2, 0);
    // convert from group to node leader
    std::vector<int> group_to_node(lower_power_of_2);
    // convert from node to group
    std::vector<int> node_to_group(num_machines);

    for (int i = 0; i < num_machines; ++i) {
      // meet new group
      if (node_type[i] == RecursiveHalvingNodeType::Normal || node_type[i] == RecursiveHalvingNodeType::GroupLeader) {
        group_to_node[group_cnt++] = i;
      }
      node_to_group[i] = group_cnt - 1;
      // add block len for this group
      group_block_len[group_cnt - 1]++;
    }
    // calculate the group block start
    group_block_start[0] = 0;
    for (int i = 1; i < lower_power_of_2; ++i) {
      group_block_start[i] = group_block_start[i - 1] + group_block_len[i - 1];
    }

    RecursiveHalvingMap rec_map(node_type[rank], k);
    if (node_type[rank] == RecursiveHalvingNodeType::Other) {
      rec_map.neighbor = rank - 1;
      // not need to construct
      return rec_map;
    }
    if (node_type[rank] == RecursiveHalvingNodeType::GroupLeader) {
      rec_map.neighbor = rank + 1;
    }
    const int cur_group_idx = node_to_group[rank];
    for (int i = 0; i < k; ++i) {
      const int dir = ((cur_group_idx / distance[i]) % 2 == 0) ? 1 : -1;
      const int next_node_idx = group_to_node[(cur_group_idx + dir * distance[i])];
      rec_map.ranks[i] = next_node_idx;
      // get receive block informations
      const int recv_block_start = cur_group_idx / distance[i];
      rec_map.recv_block_start[i] = group_block_start[recv_block_start * distance[i]];
      int recv_block_len = 0;
      // accumulate block len
      for (int j = 0; j < distance[i]; ++j) {
        recv_block_len += group_block_len[recv_block_start * distance[i] + j];
      }
      rec_map.recv_block_len[i] = recv_block_len;
      // get send block informations
      const int send_block_start = (cur_group_idx + dir * distance[i]) / distance[i];
      rec_map.send_block_start[i] = group_block_start[send_block_start * distance[i]];
      int send_block_len = 0;
      // accumulate block len
      for (int j = 0; j < distance[i]; ++j) {
        send_block_len += group_block_len[send_block_start * distance[i] + j];
      }
      rec_map.send_block_len[i] = send_block_len;
    }
    return rec_map;
  }
}

}

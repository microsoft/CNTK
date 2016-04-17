#ifdef MULTIVERSO_USE_ZMQ
#include <vector>

#include "allreduce_engine.h"

namespace multiverso {


BruckMap::BruckMap() {
  k = 0;
}

BruckMap::BruckMap(int n) {
  k = n;
  for (int i = 0; i < n; ++i) {
    in_ranks.push_back(-1);
    out_ranks.push_back(-1);
  }
}

BruckMap BruckMap::Construct(int rank, int num_machines) {
  int* dist = new int[num_machines];
  int k = 0;
  for (k = 0; (1 << k) < num_machines; k++) {
    dist[k] = 1 << k;
  }
  BruckMap bruckMap(k);
  for (int j = 0; j < k; ++j) {
    int ni = (rank + dist[j]) % num_machines;
    bruckMap.in_ranks[j] = ni;
    ni = (rank - dist[j] + num_machines) % num_machines;
    bruckMap.out_ranks[j] = ni;
  }
  delete[] dist;
  return bruckMap;
}


RecursiveHalvingMap::RecursiveHalvingMap() {
  k = 0;
}
RecursiveHalvingMap::RecursiveHalvingMap(RecursiveHalvingNodeType _type, int n) {
  type = _type;
  if (type != RecursiveHalvingNodeType::SendNeighbor) {
    k = n;
    for (int i = 0; i < n; ++i) {
      ranks.push_back(-1);
      send_block_start.push_back(-1);
      send_block_len.push_back(-1);
      recv_block_start.push_back(-1);
      recv_block_len.push_back(-1);
    }
  }
}
RecursiveHalvingMap RecursiveHalvingMap::Construct(int rank, int num_machines) {
  std::vector<RecursiveHalvingMap> rec_maps;
  for (int i = 0; i < num_machines; ++i) {
    rec_maps.push_back(RecursiveHalvingMap());
  }
  int* distance = new int[num_machines];
  RecursiveHalvingNodeType* node_type = new RecursiveHalvingNodeType[num_machines];
  int k = 0;
  for (k = 0; (1 << k) < num_machines; k++) {
    distance[k] = 1 << k;
  }
  if ((1 << k) == num_machines) {
    for (int i = 0; i < k; ++i) {
      distance[i] = 1 << (k - 1 - i);
    }
    for (int i = 0; i < num_machines; ++i) {
      rec_maps[i] = RecursiveHalvingMap(RecursiveHalvingNodeType::Normal, k);
      for (int j = 0; j < k; ++j) {
        int dir = ((i / distance[j]) % 2 == 0) ? 1 : -1;
        int ni = i + dir * distance[j];
        rec_maps[i].ranks[j] = ni;
        int t = i / distance[j];
        rec_maps[i].recv_block_start[j] = t * distance[j];
        rec_maps[i].recv_block_len[j] = distance[j];
      }
    }
  }
  else {
    k--;
    int lower_power_of_2 = 1 << k;

    int rest = num_machines - (1 << k);
    for (int i = 0; i < num_machines; ++i) {
      node_type[i] = RecursiveHalvingNodeType::Normal;
    }
    for (int i = 0; i < rest; ++i) {
      int r = num_machines - i * 2 - 1;
      int l = num_machines - i * 2 - 2;
      node_type[l] = RecursiveHalvingNodeType::ReciveNeighbor;
      node_type[r] = RecursiveHalvingNodeType::SendNeighbor;
    }
    for (int i = 0; i < k; ++i) {
      distance[i] = 1 << (k - 1 - i);
    }

    int group_idx = 0;
    int* map_len = new int[lower_power_of_2];
    int* map_start = new int[lower_power_of_2];
    int* group_2_node = new int[lower_power_of_2];
    int* node_to_group = new int[num_machines];
    for (int i = 0; i < lower_power_of_2; ++i) {
      map_len[i] = 0;
    }
    for (int i = 0; i < num_machines; ++i) {
      if (node_type[i] == RecursiveHalvingNodeType::Normal || node_type[i] == RecursiveHalvingNodeType::ReciveNeighbor) {
        group_2_node[group_idx++] = i;

      }
      map_len[group_idx - 1]++;
      node_to_group[i] = group_idx - 1;
    }
    map_start[0] = 0;
    for (int i = 1; i < lower_power_of_2; ++i) {
      map_start[i] = map_start[i - 1] + map_len[i - 1];
    }

    for (int i = 0; i < num_machines; ++i) {

      if (node_type[i] == RecursiveHalvingNodeType::SendNeighbor) {
        rec_maps[i] = RecursiveHalvingMap(RecursiveHalvingNodeType::SendNeighbor, k);
        rec_maps[i].neighbor = i - 1;
        continue;
      }
      rec_maps[i] = RecursiveHalvingMap(node_type[i], k);
      if (node_type[i] == RecursiveHalvingNodeType::ReciveNeighbor) {
        rec_maps[i].neighbor = i + 1;
      }
      for (int j = 0; j < k; ++j) {
        int dir = ((node_to_group[i] / distance[j]) % 2 == 0) ? 1 : -1;
        group_idx = group_2_node[(node_to_group[i] + dir * distance[j])];
        rec_maps[i].ranks[j] = group_idx;
        int t = node_to_group[i] / distance[j];
        rec_maps[i].recv_block_start[j] = map_start[t * distance[j]];
        int tl = 0;
        for (int tmp_i = 0; tmp_i < distance[j]; ++tmp_i) {
          tl += map_len[t * distance[j] + tmp_i];
        }
        rec_maps[i].recv_block_len[j] = tl;
      }
    }
  }
  for (int i = 0; i < num_machines; ++i) {
    if (rec_maps[i].type != RecursiveHalvingNodeType::SendNeighbor) {
      for (int j = 0; j < k; ++j) {
        int target = rec_maps[i].ranks[j];
        rec_maps[i].send_block_start[j] = rec_maps[target].recv_block_start[j];
        rec_maps[i].send_block_len[j] = rec_maps[target].recv_block_len[j];
      }
    }
  }
  return rec_maps[rank];
}

}

#endif
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// pyreallocate.cpp : allocate table algorithm, time complexity = O(V * sqrt(V) * log(sqrt(V)))
//
//

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <utility>
#include <queue>
#include <cassert>
#include <string>
#include <ios>
#include <ctime>
#include <iomanip>

#define loss first
#define id second

typedef std::vector< std::pair<double, int> > DIVector;
typedef std::vector< int > IVector;

int g_vocab_size;
int g_vocab_sqrt;

struct SortNode {
    int sort_id, word_id;
    double value;

    SortNode(int sort_id, int word_id, double value) : sort_id(sort_id), word_id(word_id), value(value) {
    }

    bool operator < (const SortNode &next) const {
        return value < next.value;
    }
};

struct InsertNode {
    /*
    * The Word Node,
    * It include the sorted loss vector of row and col,
    * and the next great position row or col for the curr word.
    * */
    DIVector prob_row;
    DIVector prob_col;
    int word_id;
    int row_id;
    int col_id;
    double row_loss_sum;
    double col_loss_sum;

    InsertNode(DIVector prob_row, DIVector prob_col, int word_id) :
        prob_row(prob_row), prob_col(prob_col), word_id(word_id), row_id(0), col_id(0) {
        for (int i = 0; i < g_vocab_sqrt; i++) {
            row_loss_sum += prob_row[i].loss;
            col_loss_sum += prob_col[i].loss;
        }
    }

    SortNode next_row() {
        int sort_id = prob_row[row_id].id;
        row_loss_sum -= prob_row[row_id].loss;
        row_id++;
        double value = row_id == g_vocab_sqrt - 1 ? 0 : row_loss_sum / (g_vocab_sqrt - row_id - 1);
        return SortNode(sort_id, word_id, value);
    }

    SortNode next_col() {
        int sort_id = prob_col[col_id].id;
        col_loss_sum -= prob_col[col_id].loss;
        col_id++;
        double value = col_id == g_vocab_sqrt - 1 ? 0 : col_loss_sum / (g_vocab_sqrt - col_id - 1);
        return SortNode(sort_id, word_id, value);
    }
};


std::vector < InsertNode > g_prob_table;
std::vector < IVector > g_table;
std::priority_queue < SortNode > search_Queue;
std::vector < std::string > index_word;

/*
* read vocab from file
* */
void get_word_location(std::string word_path) {
    std::fstream input_file(word_path, std::ios::in);
    std::string word;
    while (input_file >> word) {
        index_word.push_back(word);
    }
    input_file.close();
}

/*
* The function of saving the reallocated word table
* */
void save_allocate_word_location(std::string save_path) {
    std::fstream output_file(save_path, std::ios::out);
    std::fstream output_string_file(save_path + ".string", std::ios::out);
    for (int i = 0; i < g_vocab_sqrt; i++) {
        for (int j = 0; j < g_vocab_sqrt; j++) {
            if (g_table[i][j] == -1) {
                output_string_file << "<null>" << " ";
            }
            else {
                output_string_file << index_word[g_table[i][j]] << " ";
            }
            output_file << g_table[i][j] << " ";
        }
        output_file << "\n";
        output_string_file << "\n";
    }
    output_file.close();
    output_string_file.close();
}

/*
* content_row        : the loss vector of row
* content_col        : the loss vector of col
* vocabsize          : the size of vocabulary
* vocabbase          : the sqrt of vocabuary size
* save_location_path : the path of next word location, the reallocated table will be saved
*                      into this path
* word_path          : the path of word table
* */
void allocate(double *content_row, double *content_col,
    int vocabsize, int vocabbase,
    char* save_location_path, char* word_path) {
    clock_t start = clock();
    std::cout << "Wait for initial ... \n";

    // initial
    std::vector<InsertNode>().swap(g_prob_table);
    std::vector<IVector>().swap(g_table);
    std::priority_queue<SortNode>().swap(search_Queue);
    std::vector<std::string>().swap(index_word);

    g_vocab_size = vocabsize;
    g_vocab_sqrt = vocabbase;
    int freq = g_vocab_size / 20;
    // sort every node's position probability by loss
    for (int i = 0; i < g_vocab_size; i++) {
        DIVector current_row, current_col;
        for (int j = 0; j < g_vocab_sqrt; j++) {
            current_row.push_back(std::make_pair(content_row[i * g_vocab_sqrt + j], j));
            current_col.push_back(std::make_pair(content_col[i * g_vocab_sqrt + j], j));
        }
        sort(current_row.begin(), current_row.end());
        sort(current_col.begin(), current_col.end());
        g_prob_table.push_back(InsertNode(current_row, current_col, i));
        if (i % freq == 0) {
            std::cout << "\t\t\tFinish " << std::setw(8) << i << " / " << std::setw(8) << g_vocab_size << " Line\n";
        }
    }
    for (int i = 0; i < g_vocab_sqrt; i++) {
        g_table.push_back(IVector());
    }

    std::cout << "Ready ... \n";
    std::cout << "Start to assign row for every word\n";
    for (int i = 0; i < g_vocab_size; i++) {
        SortNode row_node = g_prob_table[i].next_row();
        search_Queue.push(row_node);
    }

    while (!search_Queue.empty()) {
        SortNode top_node = search_Queue.top();
        search_Queue.pop();
        int word_id = top_node.word_id;
        int row_id = top_node.sort_id;
        if (static_cast<int>(g_table[row_id].size()) == g_vocab_sqrt) {
            search_Queue.push(g_prob_table[word_id].next_row());
        }
        else {
            g_table[row_id].push_back(word_id);
        }
    }

    std::cout << "Finish assigning row\n";
    std::cout << "Start to assign col for every word\n";
    std::cout << "Finish assigning col\n";

    for (int i = 0; i < g_vocab_sqrt; i++) {
        for (auto &word_id : g_table[i]) {
            SortNode col_node = g_prob_table[word_id].next_col();
            search_Queue.push(col_node);
            word_id = -1;
        }
        for (int j = static_cast<int>(g_table[i].size()); j < g_vocab_sqrt; j++) {
            g_table[i].push_back(-1);
        }
        while (!search_Queue.empty()) {
            SortNode top_node = search_Queue.top();
            search_Queue.pop();
            int word_id = top_node.word_id;
            int col_id = top_node.sort_id;
            if (g_table[i][col_id] == -1) {
                g_table[i][col_id] = word_id;
            }
            else {
                search_Queue.push(g_prob_table[word_id].next_col());
            }
        }
    }
    get_word_location(word_path);
    save_allocate_word_location(save_location_path);
    clock_t end = clock();
    double cost_time = static_cast<double>((end - start) / CLOCKS_PER_SEC);
    std::cout << "Reallocate word location cost " << cost_time << " seconds\n";
}

extern "C" {
// Under Visual Studio environ
#ifdef _MT
    __declspec(dllexport) void allocate_table(double *content_row, double *content_col,
        int vocabsize, int vocabbase,
        char* save_location_path, char* word_path) {
        allocate(content_row, content_col, vocabsize, vocabbase, save_location_path, word_path);
    }
#else
    void allocate_table(double *content_row, double *content_col,
        int vocabsize, int vocabbase,
        char* save_location_path, char* word_path) {
        allocate(content_row, content_col, vocabsize, vocabbase, save_location_path, word_path);
    }
#endif
}

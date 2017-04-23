#include "util.h"

void Option::ParseArgs(int argc, char* argv[])
{
    for (int i = 1; i < argc; i += 2)
    {
        if (strcmp(argv[i], "-train_file") == 0) train_file = argv[i + 1];
        if (strcmp(argv[i], "-save_vocab_file") == 0) save_vocab_file = argv[i + 1];
        if (strcmp(argv[i], "-min_count") == 0) min_count = atoi(argv[i + 1]);
    }
}
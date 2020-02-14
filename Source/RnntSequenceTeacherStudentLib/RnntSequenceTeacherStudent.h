#include <iostream>
#include <string>
#include <assert>
#include "basics.h"
#include "ComputationNetwork.h"

using namespace std;

template <typename ElemType>
class RnntSequenceTSParams
{
    string _combination_level;  // frame or sequence
    string _combination_method; // sum or product
    int _num_teachers;
    vector<ElemType> _teacher_combination_weights;

public:
    RnntSequenceTSParams(void)
    {
        _combination_level = "frame";
        _combination_method = "sum";
        _num_teachers = 1;
        _teacher_combination_weights.resize(1);
        _teacher_combination_weights[0] = 1.0;
    }

    RnntSequenceTSParams(string combination_level, string combination_method, vector<ElemType> teacher_combination_weights)
    {
        _combination_level = combination_level;
        _combination_method = combination_method;
        _num_teachers = teacher_combination_weights.size();

        if (_num_teachers <= 0)
        {
            InvalidArgument("Must supply one or more teachers")
        }

        _teacher_combination_weights.resize(_num_teachers);
        for (int i = 0; i < _num_teachers; i++)
        {
            _teacher_combination_weights[i] = teacher_combination_weights[i];
        }
    }
};

void ForwardSingleTeacherSingleHypothesis();

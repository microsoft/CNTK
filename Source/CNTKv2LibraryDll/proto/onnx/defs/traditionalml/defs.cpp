#include "proto/onnx/core/constants.h"
#include "proto/onnx/core/op.h"

namespace ONNXIR {

    REGISTER_OPERATOR_SCHEMA(ArrayFeatureExtractor)
        .SetDomain(c_mlDomain)
        .Input("X", "Data to be selected from", "T1")
        .Input("Y", "Data to be selected from", "T2")
        .Output("Z", "Selected data as an array", "T1")
        .Description(R"DOC(
            Select a subset of the data from input1 based on the indices provided in input2.
            )DOC")
        .TypeConstraint("T1", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)", "tensor(string)" }, " allowed types.")
        .TypeConstraint("T2", { "tensor(int64)" }, " Index value types .");


    REGISTER_OPERATOR_SCHEMA(Binarizer)
        .SetDomain(c_mlDomain)
        .Input("X", "Data to be binarized", "T")
        .Output("Y", "Binarized output data", "T")
        .Description(R"DOC(
            Makes values 1 or 0 based on a single threshold.
            )DOC")
        .TypeConstraint("T", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .Attr("threshold", "Values greater than this are set to 1, else set to 0", AttrType::AttributeProto_AttributeType_FLOAT);

    REGISTER_OPERATOR_SCHEMA(CastMap)
        .SetDomain(c_mlDomain)
        .Input("X", "The input values", "T1")
        .Output("Y", "The output values", "T2")
        .Description(R"DOC(
            Casts the input into an output tensor.
            )DOC")
        .TypeConstraint("T1", { "map(int64, string)", "map(int64, float)" }, " allowed input types.")
        .TypeConstraint("T2", { "tensor(string)","tensor(float)","tensor(int64)" }, " allowed output types.")
        .Attr("cast_to", "what type of tensor to cast the input to, enum 'TO_FLOAT','TO_STRING','TO_INT64'", AttrType::AttributeProto_AttributeType_STRING)
        .Attr("map_form", "if casting from a map with int64 keys, should we pad spaces between the keys or pack them, enum 'PACK, 'SPARSE'", AttrType::AttributeProto_AttributeType_STRING)
        .Attr("max_map", "if casting from a sparse map, what is the max key in the map", AttrType::AttributeProto_AttributeType_INT);

    REGISTER_OPERATOR_SCHEMA(CategoryMapper)
        .SetDomain(c_mlDomain)
        .Input("X", "Input data", "T1")
        .Output("Y", "Output data, if strings are input, then output is INTS, and vice versa.", "T2")
        .Description(R"DOC(
            Convert strings to INTS and vice versa.
            Takes in a map to use for the conversion.
            The index position in the strings and ints repeated inputs
             is used to do the mapping.
            Each instantiated operator converts either ints to strings or strings to ints.
            This behavior is triggered based on which default value is set.
            If the string default value is set, it will convert ints to strings.
            If the int default value is set, it will convert strings to ints.
            )DOC")
        .TypeConstraint("T1", { "tensor(string)", "tensor(int64)" }, " allowed types.")
        .TypeConstraint("T2", { "tensor(string)", "tensor(int64)" }, " allowed types.")
        .Attr("cats_strings", "strings part of the input map, must be same size as the ints", AttrType::AttributeProto_AttributeType_STRINGS)
        .Attr("cats_int64s", "ints part of the input map, must be same size and the strings", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("default_string", "string value to use if the int is not in the map", AttrType::AttributeProto_AttributeType_STRING)
        .Attr("default_int64", "int value to use if the string is not in the map", AttrType::AttributeProto_AttributeType_INT);


    REGISTER_OPERATOR_SCHEMA(DictVectorizer)
        .SetDomain(c_mlDomain)
        .Input("X", "The input dictionary", "T1")
        .Output("Y", "The tensor", "T2")
        .Description(R"DOC(
            Uses an index mapping to convert a dictionary to an array.
            The output array will be equal in length to the index mapping vector parameter.
            All keys in the input dictionary must be present in the index mapping vector.
            For each item in the input dictionary, insert its value in the ouput array.
            The position of the insertion is determined by the position of the item's key
            in the index mapping. Any keys not present in the input dictionary, will be
            zero in the output array.  Use either string_vocabulary or int64_vocabulary, not both.
            For example: if the ``string_vocabulary`` parameter is set to ``["a", "c", "b", "z"]``,
            then an input of ``{"a": 4, "c": 8}`` will produce an output of ``[4, 8, 0, 0]``.
            )DOC")
        .TypeConstraint("T1", { "map(string, int64)", "map(int64, string)", "map(int64, float)", "map(int64, double)", "map(string, float)", "map(string, double)"}, " allowed types.")
        .TypeConstraint("T2", { "tensor(int64)", "tensor(float)", "tensor(double)", "tensor(string)" }, " allowed types.")
        .Attr("string_vocabulary", "The vocabulary vector of strings", AttrType::AttributeProto_AttributeType_STRINGS)
        .Attr("int64_vocabulary", "The vocabulary vector of int64s", AttrType::AttributeProto_AttributeType_INTS);


    REGISTER_OPERATOR_SCHEMA(Imputer)
        .SetDomain(c_mlDomain)
        .Input("X", "Data to be imputed", "T")
        .Output("Y", "Imputed output data", "T")
        .Description(R"DOC(
            Replace imputs that equal replaceValue/s  with  imputeValue/s.
            All other inputs are copied to the output unchanged.
            This op is used to replace missing values where we know what a missing value looks like.
            )DOC")
        .TypeConstraint("T", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .Attr("imputed_value_floats", "value(s) to change to, can be length 1 or length F if using int type", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("replaced_value_float", "value that needs replacing if using int type", AttrType::AttributeProto_AttributeType_FLOAT)
        .Attr("imputed_value_int64s", "value(s) to change to, can be length 1 or length F if using int type", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("replaced_value_int64", "value that needs replacing if using int type", AttrType::AttributeProto_AttributeType_INT);


    REGISTER_OPERATOR_SCHEMA(FeatureVectorizer)
        .SetDomain(c_mlDomain)
        .Input("X", "ordered input tensors", "T")
        .Output("Y", "flattened output vector.", "T")
        .Description(R"DOC(
            Concatenates a list of input tensors of floats into one tensor.
            The size of each input in the input list is expressed in inputdimensions.
            )DOC")
        .TypeConstraint("T", { "tensor(float)" }, " allowed types.")
        .Attr("inputdimensions", "the size of the inputs in the input list", AttrType::AttributeProto_AttributeType_INTS);


    REGISTER_OPERATOR_SCHEMA(LabelEncoder)
        .SetDomain(c_mlDomain)
        .Input("X", "Data to be encoded", "T1")
        .Output("Y", "Encoded output data", "T2")
        .Description(R"DOC(
            Convert class label to their integral type and vice versa.
            In both cases the operator is instantiated with the list of class strings.
            The integral value of the string is the index position in the list.
            )DOC")
        .TypeConstraint("T1", { "tensor(string)", "tensor(int64)" }, " allowed types.")
        .TypeConstraint("T2", { "tensor(string)", "tensor(int64)" }, " allowed types.")
        .Attr("classes_strings", "List of class label strings to be encoded as INTS", AttrType::AttributeProto_AttributeType_STRINGS)
        .Attr("default_int64", "Default value if not in class list as int64", AttrType::AttributeProto_AttributeType_INT)
        .Attr("default_string", "Default value if not in class list as string", AttrType::AttributeProto_AttributeType_STRING);


    REGISTER_OPERATOR_SCHEMA(LinearClassifier)
        .SetDomain(c_mlDomain)
        .Input("X", "Data to be classified", "T1")
        .Output("Y", "Classification outputs (one class per example", "T2")
        .Output("Z", "Classification outputs (All classes scores per example,N,E", "tensor(float)")
        .Description(R"DOC(
            Linear classifier prediction (choose class)
            )DOC")
        .TypeConstraint("T1", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .TypeConstraint("T2", { "tensor(string)", "tensor(int64)" }, " allowed types.")
        .Attr("coefficients", "weights of the model(s)", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("intercepts", "weights of the intercepts (if used)", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("post_transform", "post eval transform for score, enum 'NONE', 'SOFTMAX', 'LOGISTIC', 'SOFTMAX_ZERO', 'PROBIT'", AttrType::AttributeProto_AttributeType_STRING)
        .Attr("multi_class", "whether to do OvR or multinomial (0=OvR and is default)", AttrType::AttributeProto_AttributeType_INT)
        .Attr("classlabels_strings", "class labels if using string labels, size E", AttrType::AttributeProto_AttributeType_STRINGS)
        .Attr("classlabels_ints", "class labels if using int labels, size E", AttrType::AttributeProto_AttributeType_INTS);


    REGISTER_OPERATOR_SCHEMA(LinearRegressor)
        .SetDomain(c_mlDomain)
        .Input("X", "Data to be regressed", "T")
        .Output("Y", "Regression outputs (one per target, per example", "tensor(float)")
        .Description(R"DOC(
            Generalized linear regression evaluation.
            If targets is set to 1 (default) then univariate regression is performed.
            If targets is set to M then M sets of coefficients must be passed in as a sequence
            and M results will be output for each input n in N.
            Coefficients are of the same length as an n, and coefficents for each target are contiguous.
           "Intercepts are optional but if provided must match the number of targets.
            )DOC")
        .TypeConstraint("T", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .Attr("coefficients", "weights of the model(s)", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("intercepts", "weights of the intercepts (if used)", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("targets", "total number of regression targets (default is 1)", AttrType::AttributeProto_AttributeType_INT)
        .Attr("post_transform", "post eval transform for score, enum 'NONE', 'SOFTMAX', 'LOGISTIC', 'SOFTMAX_ZERO', 'PROBIT'", AttrType::AttributeProto_AttributeType_STRING);


    REGISTER_OPERATOR_SCHEMA(Normalizer)
        .SetDomain(c_mlDomain)
        .Input("X", "Data to be encoded", "T")
        .Output("Y", "encoded output data", "tensor(float)")
        .Description(R"DOC(
            Normalize the input.  There are three normalization modes,
            which have the corresponding formulas:
            Max .. math::     max(x_i)
            L1  .. math::  z = ||x||_1 = \sum_{i=1}^{n} |x_i|
            L2  .. math::  z = ||x||_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
            )DOC")
        .TypeConstraint("T", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .Attr("norm", "enum 'MAX', 'L1', 'L2'", AttrType::AttributeProto_AttributeType_STRING);


    REGISTER_OPERATOR_SCHEMA(OneHotEncoder)
        .SetDomain(c_mlDomain)
        .Input("X", "Data to be encoded", "T")
        .Output("Y", "encoded output data", "tensor(float)")
        .Description(R"DOC(
            Replace the inputs with an array of ones and zeros, where the only
            one is the zero-based category that was passed in.  The total category count
            will determine the length of the vector. For example if we pass a
            tensor with a single value of 4, and a category count of 8, the
            output will be a tensor with 0,0,0,0,1,0,0,0 .
            This operator assumes every input in X is of the same category set
            (meaning there is only one category count).

            If the input is a tensor of float, int32, or double, the data will be cast
            to int64s and the cats_int64s category list will be used for the lookups.
            )DOC")
        .TypeConstraint("T", { "tensor(string)", "tensor(int64)","tensor(int32)", "tensor(float)","tensor(double)" }, "allowed types.")
        .Attr("cats_int64s", "list of cateogries, ints", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("cats_strings", "list of cateogries, strings", AttrType::AttributeProto_AttributeType_STRINGS)
        .Attr("zeros", "if true and category is not present, will return all zeros, if false and missing category, operator will return false", AttrType::AttributeProto_AttributeType_INT);


    // Input: X, output: Y
    REGISTER_OPERATOR_SCHEMA(Scaler)
        .SetDomain(c_mlDomain)
        .Input("X", "Data to be scaled", "T")
        .Output("Y", "Scaled output data", "tensor(float)")
        .Description(R"DOC(
            Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
            )DOC")
        .TypeConstraint("T", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .Attr("scale", "second, multiply by this", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("offset", "first, offset by thisfirst, offset by this, can be one value or a separate value for each feature", AttrType::AttributeProto_AttributeType_FLOATS);

    REGISTER_OPERATOR_SCHEMA(SVMClassifier)
        .SetDomain(c_mlDomain)
        .Input("X", "Data to be classified", "T1")
        .Output("Y", "Classification outputs, one class per example", "T2")
        .Output("Z", "Classification outputs, All classes scores per example,N,E*(E-1)/2 if dual scores, or E if probabilities are used.", "tensor(float)")
        .Description(R"DOC(
            SVM classifier prediction (two class or multiclass).
            Will output probabilities in Z if prob_A and prob_B are filled in.
            )DOC")
        .TypeConstraint("T1", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .TypeConstraint("T2", { "tensor(string)", "tensor(int64)" }, " allowed types.")
        .Attr("kernel_type", "enum 'LINEAR', 'POLY', 'RBF', 'SIGMOID', defaults to linear", AttrType::AttributeProto_AttributeType_STRING)
        .Attr("kernel_params", "Tensor of 3 elements containing gamma, coef0, degree in that order.  Zero if unused for the kernel.", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("prob_a", "probability vector a, must be either 0 length or E*(E-1)/2 length", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("prob_b", "probability vector b, must be same length as prob_a", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("vectors_per_class", "", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("support_vectors", "", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("coefficients", "", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("rho", "", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("post_transform", "post eval transform for score, enum 'NONE', 'SOFTMAX', 'LOGISTIC', 'SOFTMAX_ZERO', 'PROBIT'", AttrType::AttributeProto_AttributeType_STRING)
        .Attr("classlabels_strings", "class labels if using string labels", AttrType::AttributeProto_AttributeType_STRINGS)
        .Attr("classlabels_ints", "class labels if using int labels", AttrType::AttributeProto_AttributeType_INTS);


    REGISTER_OPERATOR_SCHEMA(SVMRegressor)
        .SetDomain(c_mlDomain)
        .Input("X", "Input N,F", "T")
        .Output("Y", "All target scores, N,E", "tensor(float)")
        .Description(R"DOC(
            SVM regressor. Also supports oneclass svm.
            )DOC")
        .TypeConstraint("T", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .Attr("kernel_type", "enum 'LINEAR', 'POLY', 'RBF', 'SIGMOID', defaults to linear", AttrType::AttributeProto_AttributeType_STRING)
        .Attr("kernel_params", "Tensor of 3 elements containing gamma, coef0, degree in that order.  Zero if unused for the kernel.", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("post_transform", "post eval transform for score, enum 'NONE', 'SOFTMAX', 'LOGISTIC', 'SOFTMAX_ZERO', 'PROBIT'", AttrType::AttributeProto_AttributeType_STRING)
        .Attr("vectors_per_class", "", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("support_vectors", "", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("n_supports", "number of support vectors", AttrType::AttributeProto_AttributeType_INT)
        .Attr("coefficients", "", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("rho", "", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("one_class", "If this regressor is a oneclass svm set this param to 1, otherwise use 0 (default is zero)", AttrType::AttributeProto_AttributeType_INT);

    REGISTER_OPERATOR_SCHEMA(TreeEnsembleClassifier)
        .SetDomain(c_mlDomain)
        .Input("X", "Data to be classified", "T1")
        .Output("Y", "Classification outputs (one class per example", "T2")
        .Output("Z", "Classification outputs (All classes scores per example,N,E", "tensor(float)")
        .Description(R"DOC(
            Tree Ensemble classifier.  Returns the top class for each input in N.
            All args with nodes_ are fields of a tuple of tree nodes, and
            it is assumed they are the same length, and an index i will decode the
            tuple across these inputs.  Each node id can appear only once
            for each tree id."
            All fields prefixed with class_ are tuples of votes at the leaves.
            A leaf may have multiple votes, where each vote is weighted by
            the associated class_weights index.
            It is expected that either classlabels_strings or classlabels_INTS
            will be passed and the class_ids are an index into this list.
            Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF.
            )DOC")
        .TypeConstraint("T1", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .TypeConstraint("T2", { "tensor(string)", "tensor(int64)" }, " allowed types.")
        .Attr("nodes_treeids", "tree id for this node", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("nodes_nodeids", "node id for this node, node ids may restart at zero for each tree (but not required).", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("nodes_featureids", "feature id for this node", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("nodes_values", "thresholds to do the splitting on for this node.", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("nodes_hitrates", "", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("nodes_modes", "enum of behavior for this node 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'", AttrType::AttributeProto_AttributeType_STRINGS)
        .Attr("nodes_truenodeids", "child node if expression is true", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("nodes_falsenodeids", "child node if expression is false", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("nodes_missing_value_tracks_true", "for each node, decide if the value is missing (nan) then use true branch, this field can be left unset and will assume false for all nodes", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("base_values", "starting values for each class, can be omitted and will be assumed as 0", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("class_treeids", "tree that this node is in", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("class_nodeids", "node id that this weight is for", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("class_ids", "index of the class list that this weight is for", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("class_weights", "the weight for the class in class_id", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("post_transform", "post eval transform for score, enum 'NONE', 'SOFTMAX', 'LOGISTIC', 'SOFTMAX_ZERO', 'PROBIT'", AttrType::AttributeProto_AttributeType_STRING)
        .Attr("classlabels_strings", "class labels if using string labels, size E", AttrType::AttributeProto_AttributeType_STRINGS)
        .Attr("classlabels_int64s", "class labels if using int labels, size E, one of the two class label fields must be used", AttrType::AttributeProto_AttributeType_INTS);


    REGISTER_OPERATOR_SCHEMA(TreeEnsembleRegressor)
        .SetDomain(c_mlDomain)
        .Input("X", "Input N,F", "T")
        .Output("Y", "NxE floats", "tensor(float)")
        .Description(R"DOC(
            Tree Ensemble regressor.  Returns the regressed values for each input in N.
            All args with nodes_ are fields of a tuple of tree nodes, and
            it is assumed they are the same length, and an index i will decode the
            tuple across these inputs.  Each node id can appear only once
            for each tree id.
            All fields prefixed with target_ are tuples of votes at the leaves.
            A leaf may have multiple votes, where each vote is weighted by
            the associated target_weights index.
            All trees must have their node ids start at 0 and increment by 1.
            Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
            )DOC")
        .TypeConstraint("T", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .Attr("nodes_treeids", "tree id for this node", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("nodes_nodeids", "node id for this node, node ids may restart at zero for each tree (but not required).", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("nodes_featureids", "feature id for this node", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("nodes_values", "thresholds to do the splitting on for this node.", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("nodes_hitrates", "", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("nodes_modes", "enum of behavior for this node.  enum 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'", AttrType::AttributeProto_AttributeType_STRINGS)
        .Attr("nodes_truenodeids", "child node if expression is true", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("nodes_falsenodeids", "child node if expression is false", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("nodes_missing_value_tracks_true", "for each node, decide if the value is missing (nan) then use true branch, this field can be left unset and will assume false for all nodes", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("target_treeids", "tree that this node is in", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("target_nodeids", "node id that this weight is for", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("target_ids", "index of the class list that this weight is for", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("target_weights", "the weight for the class in target_id", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("n_targets", "number of regression targets", AttrType::AttributeProto_AttributeType_INT)
        .Attr("post_transform", "post eval transform for score, enum 'NONE', 'SOFTMAX', 'LOGISTIC', 'SOFTMAX_ZERO', 'PROBIT'", AttrType::AttributeProto_AttributeType_STRING)
        .Attr("aggregate_function", "post eval transform for score,  enum 'AVERAGE', 'SUM', 'MIN', 'MAX'", AttrType::AttributeProto_AttributeType_STRING)
        .Attr("base_values", "base values for regression, added to final score, size must be the same as n_outputs or can be left unassigned (assumed 0)", AttrType::AttributeProto_AttributeType_FLOATS);

    REGISTER_OPERATOR_SCHEMA(ZipMap)
        .SetDomain(c_mlDomain)
        .Input("X", "The input values", "tensor(float)")
        .Output("Y", "The output map", "T")
        .Description(R"DOC(
            Makes a map from the input and the attributes.
            Assumes input 0 are the values, and the keys are specified by the attributes.
            Must provide keys in either classlabels_strings or classlabels_int64s (but not both).
            Input 0 may have a batch size larger than 1,
            but each input in the batch must be the size of the keys specified by the attributes.
            The order of the input and attributes determines the key-value mapping.
            )DOC")
        .TypeConstraint("T", { "map(string, float)", "map(int64, float)" }, " allowed types.")
        .Attr("classlabels_strings", "keys if using string keys", AttrType::AttributeProto_AttributeType_STRINGS)
        .Attr("classlabels_int64s", "keys if using int keys", AttrType::AttributeProto_AttributeType_INTS);

}

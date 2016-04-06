import argparse

parser = argparse.ArgumentParser(
    description="UCI to CNTKText format converter",
    epilog=("Quick example - converting MNIST data (see Examples/Image/MNIST):"
            "\n\n\t"
            "--input_file Examples/Image/MNIST/Data/Train-28x28.txt "
            "--features_start 1 "
            "--features_dim 784 "
            "--labels_start 0 "
            "--labels_dim 1 "
            "--num_labels 10 "
            "--output_file Examples/Image/MNIST/Data/Train-28x28_cntk_text.txt"
            "\n\n"
            "For more information please visit "
            "https://github.com/Microsoft/CNTK/wiki/CNTKTextFormat-Reader"),
    formatter_class=argparse.RawTextHelpFormatter)

requiredNamed = parser.add_argument_group('required arguments')

requiredNamed.add_argument("-in", "--input_file",
                           help="input file path", required=True)
requiredNamed.add_argument("-fs", "--features_start", type=int,
                           help="start offset of feature columns", required=True)
requiredNamed.add_argument("-fd", "--features_dim", type=int,
                           help=("dimension of the feature vector "
                                 "(number of feature columns in the input file)"),
                           required=True)

parser.add_argument("-lt", "--label_type", default="Category",
                    help=("Label type (indicates how the label columns should "
                          " be interpreted)"),
                    choices=["Category", "Regression", "None"])
parser.add_argument("-ls", "--labels_start", type=int,
                    help=("dimension of the label vector "
                          "(number of label columns in the input file)"))
parser.add_argument("-nl", "--num_labels", type=int,
                    help="number of possible label values "
                         "(required for categorical labels)")
parser.add_argument("-ld", "--labels_dim", type=int, default=1,
                    help=("dimension of the input label vector "
                          "(number of label columns in the input file, "
                          "default is 1)"))
parser.add_argument("--mapping_file",
                    help=("the path to a file used to map from the label value "
                          "to a numerical label identifier (if omitted, the "
                          "label value is interpreted as a numerical "
                          "identifier)"))
parser.add_argument("-out", "--output_file", help="output file path")

args = parser.parse_args()

# a number of sanity checks
if args.label_type != "None" and args.labels_start is None:
    parser.error("-ls/--label_start is required when label type is not 'None'")

if args.label_type == "Category":
    if args.num_labels is None:
        parser.error("-nl/--num_labels is required when label type is 'Category'")
    if args.labels_dim > 1:
        parser.error("-ld/--labels_dim cannot be greater than 1 "
                     "when label type is 'Category'")

if args.label_type == "Regression":
    if args.num_labels > args.labels_dim:
        parser.error("-nl/--num_labels is optional and "
                     " cannot exceed -ld/--labels_dim "
                     " when label type is 'Regression'")

if args.label_type != 'None':
    if (((args.labels_start <= args.features_start) and
         (args.labels_start + args.labels_dim > args.features_start)) or
            ((args.labels_start > args.features_start) and
             (args.features_start + args.features_dim > args.labels_start))):
        parser.error("Label and feature column ranges must not overlap.")

file_in = args.input_file
file_out = args.output_file

num_labels = args.num_labels
label_map = {}
if args.label_type == "Category":
    if args.mapping_file is not None:
        with open(args.mapping_file, 'r') as f:
            for line in f.read().splitlines():
                label_map[line] = len(label_map)

        num_labels = max(num_labels, len(label_map))
    else:
        label_map = {str(x) : x for x in range(num_labels)}

if not file_out:
    dot = file_in.rfind(".")
    if dot == -1:
        dot = len(file_in)
    file_out = file_in[:dot] + "_cntk_text" + file_in[dot:]

print (" Converting from UCI format\n\t '{}'\n"
       " to CNTK text format\n\t '{}'").format(file_in, file_out)

input_file = open(file_in, 'r')
output_file = open(file_out, 'w')

for line in input_file.readlines():
    values = line.split()

    if args.label_type != 'None':
        max_length = max(args.labels_start + args.labels_dim, 
                         args.features_start + args.features_dim)
        if len(values) < (args.labels_dim + args.features_dim):
            raise RuntimeError(
                ("Too few input columns ({} out of expected {}) ")
                .format(len(values), (args.labels_dim + args.features_dim)))
        elif len(values) < max_length:
            raise RuntimeError(
                ("Too few input columns ({} out of expected {}) ")
                .format(len(values), max_length))


        labels = values[args.labels_start:args.labels_start+args.labels_dim]

        if args.label_type == 'Category':
            one_hot = ['0'] * num_labels
            # there's only one label
            label = labels[0]
            if label not in label_map:
                raise RuntimeError(("Illegal label value: '{}'").format(label))
            one_hot[label_map[label]] = '1'
            labels = one_hot

        output_file.write("|labels " + " ".join(labels))
        output_file.write("\t")

    elif len(values) < args.features_start+args.features_dim:
        raise RuntimeError(
            ("Too few input columns ({} out of expected {}) ")
            .format(len(values), args.features_start+args.features_dim))

    output_file.write(
        "|features " +
        " ".join(values[args.features_start:args.features_start+args.features_dim]))
    output_file.write("\n")

input_file.close()
output_file.close()

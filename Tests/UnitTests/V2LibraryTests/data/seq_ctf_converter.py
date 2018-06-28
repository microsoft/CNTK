import random

def convert_to_sequence(infile_path, outfile_path, max_crop=3):
    common_prefix = '|'
    label_prefix = 'labels'
    feature_prefix = 'features'

    with open(infile_path, 'r') as infile:
        with open(outfile_path, 'w') as outfile:
            line = infile.readline()
            data_count = 0
            while line:
                if data_count % 5000 == 0:
                    print('process to data_count: ' + str(data_count))

                line_arr = line.strip().split('|')
                # ['', 'labels ... ', 'features ... ']
                label_str = line_arr[1]
                feature_str = line_arr[2]

                label_data_arr = label_str[len(label_prefix):].strip().split(' ')
                feature_data_arr = feature_str[len(feature_prefix):].strip().split(' ')

                ignore_top_row_count = random.randint(0, 3)
                ignore_bottom_row_count = random.randint(0, 3)

                seq_count = 0
                new_count_str = str(data_count)
                new_label_str = ' '.join([common_prefix + label_prefix] + label_data_arr)
                
                while seq_count < 28:
                    if seq_count < ignore_top_row_count:
                        seq_count += 1
                        continue
                    if seq_count >= 28 - ignore_bottom_row_count:
                        break

                    new_feature_str = ' '.join([common_prefix + feature_prefix] + feature_data_arr[seq_count*28:(seq_count + 1)*28])

                    new_line = '\t'.join([new_count_str, new_feature_str])
                    if seq_count - ignore_top_row_count == 0:
                        new_line = '\t'.join([new_line, new_label_str])

                    outfile.write(new_line + '\n')

                    seq_count += 1
                data_count += 1
                line = infile.readline()


#convert_to_sequence('Train-28x28_cntk_text.txt', 'Train-28xseq_cntk_text.txt')
convert_to_sequence('Test-28x28_cntk_text.txt', 'Test-28xseq_cntk_text.txt')
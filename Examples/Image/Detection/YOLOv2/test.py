
import numpy as np

def create_empty_array_of_shape(shape):
    if shape: return [create_empty_array_of_shape(shape[1:]) for i in range(shape[0])]

def create_array(first_dim = 5,second_dim = 2,third_dim=2):

    array = create_empty_array_of_shape([1,first_dim,second_dim,third_dim])
    #array = [[[[] for i in range(third_dim)]for i in range(second_dim)]for i in range(first_dim)]# [[[.0]*third_dim]*second_dim]*first_dim
    array = array[0]
    print (array)
    for x in range(first_dim):
        print("x=" + str(x))
        for y in range(second_dim):
            print("y=" + str(y))
            for z in range(third_dim):
                print("z=" + str(z))
                array[x][y][z] =  (x*100+y)*100+z  #str(x)+"_"+str(y)+"_"+str(z)
                # print(array)

    # array = np.asarray()

    return array


#predict_and_show_image(output, r"D:\local\CNTK-2-0-rc1\cntk\Examples\Image\DataSets\Pascal\VOCdevkit\VOC2007\JPEGImages\000118.jpg", conf_threshold=0.49)

#if False:
    remaining_epochs = nr_of_epoch
    nr_of_stages = len(par_stages_epochs) + 1
    for i in range(nr_of_stages):
        curr_epochs = remaining_epochs if (i == nr_of_stages - 1 or remaining_epochs < par_stages_epochs[i]) else \
        par_stages_epochs[i]
        curr_lambda_coord = par_stages_lambda_coord[i]
        curr_lambda_no_obj = par_stages_lambda_no_obj[i]
        import ipdb;

        ipdb.set_trace()

        if curr_epochs > 0:
            network['trainfunction'].set_lambda_coord(curr_lambda_coord)
            network['trainfunction'].set_lambda_no_obj(curr_lambda_no_obj)
            output = yolov2_train_and_eval(network, train_data, test_data,
                                           max_epochs=curr_epochs,
                                           restore=not args['restart'],
                                           log_to_file=os.path.join(logdir,
                                                                    "log" + str(i)) if logdir is not None else None,
                                           num_mbs_per_log=50,
                                           num_quantization_bits=args['quantized_bits'],
                                           block_size=args['block_samples'],
                                           warm_up=args['distributed_after'],
                                           minibatch_size=args['minibatch_size'],
                                           epoch_size=args['epoch_size'],
                                           gen_heartbeat=True)

        remaining_epochs -= curr_epochs

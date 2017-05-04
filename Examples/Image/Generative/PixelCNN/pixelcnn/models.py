
import cntk as ct
import pixelcnn.nn as nn
import pixelcnn.blocks as bk


def build_model(input_norm, model, loss, first_run=False):
    per_pixel_count = 3*256 if (loss == 'category') else 100

    if model == 'pixelcnn':
        return build_pixelcnn_model(input_norm, per_pixel_count=per_pixel_count)        
    elif model == 'pixelcnn2':
        return build_pixelcnn_2_model(input_norm, per_pixel_count=per_pixel_count)        
    elif model == 'pixelcnnpp':
        return build_pixelcnn_pp_model(input_norm, per_pixel_count=per_pixel_count, first_run=first_run)

    return None

def build_pixelcnn_model(input, 
                         residual_block_count = 4, 
                         input_feature_map    = 256,
                         output_feature_map   = 1024,
                         per_pixel_count      = 3*256):
    '''
    Based on PixelRNN paper (https://arxiv.org/pdf/1601.06759v3.pdf), input must be normalized 
    from -1 to 1 range.
    '''
    input_shape  = input.shape

    net = bk.conv2d(input, input_feature_map, (7,7), (1,1), True, mask_type = 'a')
    for _ in range(residual_block_count):
        net = bk.residual_block(net, mask_type = 'b')

    net = bk.conv2d(net, output_feature_map, (1,1), (1,1), True, mask_type = 'b')

    net = bk.conv2d(net, per_pixel_count, (1,1), (1,1), True, mask_type = 'b')    
    return net

def build_pixelcnn_2_model(input, 
                           block_count        = 4, 
                           input_feature_map  = 256,
                           output_feature_map = 1024,
                           per_pixel_count    = 3*256):
    '''
    Based on PixelCNN2.0 paper (https://arxiv.org/pdf/1606.05328v2.pdf), input must be normalized 
    from -1 to 1 range.
    '''
    input_shape  = input.shape

    net = bk.conv2d(input, input_feature_map, (7,7), (1,1), True, mask_type = 'a')

    input_v = net
    input_h = net
    for _ in range(block_count):
        input_v, input_h = bk.gated_residual_block(input_v, input_h, (3, 3), mask_type = 'b')        
    
    net = input_h
    net = bk.conv2d(net, output_feature_map, (1,1), (1,1), True, mask_type = 'b')

    net = bk.conv2d(net, per_pixel_count, (1,1), (1,1), True, mask_type = 'b')
    return net

def build_pixelcnn_pp_model(x, h = None, dropout_p=0.5, nr_resnet=0, nr_filters=160, per_pixel_count=100, resnet_nonlinearity=nn.concat_elu, nonlinearity=ct.elu, first_run=False):
    """
    Based on PixelCNN++ from https://openreview.net/pdf?id=BJrFC6ceg, the implementation is 
    a port from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp 

    We receive a Tensor x of shape (D1,H,W) (e.g. (3,32,32)) and produce
    a Tensor x_out of shape (D2,H,W) (e.g. (100,32,32)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """
    counters = {}
    xs = x.shape
    x_pad = ct.splice(x, nn.ones((1,)+xs[1:]), axis=0) # add channel of ones to distinguish image from padding later on
    u_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_shape=(2, 3), counters=counters, first_run=first_run))] # stream for pixels above
    ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_shape=(1, 3), counters=counters, first_run=first_run)) + \
                nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_shape=(2, 1), counters=counters, first_run=first_run))] # stream for up and to the left

    for rep in range(nr_resnet):
        u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d, nonlinearity=resnet_nonlinearity, dropout_p=dropout_p, counters=counters, first_run=first_run))
        ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d, nonlinearity=resnet_nonlinearity, dropout_p=dropout_p, counters=counters, first_run=first_run))

    u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, strides=(2, 2), counters=counters, first_run=first_run))
    ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, strides=(2, 2), counters=counters, first_run=first_run))

    for rep in range(nr_resnet):
        u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d, nonlinearity=resnet_nonlinearity, dropout_p=dropout_p, counters=counters, first_run=first_run))
        ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d, nonlinearity=resnet_nonlinearity, dropout_p=dropout_p, counters=counters, first_run=first_run))

    u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, strides=(2, 2), counters=counters, first_run=first_run))
    ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, strides=(2, 2), counters=counters, first_run=first_run))

    for rep in range(nr_resnet):
        u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d, nonlinearity=resnet_nonlinearity, dropout_p=dropout_p, counters=counters, first_run=first_run))
        ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d, nonlinearity=resnet_nonlinearity, dropout_p=dropout_p, counters=counters, first_run=first_run))

    # /////// down pass ////////
    u = u_list.pop()
    ul = ul_list.pop()
    for rep in range(nr_resnet):
        u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d, nonlinearity=resnet_nonlinearity, dropout_p=dropout_p, counters=counters, first_run=first_run)
        ul = nn.gated_resnet(ul, ct.splice(u, ul_list.pop(), axis=0), conv=nn.down_right_shifted_conv2d, nonlinearity=resnet_nonlinearity, dropout_p=dropout_p, counters=counters, first_run=first_run)

    u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, strides=(2, 2), counters=counters, first_run=first_run)
    ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, strides=(2, 2), counters=counters, first_run=first_run)

    for rep in range(nr_resnet+1):
        u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d, nonlinearity=resnet_nonlinearity, dropout_p=dropout_p, counters=counters, first_run=first_run)
        ul = nn.gated_resnet(ul, ct.splice(u, ul_list.pop(), axis=0), conv=nn.down_right_shifted_conv2d, nonlinearity=resnet_nonlinearity, dropout_p=dropout_p, counters=counters, first_run=first_run)

    u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, strides=(2, 2), counters=counters, first_run=first_run)
    ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, strides=(2, 2), counters=counters, first_run=first_run)

    for rep in range(nr_resnet+1):
        u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d, nonlinearity=resnet_nonlinearity, dropout_p=dropout_p, counters=counters, first_run=first_run)
        ul = nn.gated_resnet(ul, ct.splice(u, ul_list.pop(), axis=0), conv=nn.down_right_shifted_conv2d, nonlinearity=resnet_nonlinearity, dropout_p=dropout_p, counters=counters, first_run=first_run)

    x_out = nn.nin(nonlinearity(ul),per_pixel_count, counters=counters, first_run=first_run)

    assert len(u_list) == 0
    assert len(ul_list) == 0

    return x_out

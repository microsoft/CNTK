import pytest
import numpy as np
from cntk.contrib import crosstalk as cstk
import tempfile
workdir = tempfile.gettempdir()

shape1 = (100, 200,)
shape2 = (10, 20,)
param1 = np.random.random(shape1).astype(np.float32)
param2 = np.random.random(shape2).astype(np.float32)

def cntk_baseline_basic():
    import cntk as C
    import cntk.contrib.crosstalk.crosstalk_cntk as crct
    ci = crct.instance
    
    p1 = C.parameter(shape1, init=param1)
    p2 = C.parameter(shape2, init=param2)
    ci.watch(p1, 'p1')
    ci.watch({'param1':p1, 'param2':p2}, 'p1_p2', var_type=crct.DictParameterType)
    ci.set_workdir(workdir)
    ci.fetch('p1', save=True)
    ci.fetch('p1_p2', save=True)
    ci.reset()

def tf_baseline_basic():
    import tensorflow as tf
    import cntk.contrib.crosstalk.crosstalk_tensorflow as crtf
    ci = crtf.instance

    tf.reset_default_graph()
    
    p1 = tf.get_variable("param1", initializer=param1, dtype=tf.float32)
    p2 = tf.get_variable("param2", initializer=param2, dtype=tf.float32)
    ci.watch(p1, 'p1', var_type=crtf.TrainableType)
    ci.watch({'param1':p1, 'param2':p2}, 'p1_p2', var_type=crtf.DictTrainableType)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ci.set_workdir(workdir)
        ci.set_data(sess, None)
        ci.fetch('p1', save=True)
        ci.fetch('p1_p2', save=True)
        ci.reset()
        sess.close()

def test_cntk_basic():
    try:
        import tensorflow
        has_tensorflow = True
    except:
        has_tensorflow = False

    if has_tensorflow:
        tf_baseline_basic()
    else:
        cntk_baseline_basic()

    import cntk as C
    import cntk.contrib.crosstalk.crosstalk_cntk as crct
    ci = crct.instance
    ci.set_workdir(workdir)

    p1 = C.parameter(shape1)
    p2 = C.parameter(shape2)
    ci.watch(p1, 'p1')
    ci.watch({'param1':p1, 'param2':p2}, 'p1_p2', var_type=crct.DictParameterType)
    
    ci.assign('p1', load=True)
    assert np.isclose(p1.value, param1).all()
    
    ci.assign('p1_p2', load=True)
    assert np.isclose(p1.value, param1).all() and np.isclose(p2.value, param2).all()
    
    # test assign with value
    ci.assign('p1', value=param1)
    ci.assign('p1_p2', value={'param1':param1, 'param2':param2})
    
    ci.reset()

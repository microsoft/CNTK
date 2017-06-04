import pytest
import numpy as np
from cntk.contrib import crosstalk as cstk
import tempfile
workdir = tempfile.gettempdir()

input_dim = 3
emb_dim = 100
dict1 = ['a', 'b', 'c']
dict2 = ['c', 'a', 'b']
emb1 = np.random.random((input_dim, emb_dim)).astype(np.float32)
emb2 = [emb1[2], emb1[0], emb1[1]]

def cntk_baseline_embed():
    import cntk as C
    import cntk.contrib.crosstalk.crosstalk_cntk as crct
    ci = crct.instance
    
    embed = C.parameter((input_dim, emb_dim), init=emb1)
    ci.watch(embed, 'embed', var_type=cstk.EmbedAttr,
          attr=cstk.EmbedAttr(dict=dict1, input_dim=input_dim))

    ci.set_workdir(workdir)
    ci.fetch('embed', save=True)
    ci.reset()

def tf_baseline_embed():
    import tensorflow as tf
    import cntk.contrib.crosstalk.crosstalk_tensorflow as crtf
    ci = crtf.instance

    tf.reset_default_graph()
    
    embed = tf.get_variable("embed", initializer=emb1, dtype=tf.float32)
    ci.watch(embed, 'embed', var_type=cstk.EmbedAttr,
          attr=cstk.EmbedAttr(dict=dict1, input_dim=input_dim))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ci.set_workdir(workdir)
        ci.set_data(sess, None)
        ci.fetch('embed', save=True)
        ci.reset()
        sess.close()

def test_cntk_embed():
    try:
        import tensorflow
        has_tensorflow = True
    except:
        has_tensorflow = False

    if has_tensorflow:
        tf_baseline_embed()
    else:
        cntk_baseline_embed()

    import cntk as C
    import cntk.contrib.crosstalk.crosstalk_cntk as crct
    ci = crct.instance
    ci.set_workdir(workdir)

    embed = C.parameter((input_dim, emb_dim,))
    ci.watch(embed, 'embed', var_type=cstk.EmbedAttr,
          attr=cstk.EmbedAttr(dict=dict2, input_dim=input_dim))

    ci.assign('embed', load=True)
    assert np.isclose(emb2, embed.value).all()
    
    # test assign with value
    ci.assign('embed', value={'a':emb1[0], 'b':emb1[1], 'c':emb1[2]})
    
    ci.reset()

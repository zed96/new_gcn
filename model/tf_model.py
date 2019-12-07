import tensorflow as tf
import math

def get_act_fn(name):
    if name == "relu":
        act_fn = tf.nn.relu
    elif name == "elu":
        act_fn = tf.nn.elu
    elif name == "selu":
        act_fn = tf.nn.selu
    elif name == "swish":
        act_fn = tf.nn.swish

    return act_fn

def get_init_fn(name,*args,**kwargs):
    if name == "glorot_uniform":
        init_fn = tf.glorot_uniform_initializer(*args, **kwargs)
    elif name == "glorot_normal":
        init_fn = tf.glorot_normal_initializer(*args, **kwargs)
    elif name == "truncated_normal":
        init_fn = tf.truncated_normal_initializer(*args, **kwargs)
    elif name == "lecun_uniform":
        init_fn = tf.lecun_uniform(*args, **kwargs)
    elif name == "lecun_normal":
        init_fn = tf.lecun_normal(*args, **kwargs)
    elif name == "he_uniform":
        init_fn = tf.keras.initializers.he_uniform(*args, **kwargs)
    elif name == "he_normaa":
        init_fn = tf.keras.initializers.he_normal(*args, **kwargs)
    return init_fn


def Senet_Layer(input_emb, act_fn, init_fn, use_bn, in_size, hidden_size, name="Senet", keep_prob=1.0):
    if isinstance(act_fn, str):
        act_fn = get_act_fn(act_fn)
    if isinstance(init_fn, str):
        if init_fn == "truncated_norm":
            init_fn = get_init_fn(init_fn, stddev=1.0 / math.sqrt(float(in_size)))
        else:
            init_fn = get_init_fn(init_fn)
    with tf.variable_scope(name):
        W_1 = tf.get_variable("W_1", [in_size, hidden_size], initializer= init_fn)
        W_2 = tf.get_variable("W_2", [hidden_size, in_size], initializer= init_fn)

        #reduction
        A_1 = tf.matmul(input_emb, W_1)
        a_1 = act_fn(A_1)
        if keep_prob<1.0:
            print(keep_prob)
            out = tf.nn.dropout(a_1, keep_prob)
        #resume
        A_2 = tf.matmul(a_1, W_2)
        a_2 = act_fn(A_2)
        if keep_prob<1.0:
            print(keep_prob)
            out = tf.nn.dropout(a_2, keep_prob)
        return a_2


def fc(name, act_fn, init_fn, use_bn, no, input, in_size, out_size, keep_prob = 1.0, training=True):
    with tf.variable_scope("{}_{}".format(name, no)):
        if isinstance(act_fn, str):
            act_fn = get_act_fn(act_fn)
        if isinstance(init_fn, str):
            if init_fn == "truncated_normal":
                init_fn = get_init_fn(init_fn, stddev = 1.0/math.sqrt(float(in_size)))
            else:
                init_fn = get_init_fn(init_fn)
        w = tf.get_variable('w', [in_size, out_size], initializer = init_fn)
        b = tf.get_variable('b', [out_size], initializer = tf.zeros_initializer)
        o1 = tf.add(tf.matmul(input, w), b)
        o = act_fn(o1)
        if keep_prob < 1.0:
            print(keep_prob)
            o = tf.nn.dropout(o, keep_prob)
        return o1, o

def vae_module(input, in_size, out_size, keep_prob=1.0, training=True):
    #vae_encoder
    _, mu = fc(name="vae", act_fn='relu', init_fn='truncated_normal', use_bn=False,
            no='mean', input=input, in_size=in_size, out_size=out_size, training=training)
    _, logstd = fc(name="vae", act_fn='relu', init_fn='truncated_normal', use_bn=False,
            no='logstd', input=input, in_size=in_size, out_size=out_size, training=training)
    #reparameterization trick
    #get_eps
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(logstd/2) * eps

#Simon32/64
import numpy as np
from os import urandom

def block_size():
    return(32)#

def WORD_SIZE():
    return(16)

def key_words():
    return(4)

def const_seq():
    return(8)

def all_round():
    return(32)

Z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]

MOD_MASK = (2 ** WORD_SIZE()) - 1

def rol(x,k):
    return(((x << k) & MOD_MASK) | (x >> (WORD_SIZE() - k)));

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MOD_MASK));

def left_round(value, shiftBits):
	t1 = (value >> (WORD_SIZE() - shiftBits)) ^ (value << shiftBits)
	t2 = ((2 ** WORD_SIZE()) - 1)
	return t1 & t2

def right_round(value, shiftBits):
	t1 = (value << (WORD_SIZE() - shiftBits)) ^ (value >> shiftBits)
	t2 = ((2 ** WORD_SIZE()) - 1)
	return t1 & t2

def enc_one_round(p,k):
    x,y=p[0],p[1]
    tmp=x
    x=y^((left_round(x,1))&(left_round(x,8)))^(left_round(x,2))^k
    y=tmp
    return (x,y)

def dec_one_round(c, k):
    x,y=c[0],c[1]
    tmp=y
    y=x^k^(left_round(y,2))^((left_round(y,1))&(left_round(y,8)))
    x=tmp
    return (x,y)


def expand_key_1(k):
    subkey = [0]
    subkey[0] = k[3]
    return (subkey)


def expand_keys(k,t):
    subkey= [0 for i in range(t)]
    c=pow(2,WORD_SIZE())-4
    m=key_words()
    for i in range(0,m):
        subkey[i]=k[m-1-i]
    for i in range(m,t):
        tmp=right_round(subkey[i-1], 3)
        if (m==4):
            tmp=tmp^subkey[i-3]
        tmp=tmp^(right_round(tmp,1))
        subkey[i]=c^Z0[(i-m)%31]^tmp^subkey[i-m]
    return(subkey)
    
def encryption(p,ks):
    x,y= p[0],p[1]
    for k in ks:
        x,y=enc_one_round((x,y), k)
    return(x,y)

def decryption(c, ks):
    x, y = c[0], c[1]
    
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k)
    return(x,y)

def check_testvector():
    key= (0x1918,0x1110,0x0908,0x0100)
    pt = (0x6565, 0x6877)
    ks = expand_key(key,all_round())
    ct = encrypt(pt, ks)
    p  = decrypt(ct, ks)
    if ((ct == (0xc69b, 0xe9bb))and(p == (0x6565, 0x6877))):
        print("Testvector verified.")
        return(1)
    else:
        print("Testvector not verified.")
        return(0)

def convert_to_binaryy(l):
    n = len(l)
    k = WORD_SIZE() * n
    X = np.zeros((k, len(l[0])), dtype=np.uint8)
    for i in range(k):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - 1 - i % WORD_SIZE()
        X[i] = (l[index] >> offset) & 1
    X = X.transpose()
    return(X)


def convert_to_binary(arr,s_groups=1):
  X = np.zeros((8 * WORD_SIZE() * s_groups,len(arr[0])),dtype=np.uint8) 
  for i in range(8 * WORD_SIZE() * s_groups):
    index = i // WORD_SIZE() 
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
    X[i] = (arr[index] >> offset) & 1
  X = X.transpose() 
  return(X)

def make_train_data(n, nr, a=((0x0,0x1), (0x0,0x4), (0x0,0x8)), b=((0x0,0x2),(0x0000,0x0010),(0x0,0x0040)), s_groups=1): 
   
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    num_rand_samples = np.sum(Y==0)

    key = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1) 

    keys = expand_keys(key, nr)

    X = []
    
    for i in range(s_groups):

        plain1_1 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        plain1_2 = np.frombuffer(urandom(2 * n), dtype=np.uint16)

        plain2_1 = np.where(Y == 0, plain1_1 ^ a[0][0], plain1_1 ^ b[0][0])
        plain2_2 = np.where(Y == 0, plain1_2 ^ a[0][1], plain1_2 ^ b[0][1])
        
        plain3_1 = np.where(Y == 0, plain1_1 ^ a[1][0], plain1_1 ^ b[1][0])
        plain3_2 = np.where(Y == 0, plain1_2 ^ a[1][1], plain1_2 ^ b[1][1])
       
        plain4_1 = np.where(Y == 0, plain1_1 ^ a[2][0], plain1_1 ^ b[2][0])
        plain4_2 = np.where(Y == 0, plain1_2 ^ a[2][1], plain1_2 ^ b[2][1])
        
        cipher1_1, cipher1_2 = encryption((plain1_1, plain1_2), keys)
        cipher2_1, cipher2_2 = encryption((plain2_1, plain2_2), keys)
        cipher3_1, cipher3_2 = encryption((plain3_1, plain3_2), keys)
        cipher4_1, cipher4_2 = encryption((plain4_1, plain4_2), keys)


        X.append(cipher1_1)
        X.append(cipher1_2)
        X.append(cipher2_1)
        X.append(cipher2_2)
        X.append(cipher3_1)
        X.append(cipher3_2)
        X.append(cipher4_1)
        X.append(cipher4_2)

    XT = convert_to_binary(X,s_groups=s_groups)


    return XT, Y

def make_train_rkdata(n, nr, a=((0x0040,0x0), (0x0,0x8000), (0x0060,0x0)), b=((0x0020,0x0),(0x0040,0x8000),(0x0010,0x2000)), s_groups=1): 
   
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    num_rand_samples = np.sum(Y==0)

    key = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)

    key01 = np.array([key[0], key[1], key[2], key[3] ^ a[0][1]], dtype=np.uint16)
    key02 = np.array([key[0], key[1], key[2], key[3] ^ a[1][1]], dtype=np.uint16)
    key03 = np.array([key[0], key[1], key[2], key[3] ^ a[2][1]], dtype=np.uint16)
    key11 = np.array([key[0], key[1], key[2], key[3] ^ b[0][1]], dtype=np.uint16)
    key12 = np.array([key[0], key[1], key[2], key[3] ^ b[1][1]], dtype=np.uint16)
    key13 = np.array([key[0], key[1], key[2], key[3] ^ b[2][1]], dtype=np.uint16)
   
    #key0 = np.array([key[0], key[1], key[2], key[3] ^ 0x40], dtype=np.uint16)
    #key1 = np.array([key[0], key[1], key[2], key[3] ^ 0x20], dtype=np.uint16)
 

    keys01 = expand_keys(key01, nr)
    keys02 = expand_keys(key02, nr)
    keys03 = expand_keys(key03, nr)
    keys11 = expand_keys(key11, nr)
    keys12 = expand_keys(key12, nr)
    keys13 = expand_keys(key13, nr)

    keys = expand_keys(key, nr)
    
    #keys0 = expand_keys(key0, nr)
    #keys1 = expand_keys(key1, nr)

    X = []
    
    for i in range(s_groups):

        plain1_1 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        plain1_2 = np.frombuffer(urandom(2 * n), dtype=np.uint16)

        plain2_1 = np.where(Y == 0, plain1_1 ^ a[0][0], plain1_1 ^ b[0][0])
        plain2_2 = np.where(Y == 0, plain1_2 ^ a[0][1], plain1_2 ^ b[0][1])
        
        plain3_1 = np.where(Y == 0, plain1_1 ^ a[1][0], plain1_1 ^ b[1][0])
        plain3_2 = np.where(Y == 0, plain1_2 ^ a[1][1], plain1_2 ^ b[1][1])
       
        plain4_1 = np.where(Y == 0, plain1_1 ^ a[2][0], plain1_1 ^ b[2][0])
        plain4_2 = np.where(Y == 0, plain1_2 ^ a[2][1], plain1_2 ^ b[2][1])
        
        cipher1_1, cipher1_2 = encryption((plain1_1, plain1_2), keys)
        cipher2_1, cipher2_2 = np.where(Y == 0, encryption((plain2_1, plain2_2), keys01), encryption((plain2_1, plain2_2), keys11))
        cipher3_1, cipher3_2 = np.where(Y == 0, encryption((plain3_1, plain3_2), keys02), encryption((plain3_1, plain3_2), keys12))
        cipher4_1, cipher4_2 = np.where(Y == 0, encryption((plain4_1, plain4_2), keys03), encryption((plain4_1, plain4_2), keys13))


        X.append(cipher1_1)
        X.append(cipher1_2)
        X.append(cipher2_1)
        X.append(cipher2_2)
        X.append(cipher3_1)
        X.append(cipher3_2)
        X.append(cipher4_1)
        X.append(cipher4_2)
        
    XT = convert_to_binary(X,s_groups=s_groups)
	
def make_classical_data(n, nr, a=((0x0001,0x1150), (0x0100,0x0444), (0x0101,0x4104)), s_groups=1): 

    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    num_rand_samples = np.sum(Y==0)

    key = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)

    keys = expand_keys(key, nr)
    

    X = []
    
    for i in range(s_groups):

        plain1_1 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        plain1_2 = np.frombuffer(urandom(2 * n), dtype=np.uint16)

        plain2_1 = plain1_1 ^ a[0][0]
        plain2_2 = plain1_2 ^ a[0][1]
        
        plain3_1 = plain1_1 ^ a[1][0]
        plain3_2 = plain1_2 ^ a[1][1]
       
        plain4_1 = plain1_1 ^ a[2][0]
        plain4_2 = plain1_2 ^ a[2][1]
        
        plain2_1[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
        plain2_2[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
        
        plain3_1[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
        plain3_2[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
       
        plain4_1[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
        plain4_2[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
        
        cipher1_1, cipher1_2 = encryption((plain1_1, plain1_2), keys)
        cipher2_1, cipher2_2 = encryption((plain2_1, plain2_2), keys)
        cipher3_1, cipher3_2 = encryption((plain3_1, plain3_2), keys)
        cipher4_1, cipher4_2 = encryption((plain4_1, plain4_2), keys)


        X.append(cipher1_1)
        X.append(cipher1_2)
        X.append(cipher2_1)
        X.append(cipher2_2)
        X.append(cipher3_1)
        X.append(cipher3_2)
        X.append(cipher4_1)
        X.append(cipher4_2)
        
    XT = convert_to_binary(X,s_groups=s_groups)


    return XT, Y
	
def make_diff_data(n, nr, a=((0x0040,0x0), (0x0,0x8000), (0x0060,0x0)), b=((0x0020,0x0),(0x0040,0x8000),(0x0010,0x2000)), s_groups=1):
  #generate labels
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  num_rand_samples = np.sum(Y==0)
  #generate keys
  key = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  #expand keys
  keys = expand_keys(key, nr);
  #generate blinding values
  R0 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  R1 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  X = []
    
  for i in range(s_groups):
  	#generate plaintexts
  	plain1_1 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
  	plain1_2 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
  	#apply input difference
  	plain2_1 = np.where(Y == 0, plain1_1 ^ a[0][0], plain1_1 ^ b[0][0])
  	plain2_2 = np.where(Y == 0, plain1_2 ^ a[0][1], plain1_2 ^ b[0][1])
        
  	plain3_1 = np.where(Y == 0, plain1_1 ^ a[1][0], plain1_1 ^ b[1][0])
  	plain3_2 = np.where(Y == 0, plain1_2 ^ a[1][1], plain1_2 ^ b[1][1])
  	plain4_1 = np.where(Y == 0, plain1_1 ^ a[2][0], plain1_1 ^ b[2][0])
  	plain4_2 = np.where(Y == 0, plain1_2 ^ a[2][1], plain1_2 ^ b[2][1])


  	cipher1_1, cipher1_2 = encryption((plain1_1, plain1_2), keys)
  	cipher2_1, cipher2_2 = np.where(Y == 0, encryption((plain2_1, plain2_2), keys), encryption((plain2_1, plain2_2), keys))
  	cipher3_1, cipher3_2 = np.where(Y == 0, encryption((plain3_1, plain3_2), keys), encryption((plain3_1, plain3_2), keys))
  	cipher4_1, cipher4_2 = np.where(Y == 0, encryption((plain4_1, plain4_2), keys), encryption((plain4_1, plain4_2), keys))

  	#apply blinding to the samples labelled as random
  	cipher1_1[Y==0] = cipher1_1[Y==0] ^ R0 
  	cipher1_2[Y==0] = cipher1_2[Y==0] ^ R1
  	cipher2_1[Y==0] = cipher2_1[Y==0] ^ R0 
  	cipher2_2[Y==0] = cipher2_2[Y==0] ^ R1
  	cipher3_1[Y==0] = cipher3_1[Y==0] ^ R0 
  	cipher3_2[Y==0] = cipher3_2[Y==0] ^ R1
  	cipher4_1[Y==0] = cipher4_1[Y==0] ^ R0 
  	cipher4_2[Y==0] = cipher4_2[Y==0] ^ R1
  	
  	X.append(cipher1_1)
  	X.append(cipher1_2)
  	X.append(cipher2_1)
  	X.append(cipher2_2)
  	X.append(cipher3_1)
  	X.append(cipher3_2)
  	X.append(cipher4_1)
  	X.append(cipher4_2)
        
  XT = convert_to_binary(X,s_groups=s_groups)

  return(XT,Y)
	
from tensorflow.keras.models import Model
from tensorflow.keras.layers import multiply,GlobalAvgPool1D,Dense, Conv1D, Input, Reshape, Add, Flatten, BatchNormalization, Activation, LayerNormalization, MultiHeadAttention, LSTM, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np
from pickle import dump
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation, concatenate
from keras import backend as K
from keras.regularizers import l2

def get_dilation_rates(input_size):
    drs = []
    while input_size >= 8:
        drs.append(int(input_size / 2 - 1))
        input_size = input_size // 2

    return drs
    

def make_model(input_size=128, n_filters=32, kernel_size=2, n_add_filters=16):


    # determine the dilation rates from the given input size
    dilation_rates = get_dilation_rates(input_size)

    # prediction head parameters (similar to Gohr)
    d1 = 256 # TODO this can likely be reduced to 64.
    d2 = 64
    reg_param = 1e-5

    # define the input shape
    inputs = Input(shape=(input_size, 1))
    x = inputs

    # normalize the input data to a range of [-1, 1]:
    x = tf.subtract(x, 0.5)
    x = tf.divide(x, 0.5)

        
    for dilation_rate in dilation_rates:
        ### wide-narrow blocks
        x = Conv1D(filters=n_filters,
                   kernel_size=kernel_size,
                   padding='valid',
                   dilation_rate=dilation_rate,
                   strides=1,
                   activation='relu')(x)
        x = BatchNormalization()(x)
        
        x_skip = x
        
        x = Conv1D(filters=n_filters,
                   kernel_size=kernel_size,
                   padding='causal',
                   dilation_rate=1,
                   activation='relu')(x)
        
    
        x = Add()([x, x_skip])
        x = BatchNormalization()(x)

        n_filters += n_add_filters
        
    ### prediction head
    out = tf.keras.layers.Flatten()(x)

    dense0 = Dense(d1, kernel_regularizer=l2(reg_param))(out);
    dense0 = BatchNormalization()(dense0);
    dense0 = Activation('relu')(dense0);
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(dense0);
    dense1 = BatchNormalization()(dense1);
    dense1 = Activation('relu')(dense1);
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
    dense2 = BatchNormalization()(dense2);
    dense2 = Activation('relu')(dense2);
    out = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_param))(dense2)

    model = Model(inputs, out)

    return model


import warnings
warnings.filterwarnings("ignore")
# ------------------------------------------------
# TensorFlow import
# ------------------------------------------------
# TensorFlow setting: Which GPU to use and not to consume the whole GPU:
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'            # Which GPU to use.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'            # Filters TensorFlow warnings.
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'    # Prevents TensorFlow from consuming the whole GPU.
# Import TensorFlow:
import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ------------------------------------------------
# Other imports
# ------------------------------------------------
import logging
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam


# ------------------------------------------------
# Configuration and constants
# ------------------------------------------------
logging.basicConfig(level=logging.FATAL)

ABORT_TRAINING_BELOW_ACC = 0.505   # if the validation accuracy reaches or falls below this limit, abort further training.
EPOCHS = 120                        # train for 120 epochs
NUM_SAMPLES = 10**7                 # create 10 million training samples
NUM_VAL_SAMPLES = 10**6             # create 1 million validation samples
bs = 10000                    # training batch size

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

def train_one_round(model,
                    X, Y, X_val, Y_val,
                    round_number: int,
                    epochs=40,
                    model_name = 'model',
                    load_weight_file=False,
                    log_prefix = '',
                    LR_scheduler = None):
    """Train the `model` on the training data (X,Y) for one round.

    :param model: TensorFlow neural network
    :param X, Y: training data
    :param X_val, Y_val: validation data
    :param epochs: number of epochs to train
    :param load_weight_file: Boolean (if True: load weights from previous round.)
    :return: best validation accuracy
    """
    #------------------------------------------------
    # Handle model weight checkpoints
    #------------------------------------------------
    from tensorflow.keras.callbacks import ModelCheckpoint

    # load weight checkpoint from previous round?
    if load_weight_file:
        logging.info("loading weights from previous round...")
        model.load_weights(f'{log_prefix}_{model_name}_round{round_number-1}.h5')


    # create model checkpoint callback for this round
    checkpoint = ModelCheckpoint(f'{log_prefix}_{model_name}_round{round_number}.h5', monitor='val_loss', save_best_only = True)
    

    # Define callbacks
    if LR_scheduler == None:
        callbacks = [checkpoint]
    else:
        callbacks = [checkpoint, LearningRateScheduler(LR_scheduler)]



    #------------------------------------------------
    # Train the model
    #------------------------------------------------
    history = model.fit(X, Y, epochs=epochs, batch_size=bs,
                        validation_data=(X_val, Y_val), callbacks=callbacks, verbose = 1)



    # Check if this round has the best validation accuracy
    val_acc = np.max(history.history['val_acc'])
    if not hasattr(train_one_round, 'best_val_acc') or val_acc > train_one_round.best_val_acc:
        train_one_round.best_val_acc = val_acc

        # Save the accuracy plot
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['acc'], label='Training Accuracy')
        plt.plot(history.history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{log_prefix}_{model_name}_best_accuracy.png')
        plt.show()

        # Save the loss plot
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{log_prefix}_{model_name}_best_loss.png')
        plt.show()
        
    # save the training history
    pd.to_pickle(history.history, f'{log_prefix}_{model_name}_training_history_round{round_number}.pkl')
    return np.max(history.history['val_acc'])

def train_neural_distinguisher(starting_round, data_generator, model_name, input_size, word_size, log_prefix = './', _epochs = EPOCHS, _num_samples=None):
    """Staged training of model_name starting in `starting_round` for a cipher with data generated by `data_generator`.

    :param starting_round:  Integer in which round to start the neural network training.
    :param data_generator:  Data_generator(number_of_samples, current_round) returns X, Y.
    :return: best_round, best_val_acc
    """

    #------------------------------------------------
    # Create the neural network model
    #------------------------------------------------
    logging.info(f'CREATE NEURAL NETWORK MODEL {model_name}')
    lr = cyclic_lr(10, 0.001, 0.0002)
    strategy = tf.distribute.MirroredStrategy(
    devices=["/gpu:0","/gpu:1"])
    batch_size = bs * strategy.num_replicas_in_sync

    with strategy.scope():
        if model_name == 'model':
            model = make_model(2*input_size)
            optimizer = tf.keras.optimizers.Adam(amsgrad=True)
            model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])

    #------------------------------------------------
    # Start staged training from starting_round
    #------------------------------------------------
    current_round = starting_round
    load_weight_file = False
    best_val_acc = None
    best_round = None
    #------------------------------------------------
    # Using custom parameters if needed
    #------------------------------------------------
    if _epochs == None:
        epochs = EPOCHS
    else:
        epochs = _epochs
    if _num_samples == None:
        num_samples = NUM_SAMPLES
    else:
        num_samples = _num_samples

    print(f'Training on {epochs} epochs ...')
    while True:
        # ------------------------------------------------
        # Train one round
        # ------------------------------------------------
        # create data
        logging.info(f"CREATE CIPHER DATA for round {current_round} (training samples={num_samples:.0e}, validation samples={NUM_VAL_SAMPLES:.0e})...")
        X, Y = make_train_data(NUM_SAMPLES, current_round)
        X_val, Y_val = make_train_data(NUM_VAL_SAMPLES, current_round)

        # train model for the current round
        logging.info(f"TRAIN neural network for round {current_round}...")
        val_acc = train_one_round(model,
                                    X, Y, X_val, Y_val,
                                    current_round,
                                    epochs = epochs,
                                    load_weight_file = load_weight_file,
                                    log_prefix = log_prefix,
                                    model_name = model_name,
                                    LR_scheduler = lr
                                    )
        print(f'{model_name}, round {current_round}. Best validation accuracy: {val_acc}', flush=True)

        # after the starting_round, load the weight files:
        load_weight_file = True

        # abort further training if the validation accuracy is too low
        if val_acc <= ABORT_TRAINING_BELOW_ACC:
            logging.info(f"ABORT TRAINING (best validation accuracy {val_acc}<={ABORT_TRAINING_BELOW_ACC}).")
            break
        # otherwise save results as currently best reached round
        else:
            best_round = current_round
            best_val_acc = val_acc
            current_round += 1
            
        # free the memory  
        del X
        del Y
        del X_val
        del Y_val
        tf.keras.backend.clear_session()

    return best_round, best_val_acc

import os
import tqdm
from os import urandom
from glob import glob
import ray
import importlib
import numpy as np
import os
import argparse
def trainNeuralDistinguishers(output_dir ,starting_round, epochs = None, nets =['senet'], num_samples=None):
    plain_bits = 64
    key_bits = 64
    word_size = 16
    encryption_function = encryption
    results = {}
    for net in nets:
        print(f'Training {net} starting from round {starting_round}...')
        results[net] = {}
        best_round, best_val_acc = train_neural_distinguisher(
            starting_round = starting_round,
            data_generator = lambda num_samples, nr : make_train_data(num_samples, nr),
            model_name = net,
            input_size = plain_bits,
            word_size = word_size,
            log_prefix = f'{output_dir}',
            _epochs = epochs,
            _num_samples = num_samples)
        results[net]['Best round'] = best_round
        results[net]['Validation accuracy'] = best_val_acc
    results_file_path = os.path.join(output_dir, 'results.txt')  # Assuming you want to create a 'results.txt' file in the output directory

    with open(results_file_path, 'a') as f:
        for net in nets:
            f.write(f'{net} : {results[net]["Best round"]}, {results[net]["Validation accuracy"]}\n')
    print(results)
    return results

def parse_and_validate():
    parser = argparse.ArgumentParser(description='Obtain good input differences for neural cryptanalysis.')
    parser.add_argument('-o', '--output', type=str, nargs='?', default ='results',
            help=f'the folder where to store the experiments results')
    arguments, unknown = parser.parse_known_args()
    output_dir = arguments.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

    
output_dir = parse_and_validate()

trainNeuralDistinguishers(output_dir, 7, nets =['model'])

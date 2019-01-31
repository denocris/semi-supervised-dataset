
#from preprocessing import process_text
import preprocessing
import params as pm
import tensorflow as tf
# List of Models

def lstm_model_fn(features, labels, mode, params):

    hidden_units = params.hidden_units
    output_layer_size = len(pm.TARGET_LABELS)
    embedding_size = params.embedding_size
    embedding_initializer = params.embedding_initializer
    forget_bias = params.forget_bias
    learning_rate = params.learning_rate
    dropout_rate = params.dropout_rate

    # word_id_vector
    word_id_vector = preprocessing.process_text(features[pm.TEXT_FEATURE_NAME])
    #tf.Print(word_id_vector,[word_id_vector])
    #feature_length_array = [len(np.nonzero(word_id_vector[i])[0]) for i in range(BATCH_SIZE)]
    feature_length_array = tf.count_nonzero(word_id_vector, 1)


    # layer to take each word_id and convert it into vector (embeddings)
    word_embeddings = tf.contrib.layers.embed_sequence(word_id_vector,
                                                 vocab_size=pm.N_WORDS,
                                                 embed_dim=embedding_size,
                                                 initializer=embedding_initializer,
                                                 trainable=pm.TRAINABLE_EMB)

    training = (mode == tf.estimator.ModeKeys.TRAIN)
    dropout_emb = tf.layers.dropout(inputs=word_embeddings, rate=0.2, training=training)


    # configure the RNN
    #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100)

    # configure the RNN

    # rnn_layers = [tf.contrib.rnn.LayerNormBasicLSTMCell(
    #         num_units=size,
    #         forget_bias=forget_bias,
    #         activation=tf.nn.tanh,
    #         layer_norm=True,
    #         dropout_keep_prob=0.5) for size in hidden_units]

    #print('------------- emb_output ---------------',dropout_emb.get_shape())
    rnn_layers = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(
            num_units=size,
            forget_bias=forget_bias,
            activation=tf.nn.tanh), output_keep_prob=1.0,state_keep_prob=1.0,
                ) for size in hidden_units]

        # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    input_layer = dropout_emb
    #input_layer = tf.unstack(word_embeddings, axis=1)


    _, final_states = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=input_layer,
                                   sequence_length=feature_length_array,
                                   dtype=tf.float32)


    # slice to keep only the last cell of the RNN
    rnn_output = final_states[-1].h

    #print('------------- rnn_output ---------------',rnn_output.get_shape())

    # Connect the output layer (logits) to the hidden layer (no activation fn)
    logits = tf.layers.dense(inputs=rnn_output,
                             units=output_layer_size,
                             activation=None)
    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        #print("-------------------- Predicting ---------------------")
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Convert predicted_indices back into strings
        predictions = {
            'class': tf.gather(pm.TARGET_LABELS, predicted_indices),
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }

        # Provide an estimator spec for `ModeKeys.PREDICT` modes.
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)

    # weights
    #weights = features[WEIGHT_COLUNM_NAME]

    # Calculate loss using softmax cross entropy
    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

    #loss = tf.losses.sparse_softmax_cross_entropy(
    #    logits=logits, labels=labels,
    #    weights=weights)
    #
    tf.summary.scalar('loss', loss)

    probabilities = tf.nn.softmax(logits)
    predicted_indices = tf.argmax(probabilities, 1)

    accuracy = tf.metrics.accuracy(labels, predicted_indices)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Create Optimiser
        one_epoch_in_step = pm.TRAIN_SIZE/pm.BATCH_SIZE
        learning_func = learning_rate #tf.random_uniform([1],minval=learning_rate,maxval=0.0001)[0]
        #learning_func = tf.cond(tf.less(tf.train.get_global_step(), tf.cast(5*one_epoch_in_step, tf.int64)), lambda: 0.0008, lambda: learning_func)
        learning_func = tf.cond(tf.less(tf.train.get_global_step(), tf.cast(12*one_epoch_in_step, tf.int64)), lambda: 10*learning_func, lambda: learning_func)
        learning_func = tf.cond(tf.less(tf.train.get_global_step(), tf.cast(6*one_epoch_in_step, tf.int64)), lambda: 50*learning_func, lambda: learning_func)
        #learning_func = tf.cond(tf.less(tf.train.get_global_step(), tf.cast(3*one_epoch_in_step, tf.int64)), lambda: 0.0005, lambda: learning_func)
        if pm.DECAY_LEARNING_RATE_ACTIVE:
            decayed_learning_rate = tf.train.exponential_decay(learning_func,
                                            global_step=tf.train.get_global_step(),
                                            decay_steps=one_epoch_in_step * 10,
                                            decay_rate=0.90,
                                            staircase=True)
            tf.summary.scalar('learning rate', decayed_learning_rate)
        else:
            decayed_learning_rate = learning_rate

        # Create Optimiser
        optimizer = tf.train.AdamOptimizer(decayed_learning_rate)

        # Create training operation
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        tf.summary.scalar('accuracy', accuracy[1])

        # Provide an estimator spec for `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Return accuracy and area under ROC curve metrics
        labels_one_hot = tf.one_hot(
            labels,
            depth=len(pm.TARGET_LABELS),
            on_value=True,
            off_value=False,
            dtype=tf.bool
        )

        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predicted_indices) ,#, weights=weights),
            'auroc': tf.metrics.auc(labels_one_hot, probabilities) # , weights=weights)
        }

        # Provide an estimator spec for `ModeKeys.EVAL` modes.
        return tf.estimator.EstimatorSpec(mode, loss=loss,eval_metric_ops=eval_metric_ops)


def bidir_lstm_model_fn(features, labels, mode, params):

    hidden_units = params.hidden_units
    output_layer_size = len(pm.TARGET_LABELS)
    embedding_size = params.embedding_size
    embedding_initializer = params.embedding_initializer
    forget_bias = params.forget_bias
    learning_rate = params.learning_rate
    dropout_rate = params.dropout_rate

    # word_id_vector
    word_id_vector = preprocessing.process_text(features[pm.TEXT_FEATURE_NAME])
    #tf.Print(word_id_vector,[word_id_vector])
    #feature_length_array = [len(np.nonzero(word_id_vector[i])[0]) for i in range(BATCH_SIZE)]
    feature_length_array = tf.count_nonzero(word_id_vector, 1)


    # layer to take each word_id and convert it into vector (embeddings)
    word_embeddings = tf.contrib.layers.embed_sequence(word_id_vector,
                                                 vocab_size=pm.N_WORDS,
                                                 embed_dim=embedding_size,
                                                 initializer=embedding_initializer,
                                                 trainable=pm.TRAINABLE_EMB)

    training = (mode == tf.estimator.ModeKeys.TRAIN)
    dropout_emb = tf.layers.dropout(inputs=word_embeddings, rate=0.2, training=training)


    # configure the RNN
    #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100)

    # configure the RNN

    # rnn_layers = [tf.contrib.rnn.LayerNormBasicLSTMCell(
    #         num_units=size,
    #         forget_bias=forget_bias,
    #         activation=tf.nn.tanh,
    #         layer_norm=True,
    #         dropout_keep_prob=0.5) for size in hidden_units]

    print('------------- emb_output ---------------',dropout_emb.get_shape())
    rnn_layers_fw = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(
            num_units=size,
            forget_bias=forget_bias,
            activation=tf.nn.tanh), output_keep_prob=1.0,state_keep_prob=1.0,
                ) for size in hidden_units]
    rnn_layers_bw = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(
            num_units=size,
            forget_bias=forget_bias,
            activation=tf.nn.tanh), output_keep_prob=1.0,state_keep_prob=1.0,
                ) for size in hidden_units]

        # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_fw)
    multi_rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_bw)

    input_layer = dropout_emb
    #input_layer = tf.unstack(word_embeddings, axis=1)

    _, final_states = rnn.static_bidirectional_rnn(cell_fw=multi_rnn_cell_fw,
                                                   cell_bw=multi_rnn_cell_bw,
                                                   inputs=input_layer,
                                                   sequence_length=feature_length_array,
                                                   dtype=tf.float32)




    # slice to keep only the last cell of the RNN
    rnn_output = final_states[-1].h

    print('------------- rnn_output ---------------',rnn_output.get_shape())

    # Connect the output layer (logits) to the hidden layer (no activation fn)
    logits = tf.layers.dense(inputs=rnn_output,
                             units=output_layer_size,
                             activation=None)
    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        #print("-------------------- Predicting ---------------------")
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Convert predicted_indices back into strings
        predictions = {
            'class': tf.gather(pm.TARGET_LABELS, predicted_indices),
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }

        # Provide an estimator spec for `ModeKeys.PREDICT` modes.
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)

    # weights
    #weights = features[WEIGHT_COLUNM_NAME]

    # Calculate loss using softmax cross entropy
    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

    #loss = tf.losses.sparse_softmax_cross_entropy(
    #    logits=logits, labels=labels,
    #    weights=weights)
    #
    tf.summary.scalar('loss', loss)

    probabilities = tf.nn.softmax(logits)
    predicted_indices = tf.argmax(probabilities, 1)

    accuracy = tf.metrics.accuracy(labels, predicted_indices)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Create Optimiser
        one_epoch_in_step = pm.TRAIN_SIZE/pm.BATCH_SIZE
        learning_func = learning_rate #tf.random_uniform([1],minval=learning_rate,maxval=0.0001)[0]
        #learning_func = tf.cond(tf.less(tf.train.get_global_step(), tf.cast(5*one_epoch_in_step, tf.int64)), lambda: 0.0008, lambda: learning_func)
        learning_func = tf.cond(tf.less(tf.train.get_global_step(), tf.cast(12*one_epoch_in_step, tf.int64)), lambda: 10*learning_func, lambda: learning_func)
        learning_func = tf.cond(tf.less(tf.train.get_global_step(), tf.cast(6*one_epoch_in_step, tf.int64)), lambda: 50*learning_func, lambda: learning_func)
        #learning_func = tf.cond(tf.less(tf.train.get_global_step(), tf.cast(3*one_epoch_in_step, tf.int64)), lambda: 0.0005, lambda: learning_func)
        if pm.DECAY_LEARNING_RATE_ACTIVE:
            decayed_learning_rate = tf.train.exponential_decay(learning_func,
                                            global_step=tf.train.get_global_step(),
                                            decay_steps=one_epoch_in_step * 10,
                                            decay_rate=0.90,
                                            staircase=True)
            tf.summary.scalar('learning rate', decayed_learning_rate)
        else:
            decayed_learning_rate = learning_rate

        # Create Optimiser
        optimizer = tf.train.AdamOptimizer(decayed_learning_rate)

        # Create training operation
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        tf.summary.scalar('accuracy', accuracy[1])

        # Provide an estimator spec for `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Return accuracy and area under ROC curve metrics
        labels_one_hot = tf.one_hot(
            labels,
            depth=len(pm.TARGET_LABELS),
            on_value=True,
            off_value=False,
            dtype=tf.bool
        )

        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predicted_indices) ,#, weights=weights),
            'auroc': tf.metrics.auc(labels_one_hot, probabilities) # , weights=weights)
        }

        # Provide an estimator spec for `ModeKeys.EVAL` modes.
        return tf.estimator.EstimatorSpec(mode, loss=loss,eval_metric_ops=eval_metric_ops)


def cnn_model_fn(features, labels, mode, params):

    hidden_units = params.hidden_units
    output_layer_size = len(pm.TARGET_LABELS)
    embedding_size = params.embedding_size
    embedding_initializer = params.embedding_initializer
    learning_rate = params.learning_rate
    window_size = params.window_size
    dropout_rate = params.dropout_rate
    stride = int(window_size/2)
    filters = params.filters



    # word_id_vector
    word_id_vector = preprocessing.process_text(features[pm.TEXT_FEATURE_NAME])
    # print(' - - - - - - -  - - - - - - -  - - - - -', word_id_vector[0])
    # feature_length_array = [len(np.nonzero(word_id_vector[i])[0]) for i in range(BATCH_SIZE)]
    feature_length_array = tf.count_nonzero(word_id_vector, 1)

    # layer to take each word_id and convert it into vector (embeddings)
    word_embeddings = tf.contrib.layers.embed_sequence(word_id_vector,
                                                 vocab_size=pm.N_WORDS,
                                                 embed_dim=embedding_size,
                                                 initializer=embedding_initializer,
                                                 trainable=pm.TRAINABLE_EMB)


    training = (mode == tf.estimator.ModeKeys.TRAIN)
    dropout_emb = tf.layers.dropout(inputs=word_embeddings, rate=0.2, training=training)
    # convolution: a sentence can be seen like an image with dimansion length x 1 (that's why conv1d)
    # words_conv = tf.layers.conv1d(dropout_emb, filters=filters, kernel_size=3,
    #                                   strides=stride, padding='SAME', activation=tf.nn.relu)


    #------- Simple Version -----------------------
    words_conv = tf.layers.conv1d(dropout_emb, filters=filters, kernel_size=3,
                                     strides=stride, padding='SAME', activation=tf.nn.relu)
    words_conv_shape = words_conv.get_shape()
    dim = words_conv_shape[1] * words_conv_shape[2]
    input_layer = tf.reshape(words_conv,[-1, dim])

    #------- Version with 3 conv1d concatenated ----------------

    # words_conv_1 = tf.layers.conv1d(dropout_emb, filters=filters, kernel_size=4,
    #                               strides=stride, padding='SAME', activation=tf.nn.relu)
    # words_conv_2 = tf.layers.conv1d(dropout_emb, filters=filters, kernel_size=3,
    #                               strides=stride, padding='SAME', activation=tf.nn.relu)
    # words_conv_3 = tf.layers.conv1d(dropout_emb, filters=filters, kernel_size=2,
    #                               strides=stride, padding='SAME', activation=tf.nn.relu)
    #
    # max_conv_1 = tf.reduce_max(input_tensor=words_conv_1, axis=1)
    # max_conv_2 = tf.reduce_max(input_tensor=words_conv_2, axis=1)
    # max_conv_3 = tf.reduce_max(input_tensor=words_conv_3, axis=1)
    #
    # words_conv = tf.concat([max_conv_1, max_conv_2,max_conv_3], 1)
    # #print('----------------------------- 2', words_conv.get_shape())
    # input_layer = words_conv
    #
    # if pm.PRINT_SHAPE:
    #     print('----------------------------->  words_conv_1:', words_conv_1.get_shape())
    #     print('----------------------------->  max_conv_1:', max_conv_1.get_shape())
    #     print('----------------------------->  words_conv_2:', words_conv_2.get_shape())
    #     print('----------------------------->  max_conv_2:', max_conv_2.get_shape())
    #     print('----------------------------->  words_conv_3:', words_conv_3.get_shape())
    #     print('----------------------------->  max_conv_3:', max_conv_3.get_shape())

    if hidden_units is not None:

        # Create a fully-connected layer-stack based on the hidden_units in the params
        hidden_layers = tf.contrib.layers.stack(inputs=input_layer,
                                                layer=tf.contrib.layers.fully_connected,
                                                stack_args= hidden_units,
                                                activation_fn=tf.nn.relu)

        hidden_layers = tf.layers.dropout(inputs=hidden_layers, rate=dropout_rate, training=training)
        # print("hidden_layers: {}".format(hidden_layers)) # (?, last-hidden-layer-size)

    else:
        hidden_layers = input_layer

    # Connect the output layer (logits) to the hidden layer (no activation fn)
    logits = tf.layers.dense(inputs=hidden_layers,
                             units=output_layer_size,
                             activation=None)
    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Convert predicted_indices back into strings
        predictions = {
            'class': tf.gather(pm.TARGET_LABELS, predicted_indices),
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }

        # Provide an estimator spec for `ModeKeys.PREDICT` modes.
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)

    # Calculate loss using softmax cross entropy
    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

    tf.summary.scalar('loss', loss)

    probabilities = tf.nn.softmax(logits)
    predicted_indices = tf.argmax(probabilities, 1)

    accuracy = tf.metrics.accuracy(labels, predicted_indices)


    if mode == tf.estimator.ModeKeys.TRAIN:

        one_epoch_in_step = pm.TRAIN_SIZE/pm.BATCH_SIZE

        if pm.DECAY_LEARNING_RATE_ACTIVE:
            decayed_learning_rate = tf.train.exponential_decay(learning_rate,
                                            global_step=tf.train.get_global_step(),
                                            decay_steps=20*one_epoch_in_step, decay_rate=0.90, staircase=True)
            tf.summary.scalar('learning rate', decayed_learning_rate)
        else:
            decayed_learning_rate = learning_rate

        # Create Optimiser
        optimizer = tf.train.AdamOptimizer(decayed_learning_rate)

        # Create training operation
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        tf.summary.scalar('accuracy', accuracy[1])


        # Provide an estimator spec for `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Return accuracy and area under ROC curve metrics
        labels_one_hot = tf.one_hot(
            labels,
            depth=len(pm.TARGET_LABELS),
            on_value=True,
            off_value=False,
            dtype=tf.bool
        )

        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predicted_indices) ,#, weights=weights),
            'auroc': tf.metrics.auc(labels_one_hot, probabilities) # , weights=weights)
        }

        # Provide an estimator spec for `ModeKeys.EVAL` modes.
        return tf.estimator.EstimatorSpec(mode, loss=loss,eval_metric_ops=eval_metric_ops)


def cnn_lstm_model_fn(features, labels, mode, params):

    hidden_units = params.hidden_units
    output_layer_size = len(pm.TARGET_LABELS)
    embedding_size = params.embedding_size
    embedding_initializer = params.embedding_initializer
    forget_bias = params.forget_bias
    learning_rate = params.learning_rate

    window_size = params.window_size
    dropout_rate = params.dropout_rate
    stride = int(window_size/2)
    filters = params.filters


    # word_id_vector
    word_id_vector = preprocessing.process_text(features[pm.TEXT_FEATURE_NAME])

    # feature_length_array = [len(np.nonzero(word_id_vector[i])[0]) for i in range(BATCH_SIZE)]
    feature_length_array = tf.count_nonzero(word_id_vector, 1)

    # layer to take each word_id and convert it into vector (embeddings)
    word_embeddings = tf.contrib.layers.embed_sequence(word_id_vector,
                                                 vocab_size=pm.N_WORDS,
                                                 embed_dim=embedding_size,
                                                 initializer=embedding_initializer,
                                                 trainable=pm.TRAINABLE_EMB)

    training = (mode == tf.estimator.ModeKeys.TRAIN)
    dropout_emb = tf.layers.dropout(inputs=word_embeddings, rate=0.2, training=training)


    words_conv= tf.layers.conv1d(dropout_emb, filters=filters, kernel_size=window_size ,
                                  strides=stride, padding='SAME', activation=tf.nn.relu)
    #words_conv= tf.layers.conv1d(words_conv, filters=filters/2, kernel_size=window_size ,
                                      #strides=stride, padding='SAME', activation=tf.nn.relu)
    words_conv = tf.layers.dropout(inputs=words_conv, rate=dropout_rate, training=training)
    #print('----------------------------- 1', words_conv.get_shape())
    #words_conv= tf.layers.conv1d(words_conv, filters=filters/2, kernel_size=window_size ,
                                      #strides=stride, padding='SAME', activation=tf.nn.relu)
    #words_conv = tf.layers.dropout(inputs=words_conv, rate=dropout_rate, training=training)


    # words_conv_shape = words_conv.get_shape()
    # dim = words_conv_shape[1] * words_conv_shape[2]
    # input_layer = tf.reshape(words_conv,[-1, dim])

    # rnn_layers = [tf.nn.rnn_cell.LSTMCell(
    #         num_units=size,
    #         forget_bias=forget_bias,
    #         activation=tf.nn.tanh) for size in hidden_units]

    rnn_layers = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(
            num_units=size,
            forget_bias=forget_bias,
            activation=tf.nn.tanh), output_keep_prob=1.0,state_keep_prob=1.0,
                ) for size in hidden_units]

    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    input_layer = words_conv

    _, final_states = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=input_layer,
                                   sequence_length=feature_length_array,
                                   dtype=tf.float32)

    # slice to keep only the last cell of the RNN
    rnn_output = final_states[-1].h

    if pm.PRINT_SHAPE:
        print('----------- word_embeddings.shape -------------', dropout_emb.get_shape())
        print('--------------- conv1d_shape --------------',words_conv.get_shape())
        print('------------- rnn_output ---------------',rnn_output.get_shape())

    # input_layer = rnn_output
    #
    # hidden_layers = tf.contrib.layers.stack(inputs=input_layer,
    #                                             layer=tf.contrib.layers.fully_connected,
    #                                             stack_args= [32,16],
    #                                             activation_fn=tf.nn.relu)
    #
    # hidden_layers = tf.layers.dropout(inputs=hidden_layers, rate=0.2, training=training)
    #rnn_output = tf.layers.dropout(inputs=rnn_output, rate=0.5, training=training)
    pre_logits = rnn_output

    # Connect the output layer (logits) to the hidden layer (no activation fn)
    logits = tf.layers.dense(inputs=pre_logits,
                             units=output_layer_size,
                             activation=None)
    #kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        #print("-------------------- Predicting ---------------------")
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Convert predicted_indices back into strings
        predictions = {
            'class': tf.gather(pm.TARGET_LABELS, predicted_indices),
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }

        # Provide an estimator spec for `ModeKeys.PREDICT` modes.
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)

    # weights
    #weights = features[WEIGHT_COLUNM_NAME]

    # Calculate loss using softmax cross entropy
    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

    #l2_loss = tf.losses.get_regularization_loss()
    #loss += l2_loss

    #loss = tf.losses.sparse_softmax_cross_entropy(
    #    logits=logits, labels=labels,
    #    weights=weights)
    #
    tf.summary.scalar('loss', loss)

    probabilities = tf.nn.softmax(logits)
    predicted_indices = tf.argmax(probabilities, 1)

    accuracy = tf.metrics.accuracy(labels, predicted_indices)
    #tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Create Optimiser
        one_epoch_in_step = int(pm.TRAIN_SIZE/pm.BATCH_SIZE)
        #decay_after_epoch = tf.cast(10*one_epoch_in_step, tf.int64)
        one_epoch_in_step = pm.TRAIN_SIZE/pm.BATCH_SIZE
        learning_func = tf.random_uniform([1],minval=learning_rate,maxval=0.0005)[0]
        #learning_func = tf.cond(tf.less(tf.train.get_global_step(), tf.cast(5*one_epoch_in_step, tf.int64)), lambda: 0.0008, lambda: learning_func)
        #learning_func = tf.cond(tf.less(tf.train.get_global_step(), tf.cast(3*one_epoch_in_step, tf.int64)), lambda: 0.008, lambda: learning_func)
        learning_func = tf.cond(tf.less(tf.train.get_global_step(), tf.cast(3*one_epoch_in_step, tf.int64)), lambda: 0.001, lambda: learning_func)
        #learning_func = tf.random_uniform([1],minval=learning_rate,maxval=0.0001)[0]
        #learning_func = tf.cond(tf.less(tf.train.get_global_step(), tf.cast(5*one_epoch_in_step, tf.int64)), lambda: 0.0001, lambda: learning_func)
        # learning_func = tf.cond(tf.less(tf.train.get_global_step(), tf.cast(3*one_epoch_in_step, tf.int64)), lambda: 0.008, lambda: learning_func)
        # learning_func = tf.cond(tf.less(tf.train.get_global_step(), tf.cast(1*one_epoch_in_step, tf.int64)), lambda: 0.05, lambda: learning_func)
        if pm.DECAY_LEARNING_RATE_ACTIVE:
            decayed_learning_rate = tf.train.exponential_decay(learning_rate=learning_func,
                                            global_step=tf.train.get_global_step(),
                                            decay_steps=10*one_epoch_in_step,
                                            decay_rate=0.90,
                                            staircase=True)
            tf.summary.scalar('learning rate', decayed_learning_rate)
        else:
            decayed_learning_rate = learning_rate

        # Create Optimiser
        optimizer = tf.train.AdamOptimizer(decayed_learning_rate)

        # Create training operation
        #train_op = optimizer.minimize(
        #        loss=loss, global_step=tf.train.get_global_step())
        #probabilities = tf.nn.softmax(logits)
        #predicted_indices = tf.argmax(probabilities, 1)

        train_op =  optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step()) #, weights=weights),

        #accuracy = tf.metrics.accuracy(labels, predicted_indices)
        tf.summary.scalar('accuracy', accuracy[1])

        # Provide an estimator spec for `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        #probabilities = tf.nn.softmax(logits)
        #predicted_indices = tf.argmax(probabilities, 1)

        # Return accuracy and area under ROC curve metrics
        labels_one_hot = tf.one_hot(
            labels,
            depth=len(pm.TARGET_LABELS),
            on_value=True,
            off_value=False,
            dtype=tf.bool
        )

        eval_metric_ops = {
            'accuracy': accuracy, #tf.metrics.accuracy(labels, predicted_indices) ,#, weights=weights),
            'auroc': tf.metrics.auc(labels_one_hot, probabilities) # , weights=weights)
        }
        tf.summary.scalar('accuracy', accuracy[1])
        # Provide an estimator spec for `ModeKeys.EVAL` modes.
        return tf.estimator.EstimatorSpec(mode, loss=loss,eval_metric_ops=eval_metric_ops)

import tensorflow as tf
import shutil
import numpy as np
import datetime
import os, sys
from gensim.models import Word2Vec


import preprocessing
import params as pm

# Usage
# $:python3 estimator_dyn_glove.py 'model-name' 'checkpoint-path' 'train-data-path'

# Main script

tf.logging.set_verbosity(tf.logging.INFO)


def create_estimator(run_config, hparams):
    estimator = tf.estimator.Estimator(model_fn=pm.MODEL_FN,
                                  params=hparams,
                                  config=run_config)

    print("")
    print("Estimator Type: {}".format(type(estimator)))
    print("")
    return estimator

# test

def serving_input_fn():
    # At serving time, it accepts inference requests and prepares them for the model.
    receiver_tensor = {
      'sentence': tf.placeholder(tf.string, [None]),
      #'length': tf.placeholder(tf.int64, [None]),
    }

    features = {
      key: tensor
      for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(
        features, receiver_tensor)


if __name__ == "__main__":

    word_index = preprocessing.get_word_index_dict()
    # link: http://hlt.isti.cnr.it/wordembeddings/
    GLOVE_PATH = '/home/asr/nlp_lab/glove-ita/glove_WIKI'
    model = Word2Vec.load(GLOVE_PATH)



    def my_initializer(shape=None, dtype=tf.float32, partition_info=None, glove_activation=False):
        assert dtype is tf.float32
        embedding_matrix = preprocessing.load_glove_embeddings(word_index, model)
        return embedding_matrix


    hparams  = tf.contrib.training.HParams(
        num_epochs = pm.NUM_EPOCHS,
        batch_size = pm.BATCH_SIZE,
        embedding_size = pm.EMBEDDING_SIZE,
        max_steps = pm.TOTAL_STEPS,
        learning_rate = pm.LEARNING_RATE,
        embedding_initializer = my_initializer if pm.GLOVE_ACTIVE else None,
        forget_bias=pm.FORGET_BIAS,
        dropout_rate = pm.DROPOUT_RATE,
        hidden_units=pm.HIDDEN_UNITS,
        window_size = pm.WINDOW_SIZE,
        filters = pm.FILTERS,)

    run_config = tf.estimator.RunConfig(
        log_step_count_steps=5000,
        tf_random_seed=19830610,
        model_dir=pm.MODEL_DIR)

    # train_spec = tf.estimator.TrainSpec(
    #     input_fn = lambda: preprocessing.input_fn(
    #         pm.TRAIN_DATA_FILES_PATTERN,
    #         mode = tf.estimator.ModeKeys.TRAIN,
    #         num_epochs=hparams.num_epochs,
    #         batch_size=hparams.batch_size),
    #     max_steps=hparams.max_steps,
    #     hooks=None)

    train_spec = tf.estimator.TrainSpec(
        input_fn = lambda: preprocessing.input_fn(
            pm.TRAIN_DATA_FILES_PATTERN,
            mode = tf.estimator.ModeKeys.TRAIN,
            batch_size=hparams.batch_size),
        max_steps=hparams.max_steps,
        hooks=None)

    eval_spec = tf.estimator.EvalSpec(
        input_fn = lambda: preprocessing.input_fn(
            pm.VALID_DATA_FILES_PATTERN,
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=512),
        exporters=[tf.estimator.LatestExporter(  # this class regularly exports the serving graph and checkpoints.
            name="predict", # the name of the folder in which the model will be exported to under export
            serving_input_receiver_fn=serving_input_fn,
            exports_to_keep=1,
            as_text=True)],
        steps=None,
        throttle_secs = pm.EVAL_AFTER_SEC)


    if not pm.RESUME_TRAINING:
        print("Removing previous artifacts...")
        shutil.rmtree(pm.MODEL_DIR, ignore_errors=True)
    else:
        print("Resuming traconstraintining...")

    time_start = datetime.datetime.utcnow()
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print(".......................................")

    estimator = create_estimator(run_config, hparams)

    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec)

    time_end = datetime.datetime.utcnow()
    print(".......................................")
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    print("")
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))

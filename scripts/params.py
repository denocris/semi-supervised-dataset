import os, sys
import models
import tensorflow as tf


# ------------------------------------
# ------------ ANN MODEL -------------
# ------------------------------------
##
MODEL_FN = models.lstm_model_fn
#MODEL_FN = models.cnn_model_fn
#MODEL_FN = models.cnn_lstm_model_fn


PRINT_SHAPE = False
# ------------------------------------
# ----- SETUP TRAINING----------------
# ------------------------------------
RESUME_TRAINING = False
MULTI_THREADING = True
TRAIN_SIZE = int(str(sys.argv[4]))
NUM_EPOCHS = 3
BATCH_SIZE = 250
EVAL_AFTER_SEC = 30
TOTAL_STEPS = int((TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS)

# ------------------------------------
# ------------- METADATA -------------
# ------------------------------------
PAD_WORD = '#=KS=#'
HEADER = ['sentence', 'class']
HEADER_DEFAULTS = [['NA'], ['NA']]
TEXT_FEATURE_NAME = 'sentence'
TARGET_NAME = 'class'
TARGET_LABELS = ['0', '1']
#WEIGHT_COLUNM_NAME = 'weight'

# ------------------------------------
# ----------- GLOVE EMBEDDING --------
# ------------------------------------
GLOVE_ACTIVE=False
TRAINABLE_EMB=True
MAX_DOCUMENT_LENGTH = 20
EMBEDDING_SIZE = 300 if GLOVE_ACTIVE else 5

# ------------------------------------
# ------------- TRAINING PARAMS ------------
# ------------------------------------
LEARNING_RATE = 0.0001 #0.00001 #0.01
DECAY_LEARNING_RATE_ACTIVE = True
# For LSTM0
FORGET_BIAS = 1.0
# For LSTM0
DROPOUT_RATE = 0.5 #0.5
# For LSTM it refers to the size of the Cell, for CNN model instead are the FC layers
HIDDEN_UNITS = [16] #[96, 64, 16], None
# For CNN, kernel size
WINDOW_SIZE = 3
# For CNN, number of filters (i.e. feature maps)
FILTERS = 16

# ------------------------------------
# ------------- MODEL DIR ------------
# ------------------------------------
MODEL_NAME = str(sys.argv[1])
#MODEL_DIR = os.path.join(os.getcwd(),'/home/asr/rd_ssDataLearning/trained_ssDataLearning/{}'.format(MODEL_NAME))
#MODEL_DIR = os.path.join(os.getcwd(),str(sys.argv[2])+'{}'.format(MODEL_NAME))
MODEL_DIR = str(sys.argv[2])+'{}'.format(MODEL_NAME)

TRAIN_DATA_PATH = str(sys.argv[3])


# ------------------------------------
# ------- TRAIN & VALID PATH ---------
# ------------------------------------
# TRAIN_DATA_FILES_PATTERN = os.path.join(os.getcwd(),'data_calendar/train-data-maxlength16-subtitles.tsv')
# VALID_DATA_FILES_PATTERN = os.path.join(os.getcwd(),'data_calendar/test-data-maxlength16-subtitles.tsv')
# VOCAB_LIST_FILE = os.path.join(os.getcwd(),'data_calendar/vocab_list_13k_2k_mystop_nodgts.tsv')
# N_WORDS_FILE = os.path.join(os.getcwd(),'data_calendar/n_words_13k_2k_mystop_nodgts.tsv')

#TRAIN_DATA_FILES_PATTERN = os.path.join(os.getcwd(),'/home/asr/rd_ssDataLearning/dataset/train-data-maxlength16-subtitles.tsv')
#TRAIN_DATA_FILES_PATTERN = os.path.join(os.getcwd(),'/home/asr/rd_ssDataLearning/dataset/red3k-train_v0.tsv')


# TRAIN_DATA_FILES_PATTERN = os.path.join(os.getcwd(),TRAIN_DATA_PATH)
# VALID_DATA_FILES_PATTERN = os.path.join(os.getcwd(),'/home/asr/rd_ssDataLearning/calendar_set/dataset/test-data-maxlength16-subtitles.tsv')
# VOCAB_LIST_FILE = os.path.join(os.getcwd(),'/home/asr/rd_ssDataLearning/calendar_set/dataset/vocab_5k.tsv')
# N_WORDS_FILE = os.path.join(os.getcwd(),'/home/asr/rd_ssDataLearning/calendar_set/dataset/n_words_5k.tsv')

TRAIN_DATA_FILES_PATTERN = os.path.join(os.getcwd(),TRAIN_DATA_PATH)
VALID_DATA_FILES_PATTERN = os.path.join(os.getcwd(),'/home/asr/rd_ssDataLearning/six_set/dataset/test_sei_sslearn.tsv')
VOCAB_LIST_FILE = os.path.join(os.getcwd(),'/home/asr/rd_ssDataLearning/six_set/dataset/vocab_5k.tsv')
N_WORDS_FILE = os.path.join(os.getcwd(),'/home/asr/rd_ssDataLearning/six_set/dataset/n_words_5k.tsv')


# TRAIN_DATA_FILES_PATTERN = os.path.join(os.getcwd(),'data_six/train_data_length3-16_v4.tsv')
# VALID_DATA_FILES_PATTERN = os.path.join(os.getcwd(),'data_six/valid_data_length3-16_v4.tsv')
# VOCAB_LIST_FILE = os.path.join(os.getcwd(),'data_six/vocab_list_20k_v4.tsv')
# N_WORDS_FILE = os.path.join(os.getcwd(),'data_six/n_words_20k_v4.tsv')
#TRAIN_DATA_FILES_PATTERN = os.path.join(os.getcwd(),'data_e/train_data_length16.tsv')
#VALID_DATA_FILES_PATTERN = os.path.join(os.getcwd(),'data_e/valid_data_length16.tsv')
#VOCAB_LIST_FILE = os.path.join(os.getcwd(),'data_e/vocab_list_20k.tsv')
#N_WORDS_FILE = os.path.join(os.getcwd(),'data_e/n_words_20k.tsv')
with open(N_WORDS_FILE) as file:
    N_WORDS = int(file.read())+2

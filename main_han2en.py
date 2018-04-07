import tensorflow as tf
import helper
import hangul
from seq2seq import Seq2SeqModel

# Source data: Hangul
# Target data: English

chosungs = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsungs = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ',
             'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

jongsungs = [' ', '!', '@', '#', '$', '%', '^', '&']

jaso = chosungs + jungsungs + jongsungs

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


# Build int2letter and letter2int dicts
source_letter_to_int = helper.extract_letter_to_int(jaso)
target_letter_to_int = helper.extract_letter_to_int(alphabet, is_target=True)

print(len(source_letter_to_int), len(target_letter_to_int))

# Load data
train_hangul = helper.load_data('data/train/train_hangul.txt')
train_english = helper.load_data('data/train/train_english.txt')
valid_hangul = helper.load_data('data/valid/valid_hangul.txt')
valid_english = helper.load_data('data/valid/valid_english.txt')
test_hangul = helper.load_data('data/test/test_hangul.txt')
test_english = helper.load_data('data/test/test_english.txt')


# Data set size
print('training data size:', len(train_hangul), len(train_english))
print('validation data size: ', len(valid_hangul), len(valid_english))
print('test data size:', len(test_hangul), len(test_english))

# Convert hangul words to hangul jaso
train_sources = [hangul.word_to_jaso(word) for word in train_hangul]
train_targets = train_english
valid_sources = [hangul.word_to_jaso(word) for word in valid_hangul]
valid_targets = valid_english
test_sources = [hangul.word_to_jaso(word) for word in test_hangul]
test_targets = test_english


# Model parameter
num_units = 128
num_layers = 2
cell = tf.contrib.rnn.GRUCell
attention = True

# Training parameter
epochs = 500
batch_size = 512
learning_rate = 0.01
keep_prob = 0.5
optimizer = tf.train.AdamOptimizer

with tf.Session() as sess:
    seq2seq_model = Seq2SeqModel(sess,
                                 cell,
                                 num_units,
                                 num_layers,
                                 source_letter_to_int,
                                 target_letter_to_int,
                                 attention)

    train_loss_history, valid_loss_history = seq2seq_model.train([train_sources, train_targets],
                                                                 [valid_sources, valid_targets],
                                                                 learning_rate,
                                                                 batch_size,
                                                                 epochs,
                                                                 keep_prob,
                                                                 optimizer,
                                                                 save_path='ckpt_dir/han_to_en/model',
                                                                 display_size=10,
                                                                 save_size=10)

    print(train_loss_history)
    print(valid_loss_history)

    pred_words = seq2seq_model.predict(test_sources)

    acc = 0
    for han, en, pred in zip(test_hangul, test_english, pred_words):
        if en == pred:
            acc += 1
        else:
            print(han, en, pred)

    print('Test accuracy:', acc / len(test_sources))

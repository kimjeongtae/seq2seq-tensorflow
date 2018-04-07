import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from helper import word_to_int
from helper import int_to_word
from helper import pad_sequence_batch
from helper import get_batches


# Embedding: one hot encoding
class Seq2SeqModel(object):
    def __init__(self, sess, cell, num_units, num_layers, source_letter_to_int, target_letter_to_int, attention=True):
        self.sess = sess
        self.cell = cell
        self.num_units = num_units
        self.num_layers = num_layers

        self.source_letter_to_int = source_letter_to_int
        self.target_letter_to_int = target_letter_to_int
        self.source_int_to_letter = {word_i: word for word, word_i in source_letter_to_int.items()}
        self.target_int_to_letter = {word_i: word for word, word_i in target_letter_to_int.items()}

        # source_PAD == target_PAD
        self.PAD = target_letter_to_int['<PAD>']
        self.GO = target_letter_to_int['<GO>']
        self.EOS = target_letter_to_int['<EOS>']

        self.attention = attention

        self._build_model()

    def _build_model(self):
        print('Building model...')

        self._build_placeholders()
        enc_outputs, enc_state = self._build_encoder()
        self._build_decoder(enc_outputs, enc_state)

    def _build_placeholders(self):
        self._sources = tf.placeholder(tf.int32, (None, None), name='sources')
        self._targets = tf.placeholder(tf.int32, (None, None), name='targets')
        self._source_sequence_lengths = tf.placeholder(tf.int32, (None,), name='source_sequence_lengths')
        self._target_sequence_lengths = tf.placeholder(tf.int32, (None,), name='target_sequence_lengths')
        self._keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def _build_encoder(self):
        enc_inputs_size = len(self.source_letter_to_int)
        enc_one_hot_inputs = tf.one_hot(self._sources, enc_inputs_size)
        enc_cell = tf.contrib.rnn.MultiRNNCell(
            [self._dropout_cell() for _ in range(self.num_layers - 1)] + [self.cell(self.num_units)])
        enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=enc_cell,
                                                   inputs=enc_one_hot_inputs,
                                                   sequence_length=self._source_sequence_lengths,
                                                   dtype=tf.float32)

        return enc_outputs, enc_state

    def _build_decoder(self, enc_outputs, enc_state):
        dec_inputs_size = len(self.target_letter_to_int)
        batch_size = tf.shape(self._sources)[0]
        max_target_sequence_length = tf.reduce_max(self._target_sequence_lengths, name='max_target_sequence_length')

        def process_decoder_inputs():
            ending = tf.strided_slice(self._targets, [0, 0], [batch_size, -1], [1, 1])
            return tf.concat([tf.fill([batch_size, 1], self.GO), ending], 1)

        dec_one_hot_inputs = tf.one_hot(process_decoder_inputs(), dec_inputs_size)

        output_layer = Dense(dec_inputs_size, name='output_projection')

        dec_cell = tf.contrib.rnn.MultiRNNCell(
            [self._dropout_cell() for _ in range(self.num_layers-1)] + [self.cell(self.num_units)])

        if self.attention:
            print('Attention Seq2Seq...')
            attention_mechanism = \
                tf.contrib.seq2seq.BahdanauAttention(num_units=self.num_units,
                                                     memory=enc_outputs,
                                                     memory_sequence_length=self._source_sequence_lengths)

            dec_cell = tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell,
                                                           attention_mechanism=attention_mechanism,
                                                           attention_layer_size=self.num_units,
                                                           output_attention=False,
                                                           initial_cell_state=enc_state)

            initial_state = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        else:
            print('Basic Seq2Seq...')
            initial_state = enc_state

        # Build training decoder
        with tf.variable_scope('decode'):
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_one_hot_inputs,
                                                                sequence_length=self._target_sequence_lengths,
                                                                time_major=False)

            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                               helper=training_helper,
                                                               initial_state=initial_state,
                                                               output_layer=output_layer)

            training_decoder_outputs, _, _ = \
                tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                  impute_finished=True,
                                                  maximum_iterations=max_target_sequence_length)

        # Loss
        logits = tf.identity(training_decoder_outputs.rnn_output, name='logits')
        masks = tf.sequence_mask(self._target_sequence_lengths, max_target_sequence_length, tf.float32, name='masks')
        self._loss = tf.contrib.seq2seq.sequence_loss(logits, self._targets, masks, name='loss')

        # Build inference decoder
        with tf.variable_scope('decode', reuse=True):
            start_tokens = tf.tile(tf.constant([self.GO], dtype=tf.int32), [batch_size])

            inference_helper = \
                tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=lambda x: tf.one_hot(x, dec_inputs_size),
                                                         start_tokens=start_tokens,
                                                         end_token=self.EOS)

            inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                                helper=inference_helper,
                                                                initial_state=initial_state,
                                                                output_layer=output_layer)

            inference_decoder_outputs, _, self._inference_sequence_lengths = \
                tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                  impute_finished=True,
                                                  maximum_iterations=None)

        # Predictions
        self._predictions = tf.identity(inference_decoder_outputs.sample_id, name='predictions')

    def _dropout_cell(self):
        return tf.contrib.rnn.DropoutWrapper(self.cell(self.num_units), output_keep_prob=self._keep_prob)

    def train(self, train_data, valid_data=None, learning_rate=0.001, batch_size=128, epochs=100, keep_prob=0.5,
              optimizer=tf.train.AdamOptimizer, save_path='', load_path='', display_size=5, save_size=5):
        print('Prediction...')

        train_sources, train_targets = train_data

        # Convert letters to ids
        train_sources_ids = [word_to_int(word, self.source_letter_to_int) for word in train_sources]
        train_targets_ids = [word_to_int(word, self.target_letter_to_int) + [self.EOS] for word in train_targets]

        train_sources_pad_ids, train_targets_pad_ids, train_source_sequence_lengths, train_target_sequence_lengths = \
            next(get_batches(train_sources_ids, train_targets_ids, self.PAD, len(train_sources_ids)))

        if valid_data is not None:
            valid_sources, valid_targets = valid_data
            valid_sources_ids = [word_to_int(word, self.source_letter_to_int) for word in valid_sources]
            valid_targets_ids = [word_to_int(word, self.target_letter_to_int) + [self.EOS] for word in valid_targets]
            valid_sources_pad_ids, valid_targets_pad_ids, valid_source_sequence_lengths, valid_target_sequence_lengths \
                = next(get_batches(valid_sources_ids, valid_targets_ids, self.PAD, len(valid_sources_ids)))

        update = optimizer(learning_rate).minimize(self._loss)

        if load_path:
            self.load(load_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        train_loss_history = []
        valid_loss_history = []

        for epoch_i in range(1, epochs+1):
            for sources_batch, targets_batch, source_batch_sequence_lengths, target_batch_sequence_lengths \
                    in get_batches(train_sources_ids, train_targets_ids, self.PAD, batch_size):

                self.sess.run(update, feed_dict={self._sources: sources_batch,
                                                 self._targets: targets_batch,
                                                 self._source_sequence_lengths: source_batch_sequence_lengths,
                                                 self._target_sequence_lengths: target_batch_sequence_lengths,
                                                 self._keep_prob: keep_prob})

            if epoch_i % display_size == 0:
                train_loss = self.sess.run(self._loss,
                                           feed_dict={self._sources: train_sources_pad_ids,
                                                      self._targets: train_targets_pad_ids,
                                                      self._source_sequence_lengths: train_source_sequence_lengths,
                                                      self._target_sequence_lengths: train_target_sequence_lengths,
                                                      self._keep_prob: 1.0})

                train_loss_history.append(train_loss)
                if valid_data is not None:
                    valid_loss = self.sess.run(self._loss,
                                               feed_dict={self._sources: valid_sources_pad_ids,
                                                          self._targets: valid_targets_pad_ids,
                                                          self._source_sequence_lengths: valid_source_sequence_lengths,
                                                          self._target_sequence_lengths: valid_target_sequence_lengths,
                                                          self._keep_prob: 1.0})

                    valid_loss_history.append(valid_loss)
                    print('Epoch {:>3}/{} Training loss: {:>6.3f}  - Validation loss: {:>6.3f}'.
                          format(epoch_i, epochs, train_loss, valid_loss))

                    print(train_loss_history)
                    print(valid_loss_history)

                else:
                    print('Epoch {:>3}/{} Training loss: {:>6.3f}'.format(epoch_i, epochs, train_loss))

            if save_path and save_size and epoch_i % save_size == 0:
                self.save(save_path, epoch_i)

        return train_loss_history, valid_loss_history

    def predict(self, sequences, load_path=''):
        print('Trai start...')
        if load_path:
            self.load(load_path)

        sequence_lengths = np.array([len(sequence) for sequence in sequences])
        # Convert letters to ids
        sequences_ids = [word_to_int(sequence, self.source_letter_to_int) for sequence in sequences]
        pad_sequences_ids = np.array(pad_sequence_batch(sequences_ids, self.PAD))

        outputs, output_sequence_lengths = self.sess.run([self._predictions, self._inference_sequence_lengths],
                                                         feed_dict={self._sources: pad_sequences_ids,
                                                                    self._source_sequence_lengths: sequence_lengths,
                                                                    self._keep_prob: 1.0})

        # Remove <EOS> and <PAD>
        outputs = [output[:sequence_length-1] for output, sequence_length in zip(outputs, output_sequence_lengths)]
        # Convert ids to words
        return [int_to_word(output, self.target_int_to_letter) for output in outputs]

    def save(self, save_path, global_step=None):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path, global_step, write_meta_graph=False)
        print('Model save at ' + save_path + '-' + str(global_step))

    def load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        print('Model restored from' + path)









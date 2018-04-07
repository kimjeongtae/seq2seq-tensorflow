import numpy as np


def extract_letter_to_int(letter_list, is_target=False):
    special_letter = ['<PAD>', '<GO>', '<EOS>'] if is_target else ['<PAD>']
    return {word: word_i for word_i, word in enumerate(special_letter + letter_list)}


def word_to_int(word, letter_to_int):
    return [letter_to_int[letter] for letter in word]


def int_to_word(indices, int_to_letter):
    return ''.join(int_to_letter[i] for i in indices)


def load_data(path):
    with open(path, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()
    return data.strip().split('\n')


def save_data(data, path):
    with open(path, "w", encoding='utf-8', errors='ignore') as f:
        f.write('\n'.join(data))


def pad_sequence_batch(sequence_batch, pad_int, max_sequence_length=None):
    """Pad sequences with <PAD> so that each sequence of a batch has the same length"""
    if max_sequence_length is None:
        max_sequence_length = max([len(sequence) for sequence in sequence_batch])
    return [sequence + [pad_int] * (max_sequence_length - len(sequence)) for sequence in sequence_batch]


def get_batches(sources, targets, pad_int, batch_size):
    """Batch sources, targets and the lengths of their letters together"""

    for batch_i in range(len(sources) // batch_size):
        start_i = batch_i * batch_size

        sources_batch_ids = sources[start_i:start_i + batch_size]
        targets_batch_ids = targets[start_i:start_i + batch_size]

        source_batch_lengths = np.array([len(batch) for batch in sources_batch_ids])
        target_batch_lengths = np.array([len(batch) for batch in targets_batch_ids])

        # Batch sources, targets and the lengths of their letters together
        pad_sources_batch = np.array(pad_sequence_batch(sources_batch_ids, pad_int, max(source_batch_lengths)))
        pad_targets_batch = np.array(pad_sequence_batch(targets_batch_ids, pad_int, max(target_batch_lengths)))

        yield pad_sources_batch, pad_targets_batch, source_batch_lengths, target_batch_lengths


def levenshtein(string1, string2):
    length1 = len(string1)
    length2 = len(string2)
    distance = np.zeros((length1+1, length2+1), np.int32)
    distance[:, 0] = range(0, length1+1)
    distance[0, :] = range(0, length1+1)

    for i, char1 in enumerate(string1, 1):
        for j, char2 in enumerate(string2, 1):
            if char1 == char2:
                distance[i, j] = distance[i - 1, j - 1]
            else:
                distance[i, j] = min(distance[i - 1, j] + 1, distance[i, j - 1] + 1, distance[i - 1, j - 1] + 1)

    return distance[length2, length1]
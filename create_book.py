import numpy
import tensorflow
import glob
import codecs
import pickle
import time
import os
import datetime
import language_check


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text from the books split into words
    :return: A tuple of dicts
    """
    vocabulary = set(text)
    int_to_vocab = {key: word for key, word in enumerate(vocabulary)}
    vocab_to_int = {word: key for key, word in enumerate(vocabulary)}
    return vocab_to_int, int_to_vocab


def token_lookup():
    """
    Generate a dict to map punctuation into a token
    :return: dictionary mapping punctuation to token
    """
    return {
        '.': '||period||',
        ',': '||comma||',
        '"': '||quotes||',
        ';': '||semicolon||',
        '!': '||exclamation-mark||',
        '?': '||question-mark||',
        '(': '||left-parentheses||',
        ')': '||right-parentheses||',
        '--': '||emm-dash||',
        '\n': '||return||'
    }


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target data
    :param int_text: text with words replaced by their ids
    :param batch_size: the size that each batch of data should be
    :param seq_length: the length of each sequence
    :return: batches of data as a numpy array
    """
    words_per_batch = batch_size * seq_length
    num_batches = len(int_text)//words_per_batch
    if num_batches == 0:
        num_batches = 1
    int_text = int_text[:num_batches * words_per_batch]
    y = numpy.array(int_text[1:] + [int_text[0]])
    x = numpy.array(int_text)

    x_batches = numpy.split(x.reshape(batch_size, -1), num_batches, axis=1)
    y_batches = numpy.split(y.reshape(batch_size, -1), num_batches, axis=1)

    batch_data = list(zip(x_batches, y_batches))

    return numpy.array(batch_data)


def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word with some randomness
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the
    values
    :return: String of the predicted word
    """
    return numpy.random.choice(
        list(int_to_vocab.values()), 1, p=probabilities)[0]


def process_and_save(prime_words):
    # num_epochs = 2000
    batch_size = 250
    rnn_size = 256
    num_layers = 3
    keep_prob = 0.75
    embed_dim = 200
    seq_length = 50
    learning_rate = 0.001
    gen_length = 3000

    version_dir = './generated-book-v10'

    save_dir = os.path.abspath('save')

    book_files = sorted(glob.glob("data/*.txt"))

    print('found {} books'.format(len(book_files)))
    corpus_raw = u""
    for file in book_files:
        with codecs.open(file, 'r', encoding='utf-8') as book_file:
            corpus_raw += book_file.read()
    print('Corpus is {} words long'.format(len(corpus_raw)))

    token_dict = token_lookup()
    for token, replacement in token_dict.items():
        corpus_raw = corpus_raw.replace(token, ' {} '.format(replacement))
    corpus_raw = corpus_raw.lower()
    corpus_raw = corpus_raw.split()
    vocab_to_int, int_to_vocab = create_lookup_tables(corpus_raw)
    corpus_int = [vocab_to_int[word] for word in corpus_raw]
    pickle.dump((corpus_int, vocab_to_int, int_to_vocab, token_dict),
                open('preprocess.p', 'wb'))

    train_graph = tensorflow.Graph()
    with train_graph.as_default():
        # Initialize input placeholders
        input_text = tensorflow.placeholder(tensorflow.int32, [None, None],
                                            name='input')
        targets = tensorflow.placeholder(tensorflow.int32, [None, None],
                                         name='targets')
        lr = tensorflow.placeholder(tensorflow.float32, name='learning_rate')

        # Calculate text attributes
        vocab_size = len(int_to_vocab)
        input_text_shape = tensorflow.shape(input_text)

        # Build the RNN cell
        lstm = tensorflow.contrib.rnn.BasicLSTMCell(num_units=rnn_size)
        drop_cell = tensorflow.contrib.rnn.DropoutWrapper(
            lstm, output_keep_prob=keep_prob)
        cell = tensorflow.contrib.rnn.MultiRNNCell([drop_cell] * num_layers)

        # Set the initial state
        initial_state = cell.zero_state(input_text_shape[0],
                                        tensorflow.float32)
        initial_state = tensorflow.identity(initial_state,
                                            name='initial_state')

        # Create word embedding as input to RNN
        embed = tensorflow.contrib.layers.embed_sequence(input_text,
                                                         vocab_size,
                                                         embed_dim)

        # Build RNN
        outputs, final_state = tensorflow.nn.dynamic_rnn(
            cell, embed, dtype=tensorflow.float32)
        final_state = tensorflow.identity(final_state, name='final_state')

        # Take RNN output and make logits
        logits = tensorflow.contrib.layers.fully_connected(outputs, vocab_size,
                                                           activation_fn=None)

        # Calculate the probability of generating each word
        _ = tensorflow.nn.softmax(logits, name='probs')

        # Define loss function
        cost = tensorflow.contrib.seq2seq.sequence_loss(
            logits,
            targets,
            tensorflow.ones([input_text_shape[0], input_text_shape[1]])
        )

        # Learning rate optimizer
        optimizer = tensorflow.train.AdamOptimizer(learning_rate)

        # Gradient clipping to avoid exploding gradients
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tensorflow.clip_by_value(grad, -1., 1.), var) for
                            grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    pickle.dump((seq_length, save_dir), open(save_dir+'/'+'params.p', 'wb'))
    batches = get_batches(corpus_int, batch_size, seq_length)
    start_time = time.time()
    train_loss = 1
    epoch = 0
    batch_index = 0
    with tensorflow.Session(graph=train_graph) as sess:
        sess.run(tensorflow.global_variables_initializer())

        while train_loss > 0.1:
            epoch += 1
            state = sess.run(initial_state, {input_text: batches[0][0]})

            for batch_index, (x, y) in enumerate(batches):
                feed_dict = {
                    input_text: x,
                    targets: y,
                    initial_state: state,
                    lr: learning_rate
                }
                train_loss, state, _ = sess.run(
                    [cost, final_state, train_op], feed_dict)

            time_elapsed = time.time() - start_time
            print(
                'Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}   '
                'time_elapsed = {:.3f}'.format(
                    epoch + 1,
                    batch_index + 1,
                    len(batches),
                    train_loss,
                    time_elapsed))

            # save model every 5 epochs
            if epoch % 250 == 0:
                saver = tensorflow.train.Saver()
                saver.save(sess, save_dir)
                print('Model Trained and Saved')

    corpus_int, vocab_to_int, int_to_vocab, token_dict = pickle.load(
        open('preprocess.p', mode='rb'))
    seq_length, save_dir = pickle.load(open(
        save_dir+'/'+'params.p', mode='rb'))

    loaded_graph = tensorflow.Graph()

    with tensorflow.Session(graph=loaded_graph) as sess:
        # Load the saved model
        loader = tensorflow.train.import_meta_graph(save_dir + '.meta')
        loader.restore(sess, save_dir)

        # Get tensors from loaded graph
        input_text = loaded_graph.get_tensor_by_name('input:0')
        initial_state = loaded_graph.get_tensor_by_name('initial_state:0')
        final_state = loaded_graph.get_tensor_by_name('final_state:0')
        probs = loaded_graph.get_tensor_by_name('probs:0')

        # Sentences generation setup
        gen_sentences = prime_words.split()
        prev_state = sess.run(initial_state, {
            input_text: numpy.array([[1 for _ in gen_sentences]])})

        # Generate sentences
        for n in range(gen_length):
            # Dynamic Input
            dyn_input = [
                [vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
            dyn_seq_length = len(dyn_input[0])

            # Get Prediction
            probabilities, prev_state = sess.run(
                [probs, final_state],
                {input_text: dyn_input, initial_state: prev_state})

            pred_word = pick_word(probabilities[dyn_seq_length - 1],
                                  int_to_vocab)

            gen_sentences.append(pred_word)

        # Remove tokens
        chapter_text = ' '.join(gen_sentences)
        for key, token in token_dict.items():
            chapter_text = chapter_text.replace(' ' + token.lower(), key)

        # print(chapter_text)

    chapter_text = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        chapter_text = chapter_text.replace(' ' + token.lower(), key)
    chapter_text = chapter_text.replace('\n ', '\n')
    chapter_text = chapter_text.replace('( ', '(')
    chapter_text = chapter_text.replace(' ”', '”')

    if not os.path.exists(version_dir):
        os.makedirs(version_dir)

    num_chapters = len([name for name in os.listdir(version_dir) if
                        os.path.isfile(os.path.join(version_dir, name))])
    # next_chapter = version_dir + '/chapter-' + str(num_chapters + 1) + \
    #     '-not-grammar-corrected.md'
    # with open(next_chapter, "w", encoding='utf-8') as text_file:
    #     text_file.write(chapter_text)

    tool = language_check.LanguageTool('en-US')
    matches = tool.check(chapter_text)
    chapter_text = language_check.correct(chapter_text, matches)

    next_chapter = version_dir + '/chapter-' + str(num_chapters + 1) + '.md'
    with open(next_chapter, "w", encoding='utf-8') as text_file:
        text_file.write(chapter_text)


def run():
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    keywords_to_use = ['peace', 'war', 'reason', 'politics']
    # number of chapters written equals length of keywords
    for keyword in keywords_to_use:
        process_and_save(keyword)


if __name__ == '__main__':
    run()

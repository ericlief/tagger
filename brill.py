from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger, SequentialBackoffTagger, BrillTagger, brill_trainer, brill
import io, sys
from nltk.corpus import brown


def backoff_tagger(training_data, tagger_classes, backoff=None):
    for tagger in tagger_classes:
        backoff = tagger(training_data, backoff=backoff)
    return backoff

def train(train_data):

    # Instantiate initial backoff tagger (used as the base)
    backoff = DefaultTagger('NN')
    init_tagger = backoff_tagger(train_data, [UnigramTagger, BigramTagger, TrigramTagger], backoff)
    templates = brill.brill24()
    trainer = brill_trainer.BrillTaggerTrainer(init_tagger, templates, deterministic=True, trace=True)

    # Train
    brill_tagger = trainer.train(train_data)

    return brill_tagger

def evaluate(brill_tagger, file, title):
    with open(file, 'a') as f:
        res = brill_tagger.evaluate(test_data)
        print(title + '\t' + str(res))
        rules = brill_tagger.rules()
        # print(rules)
        f.write(title + '\n')
        f.write(str(res) + '\n')

if __name__ == '__main__':

    # out = '/home/liefe/Code/tagger/hw3-1-en.txt'
    out = '/home/liefe/Code/tagger/hw3-1-cz.txt'

    # with open(out, 'a') as f:
    #     f.write('test')

    # Read in data from std input, uncomment for cz versus en
    stream = io.TextIOWrapper(sys.stdin.buffer, encoding='iso-8859-2')
    # stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    words = []
    tags = []
    for line in stream:
        if line != '\n':
            # print(line)
            word, tag = line.rstrip().split('/')
            words.append(word)
            tags.append(tag)

    # Split data for initial part
    # Training-Heldout-Test
    # 20-20-40

    # Get test data, the last 40k
    test_data = []
    for word, tag in zip(words[-40000:], tags[-40000:]):
        test_data.append((word, tag))
    test_data = [test_data]     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

    # Get heldout data, the mid 20k (not used in brill).
    # Not used for this part, and just for show (not computed below)
    heldout_data = []
    for word, tag in zip(words[-60000:-40000], tags[-60000:-40000]):
        heldout_data.append((word, tag))

    # Get training data, the first ~20k
    train_data = []
    for word, tag in zip(words[:-60000], tags[:-60000]):
        train_data.append((word, tag))
    train_data = [train_data]     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

    # Train
    brill_tagger = train(train_data)
    title = 'Initial 20-20-40, Training-Heldout-Test, no sentence segmentation'
    evaluate(brill_tagger, out, title)

    # Now segment text into sentences, train and write results
    # Note that this had the effect of speeding up the training process significantly
    # probably due to the nltk implementation of the Brill tagger

    # Get test data, the last 40k
    test_data = []
    sent = []
    pos1 = -40000
    while words[pos1] != '###':
        pos1 += 1
    pos1 += 1  # skip first '#'
    for word, tag in zip(words[pos1:], tags[pos1:]):
        if word == '###':
            test_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))

    # Get heldout data, the mid 20k (not used in brill).
    heldout_data = []
    sent = []
    pos2 = -60000
    # pos2 = -40
    try:
        while words[pos2] != '###':
            pos2 += 1
    except IndexError:
        print(sys.stderr)
    pos2 += 1    # skip first '#'
    for word, tag in zip(words[pos2:pos1-1], tags[pos2:pos1-1]):
        if word == '###':
            heldout_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))

    # Get initial train data, the first ~20k
    train_data = []
    sent = []
    pos1 = 0
    try:
        while words[pos1] != '###':
            pos1 += 1
    except IndexError:
        print(sys.stderr)
    pos1 += 1  # this is the first sentence in set
    for word, tag in zip(words[pos1:pos2], tags[pos1:pos2]):
        if word == '###':
            train_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))    # heldout_data = [heldout_data]

    # Train
    brill_tagger = train(train_data)
    title = 'Initial 20-20-40, with sentence segmentation'
    evaluate(brill_tagger, out, title)



    # Cross Validate results, using 4 sample schemas
    
    # Initial Cross Validation
    # Test-Train-Heldout

    # Without sentence segmentation, uncomment for use

    # Get test/validation data
    test_data = []
    for word, tag in zip(words[:40000], tags[:40000]):
        test_data.append((word, tag))
    test_data = [test_data]     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

    # Get training data
    train_data = []
    for word, tag in zip(words[40000:60000], tags[40000:60000]):
        train_data.append((word, tag))
    train_data = [train_data]     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

    # Train
    brill_tagger = train(train_data)
    title = 'Initial cross Validation: Test-Train-Heldout, 40-20-20, no sentence segmentation'
    evaluate(brill_tagger, out, title)

    # With sentence segmentation, uncomment for use

    # Get test/validation data, the first 40k
    test_data = []
    sent = []
    pos1 = 0
    pos2 = 40000
    while words[pos1] != '###':
        pos1 += 1
    pos1 += 1  # skip first '#'
    while words[pos2] != '###':
        pos2 += 1
    for word, tag in zip(words[pos1:pos2], tags[pos1:pos2]):
        if word == '###':
            test_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))

    # Get training data
    train_data = []
    sent = []
    pos1 = pos2
    pos2 = 60000

    while words[pos1] != '###':
        pos1 += 1
    pos1 += 1  # this is the first sentence in set
    while words[pos2] != '###':
        pos2 += 1
    for word, tag in zip(words[pos1:pos2], tags[pos1:pos2]):
        if word == '###':
            train_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))    # heldout_data = [heldout_data]

    # Train
    brill_tagger = train(train_data)
    title = 'Initial cross Validation: Test-Train-Heldout, 40-20-20, with sentence segmentation'
    evaluate(brill_tagger, out, title)

    # Cross validate three more times
    # Splitting the data each time into
    # folds of 20-40-rest
    
    # Cross Validation #1
    # Heldout-Training-Test
    # 20-40-rest
    print('CV 1')


    # Without sentence segmentation, uncomment for use

    # Get train data
    train_data = []
    for word, tag in zip(words[20000:60000], tags[20000:60000]):
        train_data.append((word, tag))
    train_data = [train_data]  # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

    # Get test data
    test_data = []
    for word, tag in zip(words[60000:], tags[60000:]):
        test_data.append((word, tag))
    test_data = [test_data]     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

    # Train
    brill_tagger = train(train_data)
    title = '1st cross Validation: Heldout-Training-Test, 20-40-20, no sentence segmentation'
    evaluate(brill_tagger, out, title)

    # With sentence segmentation, uncomment for use

    # Get train data
    train_data = []
    sent = []
    pos1 = 20000
    pos2 = 60000
    while words[pos1] != '###':
        pos1 += 1
    pos1 += 1  # this is the first sentence in set
    while words[pos2] != '###':
        pos2 += 1
    for word, tag in zip(words[pos1:pos2], tags[pos1:pos2]):
        if word == '###':
            train_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))  # heldout_data = [heldout_data]

    # Get test data
    test_data = []
    sent = []
    pos1 = pos2
    while words[pos1] != '###':
        pos1 += 1
    pos1 += 1  # skip first '#'
    for word, tag in zip(words[pos1:], tags[pos1:]):
        if word == '###':
            test_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))

    # Train
    brill_tagger = train(train_data)
    title = '1st cross Validation: Heldout-Training-Test, 20-40-20, with sentence segmentation'
    evaluate(brill_tagger, out, title)

    # Cross Validation #2
    # Heldout-Test-Training
    # 20-40-rest
    print('CV 2')

    # Without sentence segmentation, uncomment for use

    # Get test data
    test_data = []
    for word, tag in zip(words[20000:60000], tags[20000:60000]):
        test_data.append((word, tag))
    test_data = [test_data]  # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

    # Get train data
    train_data = []
    for word, tag in zip(words[60000:], tags[60000:]):
        train_data.append((word, tag))
    train_data = [train_data]  # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

    # Train
    brill_tagger = train(train_data)
    title = '2nd cross Validation: Heldout-Test-Training, 20-40-20, no sentence segmentation'
    evaluate(brill_tagger, out, title)

    # With sentence segmentation, uncomment for use

    # Get test data
    test_data = []
    sent = []
    pos1 = 20000
    pos2 = 60000
    while words[pos1] != '###':
        pos1 += 1
    pos1 += 1  # this is the first sentence in set
    while words[pos2] != '###':
        pos2 += 1
    for word, tag in zip(words[pos1:pos2], tags[pos1:pos2]):
        if word == '###':
            test_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))  # heldout_data = [heldout_data]

    # Get train data
    train_data = []
    sent = []
    pos1 = pos2
    while words[pos1] != '###':
        pos1 += 1
    pos1 += 1  # skip first '#'
    for word, tag in zip(words[pos1:], tags[pos1:]):
        if word == '###':
            train_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))

    # Train
    brill_tagger = train(train_data)
    title = '2nd cross Validation: Heldout-Test-Training, 20-40-20, with sentence segmentation'
    evaluate(brill_tagger, out, title)


    # Cross Validation #3
    # Test-Heldout-Training
    # 20-rest-20
    print('CV 3')


    # Without sentence segmentation, uncomment for use

    # Get test data
    test_data = []
    for word, tag in zip(words[:20000], tags[:20000]):
        test_data.append((word, tag))
    test_data = [test_data]  # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

    # Get train data
    train_data = []
    for word, tag in zip(words[-20000:], tags[-20000:]):
        train_data.append((word, tag))
    train_data = [train_data]  # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

    # Train
    brill_tagger = train(train_data)
    title = '3nd cross Validation: Test-Heldout-Training, 20-40-20, no sentence segmentation'
    evaluate(brill_tagger, out, title)

    # With sentence segmentation, uncomment for use

    # Get test data
    test_data = []
    sent = []
    pos1 = 0
    pos2 = 20000
    while words[pos1] != '###':
        pos1 += 1
    pos1 += 1  # this is the first sentence in set
    while words[pos2] != '###':
        pos2 += 1
    for word, tag in zip(words[pos1:pos2], tags[pos1:pos2]):
        if word == '###':
            test_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))  # heldout_data = [heldout_data]

    # Get train data
    train_data = []
    sent = []
    pos1 = -20000
    while words[pos1] != '###':
        pos1 += 1
    pos1 += 1  # skip first '#'
    for word, tag in zip(words[pos1:], tags[pos1:]):
        if word == '###':
            train_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))

    # Train
    brill_tagger = train(train_data)
    title = '3nd cross Validation: Test-Heldout-Training, 20-40-20, with sentence segmentation'
    evaluate(brill_tagger, out, title)

    # test_data = treebank.sents()[0]
    # print(len(test_data))
    # print(len(train_data))
    # print(len(heldout_data))
    # print(heldout_data)
    
    
    # Construct trainer for Brill tagger using backoff: Unigram,
    # Bigram, Trigram, and otherwise to NN
    # init_tagger = backoff_tagger([train_data], [UnigramTagger, BigramTagger, TrigramTagger], backoff)
    # templates = brill.brill24()
    # trainer = brill_trainer.BrillTaggerTrainer(init_tagger, templates, deterministic=True, trace=True)
    # brill_tagger = trainer.train([train_data])
    # res = brill_tagger.evaluate([test_data])
    # print(res)
    # rules = brill_tagger.rules()
    # print(rules)
    # brill_tagger = train_brill_tagger(initial_tagger, , train_sents)
    
    
    # tags = t.tag(['Hello', "mothger"])
    # print(tags)
    #
    # # test_sent = brown.sents(categories='news')[0]
    # test_sents = [test_data]
    # sents = words[:10]
    # # print(sents)
    # # print(test_sents)
    # # unigram_tagger = UnigramTagger(brown.tagged_sents(categories='news')[:10])
    # unigram_tagger = UnigramTagger([train_data])
    # # res = [unigram_tagger.tag(s) for s in sents]
    # res = unigram_tagger.evaluate(test_sents)
    # print(res)
    # # for tok, tag in res:
    # #     print("(%s, %s), " % (tok, tag))
    


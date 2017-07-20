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

def evaluate(brill_tagger):
    res = brill_tagger.evaluate(test_data)
    print(res)
    rules = brill_tagger.rules()
    # print(rules)

stream = io.TextIOWrapper(sys.stdin.buffer, encoding='iso-8859-2')
# stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
words = []
tags = []
test_data = []
train_data = []
heldout_data = []
for line in stream:
    if line != '\n':
        # print(line)
        word, tag = line.rstrip().split('/')
        words.append(word)
        tags.append(tag)

# n = len(words)

# Split data for initial part
# T-H-V
# rest-20-40
print('initial split')
# Get test data, the last 20k
for word, tag in zip(words[-40000:], tags[-40000:]):
    test_data.append((word, tag))
test_data = [test_data]     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

# Get heldout data, the mid 20k (not used in brill).
# Not used for this part
for word, tag in zip(words[-60000:-40000], tags[-60000:-40000]):
    heldout_data.append((word, tag))

# Get train data, the first ~20k
for word, tag in zip(words[:-60000], tags[:-60000]):
    train_data.append((word, tag))
train_data = [train_data]     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

# Train
brill_tagger = train(train_data)
evaluate(brill_tagger)

# Cross Validate


# Get test/validation data, the first 40k
print('initial CV')
test_data = []
for word, tag in zip(words[:40000], tags[:40000]):
    test_data.append((word, tag))
test_data = [test_data]     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

# Get training data, the lst 40k
train_data = []
for word, tag in zip(words[40000:60000], tags[40000:60000]):
    train_data.append((word, tag))
train_data = [train_data]     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

# Train
brill_tagger = train(train_data)
evaluate(brill_tagger)

# Cross validate three more times
# Splitting the data each time into
# folds of 20-40-rest

# 1
# H-T-V
# 20-40-rest
print('CV 1')
train_data = []
for word, tag in zip(words[20000:60000], tags[20000:60000]):
    train_data.append((word, tag))
train_data = [train_data]  # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

test_data = []
for word, tag in zip(words[60000:], tags[60000:]):
    test_data.append((word, tag))
test_data = [test_data]     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

# Train
brill_tagger = train(train_data)
evaluate(brill_tagger)

# 2
# H-V-T
# 20-40-rest
print('CV 2')
test_data = []
for word, tag in zip(words[20000:60000], tags[20000:60000]):
    test_data.append((word, tag))
test_data = [test_data]  # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

train_data = []
for word, tag in zip(words[60000:], tags[60000:]):
    train_data.append((word, tag))
train_data = [train_data]  # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

# Train
brill_tagger = train(train_data)
evaluate(brill_tagger)

# 3
# V-H-T
# 20-rest-20
print('CV 3')
test_data = []
for word, tag in zip(words[:20000], tags[:20000]):
    test_data.append((word, tag))
test_data = [test_data]  # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

train_data = []
for word, tag in zip(words[-20000:], tags[-20000:]):
    train_data.append((word, tag))
train_data = [train_data]  # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

# Train
brill_tagger = train(train_data)
evaluate(brill_tagger)


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


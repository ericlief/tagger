#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

    Project: Code
    File: hmm
    Version:
    Created by liefe on 20.7.17.

"""

# from language_model.languageModel import *
from collections import defaultdict
import numpy as np
import sys, io


class HMMTagger:
    """
    This class holds all gram distribution data,
    as well as the methods to calculate probabilities
    and setter and getter methods.
    """

    def __init__(self):

        self._lambdas = None                    # smoothing params (lambdas)

    @classmethod
    def train(cls, train_data):
        """
        Supervised training method for learning from a data
        set. Special care has been taken to handle begining and
        end of text, which has been broken up into sentences, into
        which  start and end delimiters are placed.

        :param train_data: list(list)
        :return: HMMTagger
        """
        cls._words = set([])
        cls._uni_words = defaultdict(int)
        cls._tags = set([])
        cls._emission_counts = defaultdict(int)
        cls._uni_transition_counts = defaultdict(int)
        cls._bi_transition_counts = defaultdict(int)
        cls._tri_transition_counts = defaultdict(int)

        # Update dictionaries with tag transition distributions
        n = 0   # count word-tags
        for sent in train_data:

            # Handle beginning of sentence
            first = '<s>'
            second = '<s>'
            cls._bi_transition_counts[first, second] += 1
            cls._uni_transition_counts[first] += 1

            for word, tag in sent:
                third = tag
                cls._tri_transition_counts[first, second, third] += 1
                cls._bi_transition_counts[second, third] += 1
                cls._uni_transition_counts[third] += 1
                cls._emission_counts[word, tag] += 1
                cls._tags.add(tag)
                cls._words.add(word)
                cls._uni_words[word] += 1

                first = second
                second = third
                n += 1

            # Handle end of sentence
            cls._tri_transition_counts[first, second, '</s>'] += 1
            cls._bi_transition_counts[second, '</s>'] += 1
            cls._n = n


        cls._tags.add('<s>')
        cls._tags.add('</s>')

        print(cls._tags)

        return cls

    @classmethod
    def bi_tag(cls, corpus):
        """
        An alternate bigram tagger, implemented
        to compare performance with the trigram
        counterpart
        :param corpus: list
        :return: tagged sequence
        """
        words = [w for sent in corpus for w in sent]
        # print(words)
        n = len(words)
        viterbi = {}
        path = {}

        # Initialization step
        w = words[0]
        for t in cls._tags:
            viterbi[1, t] = cls._bi_transition_counts['<s>', t] * cls.calculate_emission_prob(w, t)
            path[1, t] = []

        # Recursion step
        for i in range(2, n+1):
            w = words[i-1]
            if w not in cls._words:
                w = '<unk>'
            for v in cls._tags:
                viterbi[i, v], u_max = max([(viterbi[i-1, u] * float(cls._bi_transitions[u, v]/cls._uni_transitions[u])
                                             * cls.calculate_emission_prob(w, v), u) for u in cls._tags])

                path[i, v] = u_max

        # Final step, to sentence
        viterbi[n, '</s>'], u_max = max([(viterbi[n, u] * float(cls._bi_transitions[u, '</s>']/cls._uni_transitions[u]), u)
                           for u in cls._tags])
        path[n, '</s>'] = u_max

        print(viterbi)
        print(cls._emissions)
        return HMMTagger.backtrace(path, n)

    @classmethod
    def tag(cls, sent, k=7, interpolate=True):
        """
        Tag a sequence (sentence) using
        either the supervised or semi-supervised
        trigram tagger.
        :param corpus: list
        :return: tagged sequence
        """
        m = len(sent)
        n = len(cls._tags)

        if interpolate:
            L = cls._lambdas

        viterbi = [[[0.0 for k in range(n)] for j in range(n)] for i in range(m+2)]
        path = [[[0.0 for k in range(n)] for j in range(n)] for i in range(m+2)]

        # Initialization step
        start = cls._tag_to_index['<s>']
        viterbi[0][start][start] = 1
        path[0][start][start] = []

        # Initialize active tagset for k-best search
        active_tags_t = [start]
        active_tags_u = [start]

        for i in range(1, m + 1):
            k_best = {}     # list of n-best search

            # Get next word, convert to index
            word = cls.next_word(sent, i - 1)
            if word not in cls._words:
                word = '<unk>'
            w = cls._word_to_index[word]    # indexed word

            for v in cls.possible_tags(i):
                for u in active_tags_u:
                    if interpolate:
                        score, t_max = max([(
                            viterbi[i-1][t][u] * cls.interpolate(t, u, v, L) * cls._emissions[w][v], t)
                            for t in active_tags_t])
                    else:  # no EM smoothing of tag parameters, employ tri-tag model probs
                        score, t_max = max([(
                            viterbi[i-1][t][u] * cls._tri_transitions[t][u][v] * cls._emissions[w][v], t)
                            for t in active_tags_t])

                    viterbi[i][u][v] = score
                    path[i][u][v] = t_max
                    k_best[u, v] = score

            best = sorted(k_best, key=k_best.get, reverse=True)[:k]
            active_tags_t = [t for (t, u) in best]
            active_tags_u = [u for (t, u) in best]

        # Final step, to sentence
        end = cls._tag_to_index['</s>']

        if interpolate:
            score, t_max, u_max = max([(viterbi[m][t][u] * cls.interpolate(t, u, end, L), t, u)
                                       for t in active_tags_t for u in active_tags_u])
        else:
            score, t_max, u_max = max([(viterbi[m][t][u] * cls._tri_transitions[t][u][end], t, u)
                                       for t in active_tags_t for u in active_tags_u])
        viterbi[m+1][u_max][end] = score
        path[m+1][u_max][end] = t_max

        return cls.backtrace(path, m + 1, u_max)

    @classmethod
    def baum_welch(cls, sent, emission_counts, uni_transition_counts, \
                               bi_transition_counts, tri_transition_counts, interpolate=False):
        """
        The driver method (BW step) of the
        unsupervised/semi-supervised training
        model.
        :param sent: list
        :param emission_counts:
        :param uni_transition_counts:
        :param bi_transition_counts:
        :param tri_transition_counts:
        :param interpolate:
        :return: emission_counts, uni_transition_counts, bi_transition_counts, tri_transition_counts, words, tags, m
        """

        # Compute forward (alpha) and backward (beta) probability matrices
        alpha = cls.forward(sent, interpolate)
        beta = cls.backward(sent, interpolate)
        words = set([])
        tags = set([])
        print(alpha)
        print(beta)
        # Collect counts (E-Step)
        m = len(sent)   # number of output symbols (local)
        if interpolate:
            L = cls._lambdas

        for i in range(0, m-1):

            # Get next word, transform and index
            word = cls.next_word(sent, i+1)      # string index starting at zero, so really is i+1, changed from i
            if word not in cls._words:
                word = '<unk>'
            w = cls._word_to_index[word]  # indexed word
            words.add(word)

            for t in cls.possible_tags(i-1):
                for u in cls.possible_tags(i):
                    tags.add(cls._tags[u])
                    for v in cls.possible_tags(i+1):

                        # Calculate fraction count/increment value

                        # Use smoothed params obtained from first 10k of training data????
                        # if iter == 1:
                        if interpolate:
                            increment = alpha[i][t][u] * cls.interpolate(t, u, v, L) \
                                        * cls._emissions[w][v] * beta[i+1][u][v]

                        # For remaining iterations recalculate increments from past probabilities????
                        else:
                            increment = alpha[i][t][u] * cls._tri_transitions[t][u][v] \
                                        * cls._emissions[w][v] * beta[i+1][u][v]

                        if increment != 0:
                            ti = cls._tags[t]
                            tj = cls._tags[u]
                            tk = cls._tags[v]
                            print(ti,tj,tk, increment)

                        # Convert number to string
                        ti = cls._tags[t]
                        tj = cls._tags[u]
                        tk = cls._tags[v]
                        emission_counts[word, tk] += increment
                        uni_transition_counts[ti] += increment  # unigram tag transition count
                        bi_transition_counts[tj, tk] += increment  # bigram tag transition count
                        tri_transition_counts[ti, tj, tk] += increment  # trigram tag transition count

        return emission_counts, uni_transition_counts, bi_transition_counts, tri_transition_counts, words, tags, m

    @staticmethod
    def update(d, e):
        """
        Helper method for updating data structs:
        dicts after each iteration
        :param d: dict
        :param e: dict
        :return:
        """
        for key in e:

            if key in d:

                d[key] += e[key]
            else:
                d[key] = e[key]

        return d

    @classmethod
    def train_unsupervised(cls, data, interpolate=False):
        """
        Iterative unsupervised training, learning
        params from an untagged training set. The interpolate
        params specifies whether to use linear interpolation
        for trigram probabilities (learned with EM param during
        smoothing).
        :param data: list(list)
        :param interpolate: boolean
        :return: HMMTagger
        """
        #
        # emission_counts = cls._emission_counts
        # uni_transition_counts = cls._uni_transition_counts
        # bi_transition_counts = cls._bi_transition_counts
        # tri_transition_counts = cls._tri_transition_counts
        #

        emission_counts = defaultdict(lambda: 0.0)
        uni_transition_counts = defaultdict(lambda: 0.0)
        bi_transition_counts = defaultdict(lambda: 0.0)
        tri_transition_counts = defaultdict(lambda: 0.0)
        words = set(cls._words)
        tags = set(cls._tags)
        M = 0  # total size of text

        for sent in data:

            # Baum Welch step on new sentence
            ec, uni_tc, bi_tc, tri_tc, new_words, new_tags, m = \
                cls.baum_welch(sent, emission_counts, uni_transition_counts, \
                               bi_transition_counts, tri_transition_counts, interpolate)

            # Update counts
            HMMTagger.update(emission_counts, ec)
            uni_transition_counts = HMMTagger.update(uni_transition_counts, uni_tc)
            HMMTagger.update(bi_transition_counts, bi_tc)
            HMMTagger.update(tri_transition_counts, tri_tc)
            words.update(new_words)
            tags.update(new_tags)
            M += m

            print(sent)
            print(uni_transition_counts)
            print(uni_tc)

            print(emission_counts)
            print(ec)

            # print(emission_counts, uni_transition_counts, bi_transition_counts, tri_transition_counts)

        # Add start and end symbols to sets
        tags.add('<s>')
        tags.add('</s>')

        # Convert tags and words to indices
        tags = list(tags)
        tag_to_index = {}
        for tag in tags:
            tag_to_index[tag] = tags.index(tag)

        words = list(words)
        word_to_index = {}
        for word in words:
            word_to_index[word] = words.index(word)

        # Convert emission and (tri)transition hash tables (dictionaries) to lists
        n = len(tags)
        tri_transitions = [[[0.0 for k in range(n)] for j in range(n)] for i in range(n)]
        for t, u, v in tri_transition_counts:
            i = tag_to_index[t]
            j = tag_to_index[u]
            k = tag_to_index[v]

            if tri_transition_counts[t, u, v] == 0 or bi_transition_counts[t, u] == 0:
                tri_transitions[i][j][k] == 0
                continue
            try:
                tri_transitions[i][j][k] = float(tri_transition_counts[t, u, v] / bi_transition_counts[t, u])
            except ZeroDivisionError:
                tri_transitions[i][j][k] = 0.0

        bi_transitions = [[0.0 for j in range(n)] for i in range(n)]
        for t, u in bi_transition_counts:
            i = tag_to_index[t]
            j = tag_to_index[u]
            if bi_transition_counts[t, u] == 0 or uni_transition_counts[t] == 0:
                bi_transitions[i][j] == 0
                continue
            try:
                bi_transitions[i][j] = float(bi_transition_counts[t, u] / uni_transition_counts[t])
            except ZeroDivisionError:
                bi_transitions[i][j] == 0

        uni_transitions = [0.0 for i in range(n)]
        for t in uni_transition_counts:
            i = tag_to_index[t]
            uni_transitions[i] = uni_transition_counts[t] / M

        m = len(words)
        emissions = [[0.0 for j in range(n)] for i in range(m)]
        for (word, tag), c in emission_counts.items():
            w = word_to_index[word]
            t = tag_to_index[tag]
            if c == 0 or uni_transition_counts[tag] == 0:
                emissions[w][t] = 0.0
                continue
            try:
                emissions[w][t] = c / uni_transition_counts[tag]
            except ZeroDivisionError:
                emissions[w][t] = 0.0

        cls._tri_transitions = tri_transitions
        cls._bi_transitions = bi_transitions
        cls._uni_transitions = uni_transitions
        cls._emissions = emissions
        cls._word_to_index = word_to_index
        cls._tag_to_index = tag_to_index
        cls._words = words
        cls._tags = tags
        cls._n = M

        print('the', uni_transition_counts['the'])

        return cls

    @classmethod
    def forward(cls, sent, interpolate=False):
        """
        The forward step, building alpha trellis.
        In order to avoid underflow, the suggested
        rescaling of probs has been implemented.
        Furthemore, a more efficient implementation
        using lists (arrays) rather than dicts (hash maps)
        has been added
        :param sent: list
        :param interpolate: boolean
        :return: alpha list(list(list))
        """
        m = len(sent)
        n = len(cls._tags)
        L = cls._lambdas
        alpha = [[[0.0 for k in range(n)] for j in range(n)] for i in range(m+2)]

        # Calculate forward probs

        # Initialization
        start = cls._tag_to_index['<s>']
        alpha[0][start][start] = 1

        # Iterate through sequence of length n
        for i in range(1, m + 1):

            normalization_factor_sum = 0

            # Get new word/symbol
            word = cls.next_word(sent, i - 1)
            if word not in cls._words:
                word = '<unk>'
            w = cls._word_to_index[word]  # indexed word

            # Inductive step
            for u in cls.possible_tags(i-1):
                for v in cls.possible_tags(i):

                    # Alpha or forward prob at one node (u, v)
                    if interpolate:

                        alpha[i][u][v] = sum([alpha[i-1][p][q] * cls.interpolate(q, u, v, L) * cls._emissions[w][v]
                                              for p in cls.possible_tags(i - 2) for q in cls.possible_tags(i - 1)])

                    else:
                        alpha[i][u][v] = sum(
                            [alpha[i - 1][p][q] * cls._tri_transitions[q][u][v] * cls._emissions[w][v]
                             for p in cls.possible_tags(i - 2) for q in cls.possible_tags(i - 1)])

                    # Update normalization sum
                    normalization_factor_sum += alpha[i][u][v]

            # Recalculate alphas
            for j in range(n):
                for k in range(n):
                    if normalization_factor_sum != 0:
                        alpha[i][j][k] = alpha[i][j][k] / normalization_factor_sum
                    else:
                        alpha[i][j][k] = 0.0

        # Final step?
        end = cls._tag_to_index['</s>']
        for t in cls.possible_tags(m+1):
            if interpolate:
                alpha[m+1][t][end] = sum([alpha[m][s][t] * cls.interpolate(s, t, end, L)
                                             for s in cls.possible_tags(m)])
            else:
                alpha[m+1][t][end] = sum([alpha[m][s][t] * cls._tri_transitions[s][t][end]
                                               for s in cls.possible_tags(m)])
        return alpha

    @classmethod
    def backward(cls, sent, interpolate=False):
        """
        The backward step, building beta trellis.
        In order to avoid underflow, the suggested
        rescaling of probs has been implemented.
        Furthemore, a more efficient implementation
        using lists (arrays) rather than dicts (hash maps)
        has been added
        :param sent: list
        :param interpolate: boolean
        :return: beta list(list(list))
        """
        m = len(sent)
        n = len(cls._tags)
        L = cls._lambdas
        beta = [[[0.0 for k in range(n)] for j in range(n)] for i in range(m+2)]

        # Calculate forward probs

        # Initialization
        end = cls._tag_to_index['</s>']
        for t in cls.possible_tags(m+1):
            for u in cls.possible_tags(m+1):
                if interpolate:
                    beta[m+1][u][end] = cls.interpolate(t, u, end, L)
                else:
                    beta[m+1][u][end] = cls._tri_transitions[t][u][end]


        for s in cls.possible_tags(m):
            for t in cls.possible_tags(m):
                if interpolate:
                    beta[m][s][t] = sum([beta[m+1][u][end] * cls.interpolate(t, u, end, L)
                         for u in cls.possible_tags(m)])
                else:
                    beta[m][s][t] = sum([beta[m+1][u][end] * cls._tri_transitions[t][u][end]
                                         for u in cls.possible_tags(m)])


        # Iterate through sequence of length m
        for i in range(m-1, -1, -1):

            normalization_factor_sum = 0
            word = cls.next_word(sent, i)
            if word not in cls._words:
                word = '<unk>'
            w = cls._word_to_index[word]  # indexed word

            # Inductive step
            for s in cls.possible_tags(i-1):
                for t in cls.possible_tags(i):
                    # Beta or backward prob at one node (u, v)
                    if interpolate:
                        beta[i][s][t] = sum([beta[i+1][u][v] * cls.interpolate(t, u, v, L) * cls._emissions[w][v]
                                             for u in cls.possible_tags(i) for v in cls.possible_tags(i + 1)])
                    else:
                        beta[i][s][t] = sum(
                            [beta[i+1][u][v] * cls._tri_transitions[t][u][v] * cls._emissions[w][v]
                             for u in cls.possible_tags(i) for v in cls.possible_tags(i + 1)])

                    # Update normalization sum
                    normalization_factor_sum += beta[i][s][t]

            # Recalculate betas
            for j in range(n):
                for k in range(n):
                    if normalization_factor_sum != 0:
                        beta[i][j][k] = beta[i][j][k] / normalization_factor_sum
                    else:
                        beta[i][j][k] = 0.0

        # # Final step? Not sure if needed, so left out
        # for t in cls.possible_tags(i):
        #     beta[i + 1, t, '</s>'] = sum([beta[i, s, t] * cls.calculate_interpolated_p(s, t, '</s>', L)
        #                                    for s in cls.possible_tags(i)])

        return beta

    @classmethod
    def backtrace(cls, path, n, u):
        """
        Helper method for appending tag sequence
        together using backpointers.
        :param path:
        :param n:
        :param u:
        :return: tag sequence (list)
        """

        tags = [cls._tags[u]]
        i = n
        v = cls._tag_to_index['</s>']
        while i > 2:
            t = path[i][u][v]
            tags.insert(0, cls._tags[t])
            v = u
            u = t
            i -= 1

        return tags

    @classmethod
    def possible_tags(cls, i):
        i_start = cls._tag_to_index['<s>']
        i_end = cls._tag_to_index['</s>']
        if i == -1:
            return {i_start}
        elif i == 0:
            return {i_start}
        else:
            return set(range(len(cls._tags))) - {i_start} - {i_end}

    @classmethod
    def next_word(cls, sent, i):
        """
        Helper method for scanning for
        next word, and indexing at 1.
        :param sent:
        :param i:
        :return: word string
        """
        if i < 0:
            return ''
        else:
            return sent[i]

    @classmethod
    def calculate_emission_prob(cls, w, t):
        """
        Helper method to calculate the smoothed emission
        (lexical) probability.
        :param w:
        :param t:
        :return: p
        """

        return float(cls._emission_counts[w, t] / cls._uni_transition_counts[t])

    @classmethod
    def smooth_unkown_words_with_threshold(cls, threshold=8):
        """
        Alternate lexical smoothing method which converts
        words below a threshold, to the unkown token <UNK>.
        No heldout data necessary.

        :param threshold:
        :return: smoothed prob table (dict)
        """
        emission_counts = defaultdict(int)
        smoothed_probs = defaultdict(lambda: 0.0)

        # Convert all words below threshold to ``<unk>``
        for (word, tag) in cls._emission_counts:
            cnt = cls._emission_counts[(word, tag)]
            emission_counts[(word, tag)] = cnt
            if cnt < threshold:
                emission_counts[('<unk>', tag)] += cnt

        # Calculate emission probabilities
        for w, t in emission_counts:
            # unk_probs[w, t] = unk[w, t] / n
            smoothed_probs[w, t] = emission_counts[w, t] / cls._uni_transition_counts[t]

        cls._words.add('<unk>')

        return smoothed_probs

    @classmethod
    def smooth_unknown_words_from_heldout(cls, heldout_data):
        """
        Slightly improved lexical smoothing method which converts
        words not present in a separate (heldout) data set
        to the unkown token <UNK>.

        :param heldout_data:
        :return: smoothed emission probs (dict)
        """

        unk = defaultdict(int)
        tags = defaultdict(int)
        n = 0   # size of data
        unk_cnt = 0
        cnt = 0
        for sent in heldout_data:
            for w, t in sent:
                if w not in cls._words:
                    unk['<unk>', t] += 1
                    unk_cnt += 1
                tags[t] += 1
                n += 1

        unk_mass = unk_cnt / n
        smoothed_probs = defaultdict(lambda: 0.0)

        for w, t in unk:
            smoothed_probs[w, t] = unk[w, t] / tags[t]
            if t not in cls._tags:
                cls._tags.add(t)

        for w, t in cls._emission_counts:

            smoothed_probs[w, t] = float(cls._emission_counts[w, t] / cls._uni_transition_counts[t]) * (1 - unk_mass)

        cls._words.add('<unk>')
        return smoothed_probs

    @classmethod
    def smooth_tag_model(cls, heldout_data):
        """
        Use EM to calculate tag transition params
        :param heldout_data:
        :return: lambdas
        """

        # bi_transition_counts = defaultdict(int)
        n = 0       # count word-tags
        e = .0001    # stopping condition
        L = [.25, .25, .25, .25]  # initialize lambdas uniformly
        i = 1  # iteration
        while True:
            # E Step (Step 1)
            # Iterate through all occurring trigrams
            # in the heldout.txt data (H), i.e. minimizing
            # log likelihood
            counts = [0, 0, 0, 0]
            ratio = [0, 0, 0, 0]
            nextL = 4 * [0]  # next lambda

            for sent in heldout_data:

                # Handle beginning of sentence
                t = '<s>'
                u = '<s>'


                for word, tag in sent:
                    v = tag
                    if v not in cls._tags:
                        cls._tags.add(v)

                    # Calculate expected counts of lambdas
                    ratio = cls.calc_tag_ratio(t, u, v, L)

                    # M-step (Step 2)
                    # Calculate expected counts of lambdas, i.e. weight, taking
                    # into account the number of occurrences of each trigram (cnt)
                    for j in range(len(L)):
                        counts[j] += ratio[j]  # weight of lambda in whole equation (count)

                    t = u
                    u = v

                # Handle end of sentence
                v = '</s>'
                ratio = cls.calc_tag_ratio(t, u, v, L)
                for j in range(len(L)):
                    counts[j] += ratio[j]  # weight of lambda in whole equation (count)

            # Update values for parameters given current distribution
            for k in range(len(L)):
                total = np.sum(counts)
                nextL[k] = counts[k] / total  # next lambda

            # Check if lambda values have converged
            converged = True
            for l in range(len(L)):
                if np.abs(nextL[l] - L[l]) > e:  # tolerance = e
                    converged = False
            L = nextL

            # Return values if lambdas have converged
            if converged:
                break

            i += 1  # increment iteration counter


        return L  # copy lambdas passed by reference

    @classmethod
    def calc_tag_ratio(cls, t, u, v, L):
        """
        Computes the smoothed (weighted) probability P' using
        the distribution calculated for the language model
        using training data.

        :param t: tag string
        :param u: tag string
        :param v: tag string
        :param L: tag string
        :return: weighted p
        """

        ratio = [0.0, 0.0, 0.0, 0.0]

        # Convert tag to index (string to number)
        i = cls._tag_to_index[t]
        j = cls._tag_to_index[u]
        k = cls._tag_to_index[v]

        weighted_p = cls.interpolate(i, j, k, L)
        V = len(cls._uni_transitions)     # tag vocabulary size

        ratio[3] = L[3] * cls._tri_transitions[i][j][k] / weighted_p  # ratio of p3/p' to model distribution function
        ratio[2] = L[2] * cls._bi_transitions[j][k] / weighted_p
        ratio[1] = L[1] * cls._uni_transitions[k] / weighted_p
        ratio[0] = L[0] / V / weighted_p

        return ratio

    @classmethod
    def interpolate(cls, i, j, k, L):
        """
        Helper method for computing the smoothed
        (weighted/interpolated) probability
        P' using the distribution calculated for the language model
        using training data
        :param t: w_i-2 (int)
        :param u: w_1-1 (int)
        :param v: w_i (int)
        :param L: lambdas
        :return: P'
        """

        V = len(cls._uni_transitions)  # tag vocabulary size
        interpolated_p = L[3] * cls._tri_transitions[i][j][k] + L[2] * cls._bi_transitions[j][k] \
                         + L[1] * cls._uni_transitions[k] + L[0] / V

        return interpolated_p

    @classmethod
    def initialize_params(cls, heldout_data=None, mode=None):
        """
        Initializes data structures obtained from supervised training
        data or smoothing of lexical and/or tag models.
        Efficient data structure transformation, using lists, with numerical data,
        rather than initial dictionaries.
        :param heldout_data:
        :param mode:
        :return:
        """
        # Iterate through counts and calculate the respective
        # tag transition probabilities
        tri_tag_probs = defaultdict(lambda: 0.0)
        bi_tag_probs = defaultdict(lambda: 0.0)
        uni_tag_probs = defaultdict(lambda: 0.0)

        for t, u, v in cls._tri_transition_counts:
            tri_tag_probs[t, u, v] = float(cls._tri_transition_counts[t, u, v] / cls._bi_transition_counts[t, u])
            bi_tag_probs[u, v] = float(cls._bi_transition_counts[u, v] / cls._uni_transition_counts[u])
            uni_tag_probs[v] = float(cls._uni_transition_counts[v] / cls._n)
            tri_tag_probs[t, u, v] = float(cls._tri_transition_counts[t, u, v] / cls._bi_transition_counts[t, u])

        # Smooth lexical model
        if mode == 'unk_heldout':
            emission_probs = cls.smooth_unknown_words_from_heldout(heldout_data)
        elif mode == 'unk_threshold':
            emission_probs = cls.smooth_unkown_words_with_threshold(threshold=7)
        elif mode is None:
            emission_probs = defaultdict(lambda : 0.0)
            # Calculate emission probabilities
            for w, t in cls._emission_counts:
                # unk_probs[w, t] = unk[w, t] / n
                emission_probs[w, t] = cls.calculate_emission_prob(w, t)
        else:
            raise Exception

        # Convert tags and words to indices
        tags = list(cls._tags)
        tag_to_index = {}
        for tag in tags:
            tag_to_index[tag] = tags.index(tag)

        words = list(cls._words)
        word_to_index = {}
        for word in words:
            word_to_index[word] = words.index(word)

        # Convert emission and (tri)transition hash tables (dictionaries) to lists
        n = len(tags)
        tri_transitions = [[[0.0 for k in range(n)] for j in range(n)] for i in range(n)]
        for (t, u, v), p in tri_tag_probs.items():
            i = tag_to_index[t]
            j = tag_to_index[u]
            k = tag_to_index[v]
            tri_transitions[i][j][k] = p

        bi_transitions = [[0.0 for j in range(n)] for i in range(n)]
        for (t, u), p in bi_tag_probs.items():
            i = tag_to_index[t]
            j = tag_to_index[u]
            bi_transitions[i][j] = p

        uni_transitions = [0.0 for i in range(n)]
        for t, p in uni_tag_probs.items():
            i = tag_to_index[t]
            uni_transitions[i] = p

        m = len(words)
        emissions = [[0.0 for j in range(n)] for i in range(m)]
        for (word, tag), p in emission_probs.items():
            w = word_to_index[word]
            t = tag_to_index[tag]
            emissions[w][t] = p

        cls._tri_transitions = tri_transitions
        cls._bi_transitions = bi_transitions
        cls._uni_transitions = uni_transitions
        cls._emissions = emissions
        cls._word_to_index = word_to_index
        cls._tag_to_index = tag_to_index
        cls._words = words
        cls._tags = tags

        # Smooth tag model
        if heldout_data:
            cls._lambdas = cls.smooth_tag_model(heldout_data)
        else:
            cls._lambdas = None

def write_results(model, gold, label, fname, iter=None, interpolate=False):
    """
    Helper method for writing results to disk after each iteration or once
    during supervised training.
    Uncomment for debugging each sentence and writing correct/incorrect
    results.

    :param model:
    :param gold:
    :param label:
    :param fname:
    :param iter:
    :param interpolate:
    :return: accuracy
    """
    # Strip tags
    gold_untagged_sents = []
    gold_tagged_sents = gold
    n = 0
    for tagged_sent in gold_tagged_sents:
        untagged_sent = []
        for word, tag in tagged_sent:
            untagged_sent.append(word)
            n += 1
        gold_untagged_sents.append(untagged_sent)

    # **Tag text and calculate and write accuracy**
    with open(fname, 'a') as f:
        correct = 0
        for i, sent in enumerate(gold_untagged_sents):
            # f.write('sentence ' + str(i) + '\n')
            # f.write(' '.join(sent) + '\n')
            # tags = model.tri_tag(sent, 8, interpolate)
            tags = model.tag(sent, 8, interpolate)

            # print(sent)
            # print(tags)
            for j, (w, t) in enumerate(zip(sent, tags)):
                # f.write(str(w) + '/' + str(t) + '\n')
                if gold_tagged_sents[i][j] == (w, t):
                    # f.write('correct' + str(gold_tagged_sents[i][j]) + '\t' + str((w, t)) + '\n')
                    correct += 1
                # else:
                    # f.write('incorrect' + str(gold_tagged_sents[i][j]) + '\t' + str((w, t)) + '\n')
        # print(correct)
        # print(n)
        acc = correct / n
        f.write(label + str(iter) + '\n')
        f.write('Accuracy = ' + str(acc) + '\n')

        return acc

if __name__ == '__main__':

    import pickle

    lang = '_en_'
    # lang = '_cz_'

    # stream = io.TextIOWrapper(sys.stdin.buffer, encoding='iso-8859-2')
    stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    words = []
    tags = []
    for line in stream:
        if line != '\n':
            # print(line)
            word, tag = line.rstrip().split('/')
            words.append(word)
            tags.append(tag)

    # *************************************************************************** #
    # Get test data for Part I (Viterbi)

    # With segmentation of sentences

    # test_data = []
    # sent = []
    # pos1 = -40000
    # while words[pos1] != '###':
    #     pos1 += 1
    # pos1 += 1  # skip first '#'
    # for word, tag in zip(words[pos1:], tags[pos1:]):
    #     if word == '###':
    #         if sent:
    #             test_data.append(sent)
    #         sent = []
    #         continue
    #     sent.append((word, tag))
    #
    # # Get heldout data, the mid 20k (not used in brill).
    # heldout_data = []
    # sent = []
    # # pos2 = -60000
    # pos2 = pos1
    # pos1 = -60000
    # while words[pos1] != '###':
    #     pos1 += 1
    # pos1 += 1  # skip first '#'
    # for word, tag in zip(words[pos1:pos2], tags[pos1:pos2]):
    #     # for word, tag in zip(words[pos2:pos1 - 1], tags[pos2:pos1 - 1]):
    #     if word == '###':
    #         if sent:
    #             heldout_data.append(sent)
    #         sent = []
    #         continue
    #     sent.append((word, tag))
    #
    # # Get initial train data, the first ~20k
    # train_data = []
    # sent = []
    # pos2 = pos1
    # pos1 = 0
    # while words[pos1] != '###':
    #     pos1 += 1
    # pos1 += 1  # this is the first sentence in set
    # for word, tag in zip(words[pos1:pos2], tags[pos1:pos2]):
    #     if word == '###':
    #         if sent:
    #             train_data.append(sent)
    #         sent = []
    #         continue
    #     sent.append((word, tag))  # heldout_data = [heldout_data]
    #
    # #
    # # # Without segmentation of sentences
    # #
    # # # Get test data, the last 40k
    # # test_data = []
    # # for word, tag in zip(words[-40000:], tags[-40000:]):
    # #     test_data.append((word, tag))
    # # test_data = [
    # #     test_data]  # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)
    # #
    # # # Get heldout data, the mid 20k (not used in brill).
    # # # Not used for this part, and just for show (not computed below)
    # # heldout_data = []
    # # for word, tag in zip(words[-60000:-40000], tags[-60000:-40000]):
    # #     heldout_data.append((word, tag))
    # # heldout_data = [heldout_data]
    # #
    # # # Get training data, the first ~20k
    # # train_data = []
    # # for word, tag in zip(words[:-60000], tags[:-60000]):
    # #     train_data.append((word, tag))
    # # train_data = [
    # #     train_data]  # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)
    #
    #
    # # ******************************************************************** #
    # # Task #1 **Train Viterbi**
    #
    # # Init and Train model
    # tagger = HMMTagger.train(train_data)
    #
    # # ***Smooth params/Heldout***
    # # NB: Uncomment for Viterbi (not BW)
    #
    # # params = 'heldout_data = None, mode = None'
    # # tagger.initialize_params(heldout_data=None, mode=None)
    # # write_results(tagger, test_data, params)
    # #
    # # tagger.initialize_params(heldout_data=None, mode='unk_threshold')
    # # params = "heldout_data = None, mode = 'unk_threshold'"
    # # write_results(tagger, test_data, params)
    #
    # label = "heldout_data, mode = 'unk_threshold'"
    # tagger.initialize_params(heldout_data, mode='unk_threshold')
    # write_results(tagger, test_data[:50], label, 'test.txt')
    # # (model, gold, label, fname, iter, interpolate = False):
    #
    # # params = "heldout_data, mode = 'unk_heldout'"
    # # tagger.initialize_params(heldout_data, mode='unk_heldout')
    # # write_results(tagger, test_data, params)
    #
    #
    #
    # # For test
    # # heldout_data = []
    # # with open('heldout.txt', 'r') as f:
    # #     sent = []
    # #     first = f.readline().rstrip()  # get first '###'
    # #     if first != '###':
    # #         f.seek(0)
    # #     for line in f:
    # #         if line == '\n':
    # #             continue
    # #         # word, tag = line.rstrip().split('/')
    # #         word = line.rstrip()
    # #         if word == '###':
    # #             heldout_data.append(sent)
    # #             sent = []
    # #             continue
    # #         sent.append(word)
    # #
    # #
    # # heldout_data.append(sent)
    # #         sent = []
    # #         continue
    # #     sent.append((word, tag))    # heldout_data = [heldout_data]
    # # heldout_data = [heldout_data]
    # # print(heldout_data[:5])
    # #     if word == '###':
    # #         sent = []
    # #         test_data.append(sent)
    # #         continue
    # #     sent.append((word, tag))
    # #
    # #
    # # tagger.smooth_lexical(heldout_data)
    # # L = tagger.smooth_tag_model(heldout_data)
    #
    #
    #
    #
    # # Unit test, uncomment if needed
    #
    # # print(list(tagger._tri_transitions.items())[:5])
    # # print(tagger._tri_transitions[('NN', 'DT', 'VBZ')])
    # # print(tagger._tri_transitions[('IN', 'DT', 'NN')])
    # # for k, v in tagger._tri_transitions.items():
    # #     if 'IN' in k and 'DT' in k:
    # #         print(k, v)
    # #
    # # print(tagger._bi_transitions[('<s>','<s>')])
    # # print(tagger._bi_transitions[('JJ','NNS')])
    # #
    #
    # # print(list(tagger._bi_transitions.items())[:5])
    # # print(list(tagger._uni_transitions.items())[:5])
    # # print(tagger._tags)
    # # print(tagger._words)
    # # print(tagger._emissions)
    #
    # # open unittest test data
    # # untagged_sents = []
    # # tagged_sents = []
    # # n = 0
    # # with open('gold-cz.txt', 'r') as f:
    # #     untagged_sent = []
    # #     tagged_sent = []
    # #     first = f.readline().rstrip()           # get first '###'
    # #     if first != '###/###':
    # #         f.seek(0)
    # #     for line in f:
    # #         # print(line)
    # #         if line == '\n':
    # #             continue
    # #         # word, tag = line.rstrip().split('/')
    # #         # word = line.rstrip()
    # #         word, tag = line.rstrip().split('/')
    # #         if word == '###':
    # #             untagged_sents.append(untagged_sent)
    # #             untagged_sent = []
    # #             tagged_sents.append(tagged_sent)
    # #             tagged_sent = []
    # #             continue
    # #         untagged_sent.append(word)
    # #         tagged_sent.append((word, tag))
    # #         n += 1
    #
    #
    #
    #
    #
    #
    # # print(untagged_sents[0])
    # # print(untagged_sents[-1])
    # # print(heldout_data[0])
    # # print(heldout_data[-1])
    #
    #
    # # strip test data, the last 20k
    # # for word, tag in zip(words[-40000:], tags[-40000:]):
    # #     test_data.append((word, tag))
    #
    # # untagged = words[:20]     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)
    # # untagged = words     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)
    #
    # # print(untagged)
    # # print(tagger._emissions)
    # # tagger.smooth_lexical()
    # # print(tagger._emissions)
    #
    # # tags = tagger.bi_tag(untagged)
    #
    # # for sent in untagged:
    # #
    # #     tags = tagger.tri_tag(sent)
    # #     print(sent)
    # #     print(tags)
    #
    # # print(tagger._tags)
    # # print(tagger._emissions)
    # # print(tagger._bi_transitions)
    # # print(untagged)
    #
    # # print(untagged_sents)
    # # print(tagged_sents)
    #
    #
    # #
    # # # **Tag text and calculate and write accuracy**
    # # with open('tagged.txt', 'w') as f:
    # #     correct = 0
    # #     for i, sent in enumerate(gold_untagged_sents):
    # #         f.write('sentence ' + str(i) + '\n')
    # #         f.write(' '.join(sent) + '\n')
    # #         tags = tagger.tri_tag(sent, k=8)
    # #         # print(sent)
    # #         # print(tags)
    # #         for j, (w, t) in enumerate(zip(sent, tags)):
    # #             f.write(str(w) + '/' + str(t) + '\n')
    # #             if gold_tagged_sents[i][j] == (w, t):
    # #                 f.write('correct' + str(gold_tagged_sents[i][j]) + '\t' + str((w, t)) + '\n')
    # #                 correct += 1
    # #             else:
    # #                 f.write('incorrect' + str(gold_tagged_sents[i][j]) + '\t' + str((w, t)) + '\n')
    # #     # print(correct)
    # #     # print(n)
    # #     acc = correct / n
    # #     f.write('Accuracy = ' + str(acc) + '\n')
    # #                 # te('viterbi')
    # #         #     f.write(str(x) + '\n')
    # #         # for x in tagger._path.items():
    # #         #     f.write('path')
    # #         #     f.write(str(x) + '\n')
    #
    #
    # # print(train_data[:5])
    # # print(heldout_data)
    # # print(tagger._bi_transition_counts['.', '</s>'])
    #
    #
    #


    # ************************************************************** #
    # Task #2. For remaining words in train data, strip tags

    # With segmentation of sentences

    end_train_data = 20000      # set size of unsupervised training data/end pos

    test_data = []
    sent = []
    pos1 = -40000
    while words[pos1] != '###':
        pos1 += 1
    pos1 += 1  # skip first '#'
    for word, tag in zip(words[pos1:], tags[pos1:]):
        if word == '###':
            if sent:
                test_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))

    # Get heldout data, the mid 20k (not used in brill).
    heldout_data = []
    sent = []
    # pos2 = -60000
    pos2 = pos1
    pos1 = -60000
    while words[pos1] != '###':
        pos1 += 1
    pos1 += 1  # skip first '#'

    # end_train_data = pos1   # mark pos ??????

    for word, tag in zip(words[pos1:pos2], tags[pos1:pos2]):
        # for word, tag in zip(words[pos2:pos1 - 1], tags[pos2:pos1 - 1]):
        if word == '###':
            if sent:
                heldout_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))

    # Get initial train data, the first ~10k
    train_data = []
    sent = []
    pos2 = 10000
    pos1 = 0
    while words[pos1] != '###':
        pos1 += 1
    pos1 += 1  # this is the first sentence in set
    while words[pos2] != '###':
        pos2 += 1
    for word, tag in zip(words[pos1:pos2], tags[pos1:pos2]):
        if word == '###':
            if sent:
                train_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))

    # # Strip off tags from remaining training data to train with
    # # BW, i.e. initial 10000 to -60000 words
    # data = []
    # sent = []
    # pos1 = pos2     # index = ~10000
    # # pos2 = end_train_data
    # pos2 = end_train_data
    #
    # while words[pos1] != '###':
    #     pos1 += 1
    # pos1 += 1  # this is the first sentence in set
    #
    # # OJO - FIX later
    # while words[pos2] != '###':
    #     pos2 += 1
    #
    # for word in words[pos1:pos2]:
    #     if word == '###':
    #         data.append(sent)
    #         sent = []
    #         continue
    #     sent.append(word)    # heldout_data = [heldout_data]

    data = ['I am tired .'.split(),
            'You can meet me in the bathroom .'.split()]

    # data = ['I am tired .'.split(),
    #         'You suck dick .'.split(),
    #         'I need to finish .'.split(),
    #         'Jack is a man !'. split(),
    #         'He left the store, then .'.split()]



    # print(data)

    # Train params using BW

    # Init and Train model
    print(pos1, pos2)
    print(len(train_data))
    print(len(data))
    print(end_train_data)

    tagger = HMMTagger.train(train_data)

    # ***Smooth params/Heldout***

    # params = 'heldout_data = None, mode = None'
    # tagger.initialize_params(heldout_data=None, mode=None)
    # write_results(tagger, test_data, params)
    #
    # tagger.initialize_params(heldout_data=None, mode='unk_threshold')
    # params = "heldout_data = None, mode = 'unk_threshold'"
    # write_results(tagger, test_data, params)

    tagger.initialize_params(heldout_data, mode='unk_heldout')

    print('init training')
    # print(tagger.tag('You are my man'.split()))

    # print(tagger._emission_probs)
    # print(tagger._lambdas)
    # def write_results(model, gold, label, fname, interpolate=False):


    print('unsupervised training')
    iter = 1
    if end_train_data < 0:
        size = len(words) - abs(end_train_data) - 10000
    else:
        size = abs(end_train_data - 10000)
    prev_accuracy = 0.0

    # Train new tagger, init with lambdas
    # new_tagger = tagger.train_unsupervised(data, interpolate=True)

    # Load pickle
    p_file = open('_en_size_10000_iter_2.pkl', 'rb')
    new_tagger = pickle.load(p_file)

    label = lang + 'size_' + str(size)
    fname = 'results-pic_bw' + label + '.txt'
    accuracy = write_results(new_tagger, heldout_data, label, fname, iter, interpolate=False)
    pickle._dump(new_tagger, open(label + '_iter_' + str(iter) + '.pkl', 'wb'))
    print(iter, accuracy)

    # print('the', new_tagger._uni_transitions['the'])

    e = .001
    while abs(accuracy - prev_accuracy) > e:
        prev_accuracy = accuracy
        iter += 1
        new_tagger = tagger.train_unsupervised(data, interpolate=False)
        accuracy = write_results(new_tagger, heldout_data, label, fname, iter, interpolate=False)
        pickle._dump(new_tagger, open(label+ '_iter_' + str(iter) + '.pkl', 'wb'))
        print(iter, accuracy)

        # print('the', new_tagger._uni_transitions['the'])


        # print(new_tagger._tags)
    # print(new_tagger._words)
    # print(new_tagger._n)
    # print(new_tagger._emiss_probs)
    # print(new_tagger._tri_tag_probs)
    # print(new_tagger._tri_transition_counts)

    # params = "heldout_data, mode = 'unk_heldout'"
    # tagger.initialize_params(heldout_data, mode='unk_heldout')
    # write_results(tagger, test_data, params)

    # print(tagger._emissions)
    # print(tagger._emission_probs)
    # print(tagger._lambdas)
    #





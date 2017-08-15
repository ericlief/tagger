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

        # self._trainModel = LanguageModel(train)         # training model
        # self._testModel = LanguageModel(test)           # test model
        # self._heldoutModel = LanguageModel(heldout.txt)     # heldout.txt model used for smoothing params
        self._lambdas = [0, 0, 0, 0]                    # smoothing params (lambdas)

    # @classmethod
    # def train(cls, train_data):
    #     # tagger = HMMTagger()
    #     # tagger._words = set([])
    #     # tagger._tags = set([])
    #     # tagger._emissions = defaultdict(int)
    #     # tagger._uni_transitions = defaultdict(int)
    #     # tagger._bi_transitions = defaultdict(int)
    #     # tagger._tri_transitions = defaultdict(int)
    #     tagger = HMMTagger()
    #     tagger._words = set([])
    #     cls._tags = set([])
    #     tagger._emissions = defaultdict(int)
    #     tagger._uni_transitions = defaultdict(int)
    #     tagger._bi_transitions = defaultdict(int)
    #     tagger._tri_transitions = defaultdict(int)
    #
    #     cls.test = []
    #     # # Handle beginning of text
    #     # first = '<s>'
    #     # second = '<s>'
    #
    #     print(train_data)
    #     # Update dictionaries with tag transition distributions
    #     for sent in train_data:
    #
    #         # Handle beginning of sentence
    #         first = '<s>'
    #         second = '<s>'
    #         tagger._bi_transitions[first, second] += 1
    #
    #         # if first == '<s>' and second == '<s>':  # add bigram for nulls
    #
    #         for word, tag in sent:
    #             third = tag
    #             tagger._tri_transitions[first, second, third] += 1
    #             tagger._bi_transitions[second, third] += 1
    #             # if first == '<s>' and second == '<s>':  # add bigram for nulls
    #             #     tagger._bi_transitions[('<s>', '<s>')] += 1
    #             tagger._uni_transitions[third] += 1
    #             tagger._emissions[word, tag] += 1
    #             tagger._tags.add(tag)
    #             tagger._words.add(word)
    #             first = second
    #             second = third
    #
    #         # Handle end of sentence
    #         tagger._tri_transitions[first, second, '</s>'] += 1
    #         tagger._bi_transitions[first, second] += 1
    #         tagger._tri_transitions[second, '</s>', '</s>'] += 1
    #         tagger._bi_transitions[second, '</s>'] += 1
    #
    #     return tagger
    #

    @classmethod
    def train(cls, train_data):
        # tagger = HMMTagger()
        # tagger._words = set([])
        # tagger._tags = set([])
        # tagger._emissions = defaultdict(int)
        # tagger._uni_transitions = defaultdict(int)
        # tagger._bi_transitions = defaultdict(int)
        # tagger._tri_transitions = defaultdict(int)
        cls._words = set([])
        cls._uni_words = defaultdict(int)
        cls._tags = set([])
        cls._emission_counts = defaultdict(int)
        cls._uni_transition_counts = defaultdict(int)
        cls._bi_transition_counts = defaultdict(int)
        cls._tri_transition_counts = defaultdict(int)

        # # Handle beginning of text
        # first = '<s>'
        # second = '<s>'

        # print(train_data)

        # Update dictionaries with tag transition distributions
        n = 0   # count word-tags
        for sent in train_data:

            # Handle beginning of sentence
            first = '<s>'
            second = '<s>'
            cls._bi_transition_counts[first, second] += 1
            cls._uni_transition_counts[first] += 1

            # if first == '<s>' and second == '<s>':  # add bigram for nulls

            for word, tag in sent:
                third = tag
                cls._tri_transition_counts[first, second, third] += 1
                cls._bi_transition_counts[second, third] += 1
                # if first == '<s>' and second == '<s>':  # add bigram for nulls
                #     tagger._bi_transitions[('<s>', '<s>')] += 1
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
            # cls._bi_transitions[first, second] += 1
            # cls._tri_transitions[second, '</s>', '</s>'] += 1
            cls._bi_transition_counts[second, '</s>'] += 1

            cls._n = n

        # print(cls._emissions)
        return cls

    # @classmethod
    # def calc_tag_probs(cls):
    #     """
    #     Iterate through counts and calculate the respective
    #     tag transition probabilities
    #     """
    #     cls._tri_tag_probs = defaultdict(lambda: 0.0)
    #     cls._bi_tag_probs = defaultdict(lambda: 0.0)
    #     cls._uni_tag_probs = defaultdict(lambda: 0.0)
    #
    #     for (t, u, v), cnt in cls._tri_transitions.items():
    #         print(t, u, v, cnt)
    #         cls._tri_tag_probs[t, u, v] = float(cls._tri_transition_counts[t, u, v] / cls._bi_transition_counts[t, u])
    #         cls._bi_tag_probs[u, v] = float(cls._bi_transitions_counts[u, v] / cls._uni_transitions_counts[u])
    #         cls._uni_tag_probs[v] = float(cls._uni_transitions_counts[v] / cls._n)
    #
    #     # print(cls._tri_tag_probs)
    #     # print(cls._bi_tag_probs)
    #     # print(cls._uni_tag_probs)

    @classmethod
    def bi_tag(cls, corpus):
        words = [w for sent in corpus for w in sent]
        # print(words)
        n = len(words)
        viterbi = {}
        path = {}

        # Initialization step
        w = words[0]
        for t in cls._tags:
            # viterbi[1, t] = cls._bi_transitions['<s>', t] * float(cls._emissions[word, t]/cls._uni_transitions[t])
            viterbi[1, t] = cls._bi_transition_counts['<s>', t] * cls.calculate_emission_prob(w, t)

            path[1, t] = []
            # path[t] = []

            # print('init', viterbi)

        # Recursion step
        for i in range(2, n+1):
            # temp_path = {}

            w = words[i-1]
            if w not in cls._words:
                # print('unk word @', i, w)
                w = '<unk>'

        # for i, w in enumerate(words):
        #     if i == 0:
                # t = '<s>'
                # u = '<s>'
                #
            # for t in cls._tags:
            #     for u in cls._tags:
            #         if t == '<s>' and u == '<s>':
            #             pass

            for v in cls._tags:
                # print('VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV', v, w)
                # j = 0
                # for u in cls._tags:
                #
                #     # if j > 5:
                #     #     break
                #     # print(u, viterbi[i-1, u] * float(cls._bi_transitions[u, v]/cls._uni_transitions[u])
                #     #                            * float(cls._emissions[w, v]/cls._uni_transitions[v]))
                #     print(w, u, v, viterbi[i - 1, u] * float(cls._bi_transitions[u, v] / cls._uni_transitions[u])
                #           * cls.emission_prob(w, v))
                #     # j+=1
                #
                #     # print('v=', v)
                viterbi[i, v], u_max = max([(viterbi[i-1, u] * float(cls._bi_transitions[u, v]/cls._uni_transitions[u])
                                             * cls.calculate_emission_prob(w, v), u) for u in cls._tags])

                # print('u_max=', u_max)
                # path[i, v] = path[i-1, u_max] + [u_max]
                path[i, v] = u_max
                # print('path', path[i,v])

                # temp_path[v] = path[u_max] + [v]

                # if u_max == '``':
                #     print('empty string')

            # path = temp_path

        # Final step, to sentence
        viterbi[n, '</s>'], u_max = max([(viterbi[n, u] * float(cls._bi_transitions[u, '</s>']/cls._uni_transitions[u]), u)
                           for u in cls._tags])
        # path[n, '</s>'] = path[n, u_max] + [u_max]
        path[n, '</s>'] = u_max

        print(viterbi)
        print(cls._emissions)
        return HMMTagger.backtrace(path, n)

    @classmethod
    def tri_tag(cls, sent, k):

        n = len(sent)
        # k = 10              # this is k_best beam search
        L = cls._lambdas
        viterbi = {}
        path = {}

        # Initialization step
        w1 = sent[0]
        w2 = sent[1]
        viterbi[0, '<s>', '<s>'] = 1
        path[0, '<s>', '<s>'] = []
        #path['', ''] = []
        active_tags_t = set(['<s>'])
        active_tags_u = set(['<s>'])

        for i in range(1, n + 1):
            w = cls.next_word(sent, i - 1)
            k_best = {}
            # back_ptr = {}
            if w not in cls._words:
                # print('unk word @', i, w)
                w = '<unk>'

            # for v in cls.possible_tags(i):
            for v in cls._tags:
                # for u in cls.possible_tags(i - 1):
                for u in active_tags_u:

                    # viterbi[i, u, v], t_max = max([(
                    #      viterbi[i - 1, t, u] * cls.calculate_interpolated_p(t, u, v, L) * cls._emission_probs[w, v], t)
                    #      for t in cls.possible_tags(i - 2)])
                    score, t_max = max([(
                         viterbi[i - 1, t, u] * cls.calculate_interpolated_p(t, u, v, L) * cls._emission_probs[w, v], t)
                         for t in active_tags_t])
                    viterbi[i, u, v] = score
                    path[i, u, v] = t_max
                    k_best[u, v] = score

            best = sorted(k_best, key=k_best.get, reverse=True)[:k]
            # active_tags_t = active_tags_u
            active_tags_t = [t for (t, u) in best]
            active_tags_u = [u for (t, u) in best]

                    #back_ptr[u, v] = path[t_max, u] + [v]
            # path = back_ptr

        # Final step, to sentence

        # print(n,i, )
        # prob, u_max, v_max = mpathax([(viterbi[n, t, u] * cls.calculate_interpolated_p(t, u, '</s>', L), t, u)
        #                         for t in cls._tags for u in cls._tags])

        # fixed
        # prob, t_max, u_max = max([(viterbi[n, t, u] * cls.calculate_interpolated_p(t, u, '</s>', L), t, u)
        #                                       for t in cls._tags for u in cls._tags])
        score, t_max, u_max = max([(viterbi[n, t, u] * cls.calculate_interpolated_p(t, u, '</s>', L), t, u)
                                  for t in active_tags_t for u in active_tags_u])
        viterbi[n+1, u_max, '</s>'] = score
        path[n+1, u_max, '</s>'] = t_max

        # print(u_max, v_max, max([(viterbi[n, t, u] * cls.calculate_interpolated_p(t, u, '</s>', L), t, u)
        #                         for t in cls._tags for u in cls._tags]))
        # for t in cls._tags:
        #     for u in cls._tags:
                # print(viterbi[n,t,u] * cls.calculate_interpolated_p(t, u, '</s>', L))
        # cls._viterbi = viterbi
        # cls._path = path
        # print(viterbi)
        # print(path)
        # return path[u_max, v_max]
        # return HMMTagger.backtrace(path, n+1, u_max)
        return HMMTagger.backtrace(path, n+1, u_max)

    @staticmethod
    def backtrace(path, n, u):
        # print(path)
        # tags = []
        tags = [u]
        # print(u, end=' ')
        i = n
        v = '</s>'
        while i > 2:
            t = path[i, u, v]

            print(t, end=' ')
            tags.insert(0, t)
            v = u
            u = t
            i -= 1
        # print()
        return tags

    # @classmethod
    # def possible_tags(cls, i):
    #     if i == -1:
    #         return set(['<s>'])
    #     if i == 0:
    #         return set(['<s>'])
    #     else:
    #         return cls._tags

    @classmethod
    def next_word(cls, sent, i):
        if i < 0:
            return ''
        else:
            return sent[i]

    @classmethod
    def bi_tag2(cls, corpus):
        words = [w for sent in corpus for w in sent]
        # print(words)
        n = len(words)
        viterbi = {}
        path = {}

        # Initialization step
        word = words[0]
        for t in cls._tags:
            viterbi[1, t] = cls._bi_transitions['<s>', t] * cls._emissions[word, t] / cls._uni_transitions[t]
            path[1, t] = []
            # print('init', viterbi)

        # Recursion step
        # for i in range(2, n+1):
        for i in range(2, 5):

            word = words[i - 1]
            # print(i, word)
            # for i, w in enumerate(words):
            #     if i == 0:
            # t = '<s>'
            # u = '<s>'
            #
            # for t in cls._tags:
            #     for u in cls._tags:
            #         if t == '<s>' and u == '<s>':
            #             pass


            for v in cls._tags:
                print('VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV', v, word)
                j = 0
                for u in cls._tags:
                    if j > 5:
                        break
                    # print(u, viterbi[i - 1, u] * cls._bi_transitions[u, v] / cls._uni_transitions[u]
                    #       * cls._emissions[word, v] / cls._uni_transitions[v])
                    j += 1

                    # print('v=', v)
                    # viterbi[(i, t)], argmax = max([(viterbi[(i-1, u)] * cls.bi_transitions[(u, v)]/cls._uni_transitions[u], u)
                    #                                for u in cls._tags]) * cls._emissions[word, v]
                    # viterbi[i, v], u_max = max([(viterbi[i-1, u] * cls._bi_transitions[u, v]/cls._uni_transitions[u]
                    #                                * cls._emissions[word, v]/cls._uni_transitions[v], u) for u in cls._tags])
                    #
                    # print('u_max=', u_max)
                    # path[i, v] = path[i-1, u_max] + [u_max]
                    # print('path', path[i,v])

                    # if u_max == '``':
                    #     print('empty string')

        # Final step, to sentence
        viterbi[n, '</s>'], u_max = max(
            [(viterbi[n, u] * float(cls._bi_transitions[u, '</s>'] / cls._uni_transitions[u]), u)
             for u in cls._tags])

        path[n, '</s>'] = path[n, u_max] + [u_max]

        print(viterbi)

        return path[n, '</s>']

    # @classmethod
    # def emission_prob(cls, w, t):
    #     """
    #     Calculate smoothed emission (lexical) probability.
    #     :param w:
    #     :param t:
    #     :return: p
    #     """
    #     n = sum(cls._emissions.values())
    #     # v = len(cls._words)
    #     v = len(cls._uni_words)
    #
    #     # c_w = cls._emissions[w, t]
    #     c_w = cls._uni_words[w]
    #
    #     c_t = cls._uni_transitions[t]
    #     lex_p = c_w + 1 / (n + v)
    #
    #     return float(lex_p / c_t)

    @classmethod
    def calculate_emission_prob(cls, w, t):
        """
        Helper method to calculate the smoothed emission
        (lexical) probability.
        :param w:
        :param t:
        :return: p
        """
        #
        # p_w_t = cls._emission_counts[w, t]
        # p_t = cls._uni_transition_counts[t]

        return float(cls._emission_counts[w, t] / cls._uni_transition_counts[t])

    @classmethod
    def smooth_unkown_words_with_threshold(cls, threshold=10):
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

        return smoothed_probs

    @classmethod
    def smooth_unknown_words_from_heldout(cls, heldout_data):
        """
        :param heldout_data:
        :return: smoothed emission params
        """
        # print('smoothing')
        # print(heldout_data)
        unk = defaultdict(int)
        tags = defaultdict(int)
        n = 0   # size of data
        unk_cnt = 0
        cnt = 0
        for sent in heldout_data:
            for w, t in sent:
                # if t == 'NNS':
                #     cnt += 1
                if w not in cls._words:
                # if (w, t) not in cls._emissions:
                    unk['<unk>', t] += 1
                    unk_cnt += 1
                    # print('adding unkown word', w,t)
                tags[t] += 1
                n += 1
        # print('number of unk words', unk_cnt)
        # print('size of heldout', n)
        unk_mass = unk_cnt / n
        # print('total mass', unk_mass)    # = P(unk|t)
        # print('nns', cnt)
        # print(unk[('<unk>', 'NNS')])

        # unk_probs = {}
        smoothed_probs = defaultdict(lambda: 0.0)

        for w, t in unk:
            # unk_probs[w, t] = unk[w, t] / n
            smoothed_probs[w, t] = unk[w, t] / tags[t]

        # total_unk_prob_mass = sum(smoothed_probs.values())
        # print('prob mass', total_unk_prob_mass)
        # print(smoothed_probs)
        # n = sum(cls._emissions.values())
        # v = len(cls._uni_words)

        # smoothed_probs = defaultdict(lambda: 0.0)
        # cls._e_prob = defaultdict(lambda: 0.0)
        # for (w, t), c in cls._emissions.items():
        for w, t in cls._emission_counts:

            smoothed_probs[w, t] = float(cls._emission_counts[w, t] / cls._uni_transition_counts[t]) * (1 - unk_mass)

        # # add new unkown word/tags
        # for w, t in unk:
        #     smoothed_probs[w, t] = unk_probs[w, t]

        # print(smoothed_probs['<unk>', 'NNS'])
        # print(unk_mass)
        return smoothed_probs
        # cls._words.add('<unk>')


            # n = sum(cls._emissions.values())
        # v = len(cls._uni_words)
        #
        # # cls._e_prob = defaultdict(lambda: 0.0)
        # for (w, t), c in cls._emissions.items():
        #     smoothed[w,t] = (c + 1) * float(n / (n + v))
        #
        # cls._emissions = smoothed

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
                # bi_transition_counts[t, u] += 1

                # if first == '<s>' and second == '<s>':  # add bigram for nulls

                for word, tag in sent:
                    v = tag
                    # tri_transitions[t, u, v] += 1
                    # bi_transitions[u, v] += 1
                    # # if first == '<s>' and second == '<s>':  # add bigram for nulls
                    # #     tagger._bi_transitions[('<s>', '<s>')] += 1
                    # uni_transitions[v] += 1
                    # cls._emissions[word, tag] += 1
                    # tags.add(v)
                    # words.add(word)
                    # uni_words[word] += 1


                    # Calculate expected counts of lambdas
                    ratio = cls.calc_tag_ratio(t, u, v, L)

                    # M-step (Step 2)
                    # Calculate expected counts of lambdas, i.e. weight, taking
                    # into account the number of occurrences of each trigram (cnt)
                    for j in range(len(L)):
                        counts[j] += ratio[j]  # weight of lambda in whole equation (count)

                    t = u
                    u = v
                    # n += 1

                # Handle end of sentence
                # tri_transitions[t, u, '</s>'] += 1
                v = '</s>'
                ratio = cls.calc_tag_ratio(t, u, v, L)
                for j in range(len(L)):
                    counts[j] += ratio[j]  # weight of lambda in whole equation (count)

                # cls._bi_transitions[first, second] += 1
                # cls._tri_transitions[second, '</s>', '</s>'] += 1


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
        using training data
        :param t:
        :param u:
        :param v:
        :param L:
        :return: weighted p
        """

        ratio = [0.0, 0.0, 0.0, 0.0]
        # V = len(cls._uni_tag_probs)     # tag vocabulary size
        # weighted_p = L[3] * cls._tri_tag_probs[w1, w2, w3] + L[2] * cls._bi_tag_probs[w2, w3] \
        #          + L[1] * cls._uni_tag_probs[w3] + L[0] / V
        #

        weighted_p = cls.calculate_interpolated_p(t, u, v, L)
        V = len(cls._uni_tag_probs)     # tag vocabulary size

        ratio[3] = L[3] * cls._tri_tag_probs[t, u, v] / weighted_p  # ratio of p3/p' to model distribution function
        ratio[2] = L[2] * cls._bi_tag_probs[u, v] / weighted_p
        ratio[1] = L[1] * cls._uni_tag_probs[v] / weighted_p
        ratio[0] = L[0] / V / weighted_p

        # result = L[3] * self._trainModel.p3(w1, w2, w3) + L[2] * self._trainModel.p2(w2, w3) \
        #        + L[1] * self._trainModel.p1(w3) + L[0] * self._trainModel.p0()
        return ratio

    @classmethod
    def calculate_interpolated_p(cls, t, u, v, L):
        """
        Computes the smoothed (weighted/interpolated) probability
        P' using the distribution calculated for the language model
        using training data
        :param t: w_i-2
        :param u: w_1-1
        :param v: w_i
        :param L: lambdas
        :return: P'
        """
        # L = cls._lambdas
        V = len(cls._uni_tag_probs)     # tag vocabulary size
        interpolated_p = L[3] * cls._tri_tag_probs[t, u, v] + L[2] * cls._bi_tag_probs[u, v] \
                     + L[1] * cls._uni_tag_probs[v] + L[0] / V

        return interpolated_p

    @classmethod
    def initialize_params(cls, heldout_data, mode=None):

        # Iterate through counts and calculate the respective
        # tag transition probabilities
        cls._tri_tag_probs = defaultdict(lambda: 0.0)
        cls._bi_tag_probs = defaultdict(lambda: 0.0)
        cls._uni_tag_probs = defaultdict(lambda: 0.0)

        for t, u, v in cls._tri_transition_counts:
            # print(t, u, v, cnt)
            cls._tri_tag_probs[t, u, v] = float(cls._tri_transition_counts[t, u, v] / cls._bi_transition_counts[t, u])
            cls._bi_tag_probs[u, v] = float(cls._bi_transition_counts[u, v] / cls._uni_transition_counts[u])
            cls._uni_tag_probs[v] = float(cls._uni_transition_counts[v] / cls._n)

        # Smooth lexical model
        if mode == 'unk_heldout':
            cls._emission_probs = cls.smooth_unknown_words_from_heldout(heldout_data)
        elif mode == 'unk_threshold':
            cls._emission_probs = cls.smooth_unkown_words_with_threshold()
        elif mode == None:
            cls._emission_probs = defaultdict(lambda : 0.0)
            # Calculate emission probabilities
            for w, t in cls._emission_counts:
                # unk_probs[w, t] = unk[w, t] / n
                cls._emission_probs[w, t] = cls.calculate_emission_prob(w, t)
        else:
            raise Exception

        # Smooth tag model
        cls._lambdas = cls.smooth_tag_model(heldout_data)

        # # Recalculate smoothed interpolated trigram prob
        # cls._interpolated_tag_probs = defaultdict(lambda: 0.0)
        # for t, u, v in cls._tri_tag_probs:
        #     # print(t, u, v, cnt)
        #     cls._interpolated_tag_probs[t, u, v] = cls.calculate_interpolated_p(t, u, v, cls._lambdas)

    def crossEntropy(self, L):
        """
        Compute the cross entropy H or negative
        log likelihood of test data using
        the new smoothed language model.

        :param L: lambdas
        :return: H
        """

        # Iterate through trigrams in model
        H = 0  # cross entropy
        for (w1, w2, w3), cnt in self._testModel.trigrams.items():
            # log likelihood/cross-entropy
            H += cnt * np.log2(self._p_(w1, w2, w3, L))  # this is p' (model), multiplied by the count of the trigrams

        H *= -1.0 / self._testModel.T  # per word cross entropy/negative log likelihood
        return H



if __name__ == '__main__':

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

    # print(tags)
    # Split data for initial part
    # T-H-V
    # rest-20-40

    # print('initial split')

    # Get test data, the last 40k
    test_data = []
    sent = []
    pos1 = -40000
    # pos1 = -20
    while words[pos1] != '###':
        pos1 += 1
    pos1 += 1    # skip first '#'
    for word, tag in zip(words[pos1:], tags[pos1:]):
        if word == '###':
            test_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))

    #
    # test_data = [test_data]     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

    # Get heldout.txt data, the mid 20k (not used in brill).
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
        sent.append((word, tag))    # heldout_data = [heldout_data]

    # print(heldout_data)
    #
    # # Get train data, the first ~20k
    # for word, tag in zip(words[:-60000], tags[:-60000]):
    #     train_data.append((word, tag))
    # train_data = [train_data]     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

    # Train
    # Get train data, the first ~40k
    train_data = []
    sent = []
    pos3 = 40000
    # pos3 = 40
    try:
        while words[pos3] != '###':
            pos3 += 1
    except IndexError:
        print(sys.stderr)
    pos3 += 1  # this is the final sentence in set
    for word, tag in zip(words[:pos3], tags[:pos3]):
        if word == '###':
            train_data.append(sent)
            sent = []
            continue
        sent.append((word, tag))    # heldout_data = [heldout_data]

    # for word, tag in zip(words[:10000], tags[:10000]):
    #     train_data.append((word, tag))
    # train_data = [train_data]  # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)


    # tagger._tri_tag_probs
    # tagger._emission_probs

    # print(tagger._tri_transitions)
    # print(tagger._bi_transitions)

    # n = sum(tagger._emissions.values())
    # v = len(tagger._uni_words)
    # print('n=', n, '\tv=', v)
    # print(tagger._uni_words)

    # For test
    # heldout_data = []
    # with open('heldout.txt', 'r') as f:
    #     sent = []
    #     first = f.readline().rstrip()  # get first '###'
    #     if first != '###':
    #         f.seek(0)
    #     for line in f:
    #         if line == '\n':
    #             continue
    #         # word, tag = line.rstrip().split('/')
    #         word = line.rstrip()
    #         if word == '###':
    #             heldout_data.append(sent)
    #             sent = []
    #             continue
    #         sent.append(word)
    #
    #
        # heldout_data.append(sent)
        #         sent = []
        #         continue
        #     sent.append((word, tag))    # heldout_data = [heldout_data]
    # heldout_data = [heldout_data]
    # print(heldout_data[:5])
    #     if word == '###':
    #         sent = []
    #         test_data.append(sent)
    #         continue
    #     sent.append((word, tag))
    #
    #
    # tagger.smooth_lexical(heldout_data)
    # L = tagger.smooth_tag_model(heldout_data)

    tagger = HMMTagger.train(train_data)
    tagger.initialize_params(heldout_data, mode='unk_threshold')
    # tagger.initialize_params(heldout_data, mode='unk_heldout')
    # tagger.initialize_params(heldout_data)
    print(tagger._lambdas)

    # print(tagger._emissions)

    # tagger = HMMTagger.train(test_data)

    # Unit test
    # print(list(tagger._tri_transitions.items())[:5])
    # print(tagger._tri_transitions[('NN', 'DT', 'VBZ')])
    # print(tagger._tri_transitions[('IN', 'DT', 'NN')])
    # for k, v in tagger._tri_transitions.items():
    #     if 'IN' in k and 'DT' in k:
    #         print(k, v)
    #
    # print(tagger._bi_transitions[('<s>','<s>')])
    # print(tagger._bi_transitions[('JJ','NNS')])
    #

    # print(list(tagger._bi_transitions.items())[:5])
    # print(list(tagger._uni_transitions.items())[:5])
    # print(tagger._tags)
    # print(tagger._words)
    #print(tagger._emissions)

    # open unittest test data
    # untagged_sents = []
    # tagged_sents = []
    # n = 0
    # with open('gold-cz.txt', 'r') as f:
    #     untagged_sent = []
    #     tagged_sent = []
    #     first = f.readline().rstrip()           # get first '###'
    #     if first != '###/###':
    #         f.seek(0)
    #     for line in f:
    #         # print(line)
    #         if line == '\n':
    #             continue
    #         # word, tag = line.rstrip().split('/')
    #         # word = line.rstrip()
    #         word, tag = line.rstrip().split('/')
    #         if word == '###':
    #             untagged_sents.append(untagged_sent)
    #             untagged_sent = []
    #             tagged_sents.append(tagged_sent)
    #             tagged_sent = []
    #             continue
    #         untagged_sent.append(word)
    #         tagged_sent.append((word, tag))
    #         n += 1


    # Use test data
    untagged_sents = []
    tagged_sents = test_data
    n = 0
    for tagged_sent in tagged_sents:
        untagged_sent = []
        for word, tag in tagged_sent:
            untagged_sent.append(word)
            n += 1
        untagged_sents.append(untagged_sent)

    # print(untagged_sents[0])
    # print(untagged_sents[-1])
    # print(heldout_data[0])
    # print(heldout_data[-1])


    # strip test data, the last 20k
    # for word, tag in zip(words[-40000:], tags[-40000:]):
    #     test_data.append((word, tag))

    # untagged = words[:20]     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)
    # untagged = words     # convert to nltk list(list(tuples)), i.e. a list of list of sentences (here only one large sentence)

    # print(untagged)
    # print(tagger._emissions)
    # tagger.smooth_lexical()
    # print(tagger._emissions)

    # tags = tagger.bi_tag(untagged)

    # for sent in untagged:
    #
    #     tags = tagger.tri_tag(sent)
    #     print(sent)
    #     print(tags)

    # print(tagger._tags)
    # print(tagger._emissions)
    # print(tagger._bi_transitions)
    # print(untagged)

    # print(untagged_sents)
    # print(tagged_sents)

    with open('tagged.txt', 'w') as f:
        correct = 0
        for i, sent in enumerate(untagged_sents):
            f.write('sentence ' + str(i) + '\n')
            f.write(' '.join(sent) + '\n')
            tags = tagger.tri_tag(sent, k=8)
            # print(sent)
            # print(tags)
            for j, (w, t) in enumerate(zip(sent, tags)):
                f.write(str(w) + '/' + str(t) + '\n')
                if tagged_sents[i][j] == (w, t):
                    f.write('correct' + str(tagged_sents[i][j]) + '\t' + str((w, t)) + '\n')
                    correct += 1
                else:
                    f.write('incorrect' + str(tagged_sents[i][j]) + '\t' + str((w, t)) + '\n')
        # print(correct)
        # print(n)
        acc = correct / n
        f.write('Accuracy = ' + str(acc) + '\n')
                    # te('viterbi')
            #     f.write(str(x) + '\n')
            # for x in tagger._path.items():
            #     f.write('path')
            #     f.write(str(x) + '\n')


    # print(train_data[:5])
    # print(heldout_data)
    # print(tagger._bi_transition_counts['.', '</s>'])




        # viterbi[n, u, '</s>'], t_max, u_max = max(
        #     [(viterbi[n, t, u] * cls.calculate_interpolated_p(t, u, '</s>', L), t, u) for t in cls._tags for u in cls._tags])
        #
        # # path[n, '</s>'] = path[n, u_max] + [u_max]
        # path[n, u_max, '</s>'] = t_max
        # return HMMTagger.backtrace(path, n, u_max)  # @classmethod


            # print(viterbi[1, '<s>', u], viterbi[2, u, v])
            #
            # print('init PRP', viterbi[1, '<s>', 'PRP'])
            # print('interp', cls.calculate_interpolated_p('<s>', '<s>', 'PRP', L))
            # print('emission', cls._emission_probs['I', 'PRP'])
            # print(cls._emission_counts['I', 'PRP'])
            # print(cls._uni_transition_counts['PRP'])
            # print('init PRP', viterbi[2, 'PRP', 'VB'])
            # print('interp', cls.calculate_interpolated_p('<s>', 'PRP', 'VB', L))
            # print('emission', cls._emission_probs['like', 'VB'])
            # print(cls._emission_counts['like', 'VB'])
            # print(cls._uni_transition_counts['VB'])
            # print(viterbi[2, '``', 'PRP'])
            # print(path[2, '``', 'PRP'])
            # print(cls._tags)


    # def tri_tag2(cls, sent):
    #
    #     print(sent)
    #
    #     # words = [w for sent in corpus for w in sent]
    #     # print(words)
    #     n = len(sent)
    #     L = cls._lambdas
    #     viterbi = {}
    #     path = {}
    #
    #     # Initialization step
    #     w1 = sent[0]
    #     w2 = sent[1]
    #     stop = 0
    #
    #     for t in cls._tags:
    #
    #         viterbi[1, '<s>', t] = cls.calculate_interpolated_p('<s>', '<s>', t, L) * cls._emission_probs[w1, t]
    #
    #         path[1, '<s>', t] = ''
    #
    #     res = []
    #     for v in cls._tags:
    #         for u in cls._tags:
    #             # print(stop, u, v)
    #             viterbi[1, '<s>', u] * cls.calculate_interpolated_p('<s>', u, v, L) * cls._emission_probs[w2, v],
    #             v)
    #
    #             viterbi[2, u, v], t_max = max(
    #                 [(viterbi[1, '<s>', u] * cls.calculate_interpolated_p('<s>', u, v, L) * cls._emission_probs[w2, v],
    #                   v)
    #                  for v in cls._tags])
    #
    #             # print(w, u, v)
    #             # viterbi[1, t] = cls._bi_transitions['<s>', t] * float(cls._emissions[word, t]/cls._uni_transitions[t])
    #             # viterbi[1, '<s>', u] = cls._tri_transitions['<s>', '<s>', u] * cls.emission_prob(w, u)
    #             # viterbi[1, '<s>', u] = cls._tri_tag_probs['<s>', '<s>', u] * cls.emission_prob(w, u)
    #             # viterbi[1, '<s>', u] = cls.calculate_interpolated_p('<s>', '<s>', u, L) * cls._emission_probs[w1, u]
    #             #
    #             # path[1, '<s>', u] = ''
    #             #
    #             # # path[t] = []
    #             # viterbi[2, u, v] = viterbi[1, '<s>', u] * \
    #             #                    float(cls._tri_transitions['<s>', u, v]/cls._bi_transitions['<s>', u]) * cls.emission_prob(w, v)
    #             # viterbi[2, u, v] = viterbi[1, '<s>', u] * cls.calculate_interpolated_p('<s>', u, v, L) * cls._emission_probs[w2, v]
    #             # path[2, u, v] = ''
    #             #
    #             #
    #             # print('u_max=', t_max)
    #             # path[i, v] = path[i-1, u_max] + [u_max]
    #
    #
    #             path[i, u, v] = t_max
    #
    #
    #
    #             print(viterbi[1,'<s>',u], viterbi[2,u,v])
    #
    #             stop += 1
    #
    #     # for v in cls._tags:
    #     #
    #     #     for u in cls._tags:
    #     #         print(stop, u, v)
    #     #
    #     #
    #     #
    #     #         # print(w, u, v)
    #     #         # viterbi[1, t] = cls._bi_transitions['<s>', t] * float(cls._emissions[word, t]/cls._uni_transitions[t])
    #     #         # viterbi[1, '<s>', u] = cls._tri_transitions['<s>', '<s>', u] * cls.emission_prob(w, u)
    #     #         # viterbi[1, '<s>', u] = cls._tri_tag_probs['<s>', '<s>', u] * cls.emission_prob(w, u)
    #     #         viterbi[1, '<s>', u] = cls.calculate_interpolated_p('<s>', '<s>', u, L) * cls._emission_probs[w1, u]
    #     #
    #     #         path[1, '<s>', u] = ''
    #     #
    #     #         # path[t] = []
    #     #         # viterbi[2, u, v] = viterbi[1, '<s>', u] * \
    #     #         #                    float(cls._tri_transitions['<s>', u, v]/cls._bi_transitions['<s>', u]) * cls.emission_prob(w, v)
    #     #         viterbi[2, u, v] = viterbi[1, '<s>', u] * cls.calculate_interpolated_p('<s>', u, v, L) * cls._emission_probs[w2, v]
    #     #         path[2, u, v] = ''
    #     #
    #     #         print(viterbi[1,'<s>',u], viterbi[2,u,v])
    #     #
    #     #         stop += 1
    #     #
    #     #     # if u == 'PRP':
    #     #     #     break
    #
    #
    #     print('init PRP', viterbi[1,'<s>','PRP'])
    #     print('interp', cls.calculate_interpolated_p('<s>', '<s>', 'PRP', L))
    #     print('emission', cls._emission_probs['I', 'PRP'])
    #     print(cls._emission_counts['I', 'PRP'])
    #     print(cls._uni_transition_counts['PRP'])
    #     print('init PRP', viterbi[2, 'PRP', 'VB'])
    #     print('interp', cls.calculate_interpolated_p('<s>', 'PRP', 'VB', L))
    #     print('emission', cls._emission_probs['like', 'VB'])
    #     print(cls._emission_counts['like', 'VB'])
    #     print(cls._uni_transition_counts['VB'])
    #     print(viterbi[2, '``', 'PRP'])
    #     print(path[2, '``', 'PRP'])
    #     print(cls._tags)
    #
    #     # print(viterbi))
    #     # Recursion step
    #     for i in range(3, n + 1):
    #         # temp_path = {}
    #
    #         w = sent[i - 1]
    #         if w not in cls._words:
    #             # print('unk word @', i, w)
    #             w = '<unk>'
    #
    #             # for i, w in enumerate(words):
    #             #     if i == 0:
    #             # t = '<s>'
    #             # u = '<s>'
    #             #
    #         # for t in cls._tags:
    #         #     for u in cls._tags:
    #         #         if t == '<s>' and u == '<s>':
    #         #             pass
    #
    #         for v in cls._tags:
    #             # print('VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV', v, w)
    #             # j = 0
    #             for u in cls._tags:
    #             #     # if j > 5:
    #             #     #     break
    #             #     # print(u, viterbi[i-1, u] * float(cls._bi_transitions[u, v]/cls._uni_transitions[u])
    #             #     #                            * float(cls._emissions[w, v]/cls._uni_transitions[v]))
    #             #     print(w, u, v, viterbi[i - 1, u] * float(cls._bi_transitions[u, v] / cls._uni_transitions[u])
    #             #           * cls.emission_prob(w, v))
    #             #     # j+=1
    #             #
    #             #     # print('v=', v)
    #             #     viterbi[i, u, v], t_max = max(
    #             #         [(viterbi[i - 1, t, u] * float(cls._tri_transitions[t, u, v] / cls._bi_transitions[t, u])
    #             #           * cls.emission_prob(w, v), t) for t in cls._tags])
    #                 viterbi[i, u, v], t_max = max(
    #                     [(viterbi[i - 1, t, u] * cls.calculate_interpolated_p(t, u, v, L) * cls._emission_probs[w, v], t)
    #                      for t in cls._tags])
    #
    #
    #             # print('u_max=', t_max)
    #                 # path[i, v] = path[i-1, u_max] + [u_max]
    #                 path[i, u, v] = t_max
    #             # print('path', path[i,v])
    #
    #             # temp_path[v] = path[u_max] + [v]
    #
    #             # if u_max == '``':
    #             #     print('empty string')
    #
    #             # path = temp_path
    #
    #     # Final step, to sentence
    #     # viterbi[n, u, '</s>'], t_max, u_max = max(
    #     #     [(viterbi[n, t, u] * float(cls._tri_transitions[t, u, '</s>'] / cls._bi_transitions[t, u]), t, u)
    #     #      for t in cls._tags for u in cls._tags])
    #     viterbi[n, u, '</s>'], t_max, u_max = max(
    #         [(viterbi[n, t, u] * cls.calculate_interpolated_p(t, u, '</s>', L), t, u) for t in cls._tags for u in cls._tags])
    #
    #     # path[n, '</s>'] = path[n, u_max] + [u_max]
    #     path[n, u_max, '</s>'] = t_max
    #
    #     # print(cls._tags)
    #
    #
    #     for i in range(2,3):
    #         for u in cls._tags:
    #             for v in cls._tags:
    #
    #                 print(i,u,v, repr(viterbi[i, u,v]))
    #
    #     # print(cls._emissions)
    #     return HMMTagger.backtrace(path, n, u_max)


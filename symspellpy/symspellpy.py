from collections import defaultdict, namedtuple
from enum import Enum
import gzip
from itertools import cycle
import math
import os.path
import pickle
import re
import sys

from symspellpy.editdistance import DistanceAlgorithm, EditDistance
import symspellpy.helpers as helpers

class Verbosity(Enum):
    TOP = 0 
    CLOSEST = 1  
    ALL = 2  

class SymSpell(object):
    data_version = 2
    N = 1024908267229
    bigram_count_min = sys.maxsize
    def __init__(self, max_dictionary_edit_distance=2, prefix_length=7,
                 count_threshold=1):
        if max_dictionary_edit_distance < 0:
            raise ValueError("max_dictionary_edit_distance cannot be "
                             "negative")
        if (prefix_length < 1
                or prefix_length <= max_dictionary_edit_distance):
            raise ValueError("prefix_length cannot be less than 1 or "
                             "smaller than max_dictionary_edit_distance")
        if count_threshold < 0:
            raise ValueError("count_threshold cannot be negative")
        self._words = dict()
        self._below_threshold_words = dict()
        self._bigrams = dict()
        self._deletes = defaultdict(list)
        self._max_dictionary_edit_distance = max_dictionary_edit_distance
        self._prefix_length = prefix_length
        self._count_threshold = count_threshold
        self._distance_algorithm = DistanceAlgorithm.DAMERUAUOSA
        self._max_length = 0
        self._replaced_words = dict()

    def create_dictionary_entry(self, key, count):
        if count <= 0:
            if self._count_threshold > 0:
                return False
            count = 0
        if self._count_threshold > 1 and key in self._below_threshold_words:
            count_previous = self._below_threshold_words[key]
            count = (count_previous + count
                     if sys.maxsize - count_previous > count
                     else sys.maxsize)
            if count >= self._count_threshold:
                self._below_threshold_words.pop(key)
            else:
                self._below_threshold_words[key] = count
                return False
        elif key in self._words:
            count_previous = self._words[key]
            count = (count_previous + count
                     if sys.maxsize - count_previous > count
                     else sys.maxsize)
            self._words[key] = count
            return False
        elif count < self._count_threshold:
            self._below_threshold_words[key] = count
            return False
        self._words[key] = count
        if len(key) > self._max_length:
            self._max_length = len(key)
        edits = self._edits_prefix(key)
        for delete in edits:
            self._deletes[delete].append(key)
        return True

    def delete_dictionary_entry(self, key):
        if key not in self._words:
            return False
        del self._words[key]
        if len(key) == self._max_length:
            self._max_length = max(map(len, self._words.keys()))
        edits = self._edits_prefix(key)
        for delete in edits:
            self._deletes[delete].remove(key)
        return True

    def load_bigram_dictionary(self, corpus, term_index, count_index,
                               separator=None, encoding=None):
        if not os.path.exists(corpus):
            return False
        with open(corpus, "r", encoding=encoding) as infile:
            for line in infile:
                line_parts = line.rstrip().split(separator)
                key = count = None
                if len(line_parts) >= 3 and separator is None:
                    key = "{} {}".format(line_parts[term_index],
                                         line_parts[term_index + 1])
                elif len(line_parts) >= 2 and separator is not None:
                    key = line_parts[term_index]
                if key is not None:
                    count = helpers.try_parse_int64(line_parts[count_index])
                if count is not None:
                    self._bigrams[key] = count
                    if count < self.bigram_count_min:
                        self.bigram_count_min = count
        return True

    def load_dictionary(self, corpus, term_index, count_index,
                        separator=" ", encoding=None):
        if not os.path.exists(corpus):
            return False
        with open(corpus, "r", encoding=encoding) as infile:
            for line in infile:
                line_parts = line.rstrip().split(separator)
                if len(line_parts) >= 2:
                    key = line_parts[term_index]
                    count = helpers.try_parse_int64(line_parts[count_index])
                    if count is not None:
                        self.create_dictionary_entry(key, count)
        return True

    def create_dictionary(self, corpus, encoding=None):
        if not os.path.exists(corpus):
            return False
        with open(corpus, "r", encoding=encoding) as infile:
            for line in infile:
                for key in self._parse_words(line):
                    self.create_dictionary_entry(key, 1)
        return True

    def save_pickle_stream(self, stream):
        pickle_data = {
            "deletes": self._deletes,
            "words": self._words,
            "max_length": self._max_length,
            "data_version": self.data_version
        }
        pickle.dump(pickle_data, stream)

    def save_pickle(self, filename, compressed=True):
        with (gzip.open if compressed else open)(filename, "wb") as f:
            self.save_pickle_stream(f)

    def load_pickle_stream(self, stream):
        pickle_data = pickle.load(stream)
        if ("data_version" not in pickle_data
                or pickle_data["data_version"] != self.data_version):
            return False
        self._deletes = pickle_data["deletes"]
        self._words = pickle_data["words"]
        self._max_length = pickle_data["max_length"]
        return True

    def load_pickle(self, filename, compressed=True):
        with (gzip.open if compressed else open)(filename, "rb") as f:
            return self.load_pickle_stream(f)

    def lookup(self, phrase, verbosity, max_edit_distance=None,
               include_unknown=False, ignore_token=None,
               transfer_casing=False):
        if max_edit_distance is None:
            max_edit_distance = self._max_dictionary_edit_distance
        if max_edit_distance > self._max_dictionary_edit_distance:
            raise ValueError("Distance too large")
        suggestions = list()
        phrase_len = len(phrase)
        if transfer_casing:
            original_phrase = phrase
            phrase = phrase.lower()

        def early_exit():
            if include_unknown and not suggestions:
                suggestions.append(SuggestItem(phrase, max_edit_distance + 1,
                                               0))
            return suggestions
        if phrase_len - max_edit_distance > self._max_length:
            return early_exit()
        suggestion_count = 0
        if phrase in self._words:
            suggestion_count = self._words[phrase]
            if transfer_casing:
                suggestions.append(SuggestItem(original_phrase, 0, suggestion_count))
            else:
                suggestions.append(SuggestItem(phrase, 0, suggestion_count))
            if verbosity != Verbosity.ALL:
                return early_exit()
        if (ignore_token is not None
                and re.match(ignore_token, phrase) is not None):
            suggestion_count = 1
            suggestions.append(SuggestItem(phrase, 0, suggestion_count))
            if verbosity != Verbosity.ALL:
                return early_exit()
        if max_edit_distance == 0:
            return early_exit()
        considered_deletes = set()
        considered_suggestions = set()
        considered_suggestions.add(phrase)
        max_edit_distance_2 = max_edit_distance
        candidate_pointer = 0
        candidates = list()
        phrase_prefix_len = phrase_len
        if phrase_prefix_len > self._prefix_length:
            phrase_prefix_len = self._prefix_length
            candidates.append(phrase[: phrase_prefix_len])
        else:
            candidates.append(phrase)
        distance_comparer = EditDistance(self._distance_algorithm)
        while candidate_pointer < len(candidates):
            candidate = candidates[candidate_pointer]
            candidate_pointer += 1
            candidate_len = len(candidate)
            len_diff = phrase_prefix_len - candidate_len
            if len_diff > max_edit_distance_2:
                if verbosity == Verbosity.ALL:
                    continue
                break

            if candidate in self._deletes:
                dict_suggestions = self._deletes[candidate]
                for suggestion in dict_suggestions:
                    if suggestion == phrase:
                        continue
                    suggestion_len = len(suggestion)
                    if (abs(suggestion_len - phrase_len) > max_edit_distance_2
                            or suggestion_len < candidate_len
                            or (suggestion_len == candidate_len
                                and suggestion != candidate)):
                        continue
                    suggestion_prefix_len = min(suggestion_len,
                                                self._prefix_length)
                    if (suggestion_prefix_len > phrase_prefix_len
                            and suggestion_prefix_len - candidate_len > max_edit_distance_2):
                        continue
                    distance = 0
                    min_distance = 0
                    if candidate_len == 0:
                        distance = max(phrase_len, suggestion_len)
                        if (distance > max_edit_distance_2
                                or suggestion in considered_suggestions):
                            continue
                    elif suggestion_len == 1:
                        distance = (phrase_len
                                    if phrase.index(suggestion[0]) < 0
                                    else phrase_len - 1)
                        if (distance > max_edit_distance_2
                                or suggestion in considered_suggestions):
                            continue
                    else:
                        if self._prefix_length - max_edit_distance == candidate_len:
                            min_distance = (min(phrase_len, suggestion_len) -
                                            self._prefix_length)
                        else:
                            min_distance = 0
                        if (self._prefix_length - max_edit_distance == candidate_len
                                and (min_distance > 1
                                     and phrase[phrase_len + 1 - min_distance :] != suggestion[suggestion_len + 1 - min_distance :])
                                or (min_distance > 0
                                    and phrase[phrase_len - min_distance] != suggestion[suggestion_len - min_distance]
                                    and (phrase[phrase_len - min_distance - 1] != suggestion[suggestion_len - min_distance]
                                         or phrase[phrase_len - min_distance] != suggestion[suggestion_len - min_distance - 1]))):
                            continue
                        else:
                            if ((verbosity != Verbosity.ALL
                                 and not self._delete_in_suggestion_prefix(candidate, candidate_len, suggestion, suggestion_len))
                                    or suggestion in considered_suggestions):
                                continue
                            considered_suggestions.add(suggestion)
                            distance = distance_comparer.compare(
                                phrase, suggestion, max_edit_distance_2)
                            if distance < 0:
                                continue
                    if distance <= max_edit_distance_2:
                        suggestion_count = self._words[suggestion]
                        si = SuggestItem(suggestion, distance,
                                         suggestion_count)
                        if suggestions:
                            if verbosity == Verbosity.CLOSEST:
                                if distance < max_edit_distance_2:
                                    suggestions = list()
                            elif verbosity == Verbosity.TOP:
                                if (distance < max_edit_distance_2
                                        or suggestion_count > suggestions[0].count):
                                    max_edit_distance_2 = distance
                                    suggestions[0] = si
                                continue
                        if verbosity != Verbosity.ALL:
                            max_edit_distance_2 = distance
                        suggestions.append(si)
            if (len_diff < max_edit_distance
                    and candidate_len <= self._prefix_length):
                if (verbosity != Verbosity.ALL
                        and len_diff >= max_edit_distance_2):
                    continue
                for i in range(candidate_len):
                    delete = candidate[: i] + candidate[i + 1 :]
                    if delete not in considered_deletes:
                        considered_deletes.add(delete)
                        candidates.append(delete)
        if len(suggestions) > 1:
            suggestions.sort()

        if transfer_casing:
            suggestions = [SuggestItem(
                helpers.transfer_casing_for_similar_text(original_phrase,
                                                         s.term),
                s.distance, s.count) for s in suggestions]

        early_exit()
        return suggestions

    def lookup_compound(self, phrase, max_edit_distance,
                        ignore_non_words=False,
                        transfer_casing=False):
        term_list_1 = helpers.parse_words(phrase)
        if ignore_non_words:
            term_list_2 = helpers.parse_words(phrase, True)
        suggestions = list()
        suggestion_parts = list()
        distance_comparer = EditDistance(self._distance_algorithm)
        is_last_combi = False
        for i, __ in enumerate(term_list_1):
            if ignore_non_words:
                if helpers.try_parse_int64(term_list_1[i]) is not None:
                    suggestion_parts.append(SuggestItem(term_list_1[i], 0, 0))
                    continue
                if helpers.is_acronym(term_list_2[i]):
                    suggestion_parts.append(SuggestItem(term_list_2[i], 0, 0))
                    continue
            suggestions = self.lookup(term_list_1[i], Verbosity.TOP,
                                      max_edit_distance)
            if i > 0 and not is_last_combi:
                suggestions_combi = self.lookup(
                    term_list_1[i - 1] + term_list_1[i], Verbosity.TOP,
                    max_edit_distance)
                if suggestions_combi:
                    best_1 = suggestion_parts[-1]
                    if suggestions:
                        best_2 = suggestions[0]
                    else:
                        best_2 = SuggestItem(term_list_1[i],
                                             max_edit_distance + 1,
                                             10 // 10 ** len(term_list_1[i]))
                    distance_1 = best_1.distance + best_2.distance
                    if (distance_1 >= 0
                            and (suggestions_combi[0].distance + 1 < distance_1
                                 or (suggestions_combi[0].distance + 1 == distance_1
                                     and (suggestions_combi[0].count > best_1.count / self.N * best_2.count)))):
                        suggestions_combi[0].distance += 1
                        suggestion_parts[-1] = suggestions_combi[0]
                        is_last_combi = True
                        continue
            is_last_combi = False
            if suggestions and (suggestions[0].distance == 0
                                or len(term_list_1[i]) == 1):
                suggestion_parts.append(suggestions[0])
            else:
                suggestion_split_best = None
                if suggestions:
                    suggestion_split_best = suggestions[0]
                if len(term_list_1[i]) > 1:
                    for j in range(1, len(term_list_1[i])):
                        part_1 = term_list_1[i][: j]
                        part_2 = term_list_1[i][j :]
                        suggestions_1 = self.lookup(part_1, Verbosity.TOP,
                                                    max_edit_distance)
                        if suggestions_1:
                            suggestions_2 = self.lookup(part_2, Verbosity.TOP,
                                                        max_edit_distance)
                            if suggestions_2:
                                tmp_term = (suggestions_1[0].term + " " +
                                            suggestions_2[0].term)
                                tmp_distance = distance_comparer.compare(
                                    term_list_1[i], tmp_term,
                                    max_edit_distance)
                                if tmp_distance < 0:
                                    tmp_distance = max_edit_distance + 1
                                if suggestion_split_best is not None:
                                    if tmp_distance > suggestion_split_best.distance:
                                        continue
                                    if tmp_distance < suggestion_split_best.distance:
                                        suggestion_split_best = None
                                if tmp_term in self._bigrams:
                                    tmp_count = self._bigrams[tmp_term]
                                    if suggestions:
                                        best_si = suggestions[0]
                                        if suggestions_1[0].term + suggestions_2[0].term == term_list_1[i]:
                                            tmp_count = max(tmp_count,
                                                            best_si.count + 2)
                                        elif (suggestions_1[0].term == best_si.term
                                              or suggestions_2[0].term == best_si.term):
                                            tmp_count = max(tmp_count,
                                                            best_si.count + 1)
                                    elif suggestions_1[0].term + suggestions_2[0].term == term_list_1[i]:
                                        tmp_count = max(
                                            tmp_count,
                                            max(suggestions_1[0].count,
                                                suggestions_2[0].count) + 2)
                                else:
                                    tmp_count = min(
                                        self.bigram_count_min,
                                        int(suggestions_1[0].count /
                                            self.N * suggestions_2[0].count))
                                suggestion_split = SuggestItem(
                                    tmp_term, tmp_distance, tmp_count)
                                if (suggestion_split_best is None or
                                        suggestion_split.count > suggestion_split_best.count):
                                    suggestion_split_best = suggestion_split

                    if suggestion_split_best is not None:
                        suggestion_parts.append(suggestion_split_best)
                        self._replaced_words[term_list_1[i]] = suggestion_split_best
                    else:
                        si = SuggestItem(term_list_1[i],
                                         max_edit_distance + 1,
                                         int(10 / 10 ** len(term_list_1[i])))
                        suggestion_parts.append(si)
                        self._replaced_words[term_list_1[i]] = si
                else:
                    si = SuggestItem(term_list_1[i], max_edit_distance + 1,
                                     int(10 / 10 ** len(term_list_1[i])))
                    suggestion_parts.append(si)
                    self._replaced_words[term_list_1[i]] = si
        joined_term = ""
        joined_count = self.N
        for si in suggestion_parts:
            joined_term += si.term + " "
            joined_count *= si.count / self.N
        joined_term = joined_term.rstrip()
        if transfer_casing:
            joined_term = helpers.transfer_casing_for_similar_text(phrase,
                                                                   joined_term)
        suggestion = SuggestItem(joined_term,
                                 distance_comparer.compare(
                                     phrase, joined_term, 2 ** 31 - 1),
                                 int(joined_count))
        suggestions_line = list()
        suggestions_line.append(suggestion)

        return suggestions_line

    def word_segmentation(self, phrase, max_edit_distance=None,
                          max_segmentation_word_length=None,
                          ignore_token=None):
        if max_edit_distance is None:
            max_edit_distance = self._max_dictionary_edit_distance
        if max_segmentation_word_length is None:
            max_segmentation_word_length = self._max_length
        array_size = min(max_segmentation_word_length, len(phrase))
        compositions = [Composition()] * array_size
        circular_index = cycle(range(array_size))
        idx = -1

        for j in range(len(phrase)):
            imax = min(len(phrase) - j, max_segmentation_word_length)
            for i in range(1, imax + 1):
                part = phrase[j : j + i]
                separator_len = 0
                top_ed = 0
                top_log_prob = 0.0
                top_result = ""

                if part[0].isspace():
                    part = part[1 :]
                else:
                    separator_len = 1
                top_ed += len(part)
                part = part.replace(" ", "")
                top_ed -= len(part)

                results = self.lookup(part, Verbosity.TOP, max_edit_distance,
                                      ignore_token=ignore_token)
                if results:
                    top_result = results[0].term
                    top_ed += results[0].distance
                    top_log_prob = math.log10(float(results[0].count) /
                                              float(self.N))
                else:
                    top_result = part
                    top_ed += len(part)
                    top_log_prob = math.log10(10.0 / self.N /
                                              math.pow(10.0, len(part)))

                dest = (i + idx) % array_size
                if j == 0:
                    compositions[dest] = Composition(part, top_result,
                                                     top_ed, top_log_prob)
                elif (i == max_segmentation_word_length
                      or ((compositions[idx].distance_sum + top_ed == compositions[dest].distance_sum
                           or compositions[idx].distance_sum + separator_len + top_ed == compositions[dest].distance_sum)
                          and compositions[dest].log_prob_sum < compositions[idx].log_prob_sum + top_log_prob)
                      or compositions[idx].distance_sum + separator_len + top_ed < compositions[dest].distance_sum):
                    compositions[dest] = Composition(
                        compositions[idx].segmented_string + " " + part,
                        compositions[idx].corrected_string + " " + top_result,
                        compositions[idx].distance_sum + separator_len + top_ed,
                        compositions[idx].log_prob_sum + top_log_prob)
            idx = next(circular_index)
        return compositions[idx]

    def _delete_in_suggestion_prefix(self, delete, delete_len, suggestion,
                                     suggestion_len):
        if delete_len == 0:
            return True
        if self._prefix_length < suggestion_len:
            suggestion_len = self._prefix_length
        j = 0
        for i in range(delete_len):
            del_char = delete[i]
            while j < suggestion_len and del_char != suggestion[j]:
                j += 1
            if j == suggestion_len:
                return False
        return True

    def _parse_words(self, text):
        matches = re.findall(r"(([^\W_]|['’])+)", text.lower())
        matches = [match[0] for match in matches]
        return matches

    def _edits(self, word, edit_distance, delete_words):
        edit_distance += 1
        word_len = len(word)
        if word_len > 1:
            for i in range(word_len):
                delete = word[: i] + word[i + 1 :]
                if delete not in delete_words:
                    delete_words.add(delete)
                    if edit_distance < self._max_dictionary_edit_distance:
                        self._edits(delete, edit_distance, delete_words)
        return delete_words

    def _edits_prefix(self, key):
        hash_set = set()
        if len(key) <= self._max_dictionary_edit_distance:
            hash_set.add("")
        if len(key) > self._prefix_length:
            key = key[: self._prefix_length]
        hash_set.add(key)
        return self._edits(key, 0, hash_set)

    @property
    def below_threshold_words(self):
        return self._below_threshold_words

    @property
    def bigrams(self):
        return self._bigrams

    @property
    def deletes(self):
        return self._deletes

    @property
    def replaced_words(self):
        return self._replaced_words

    @property
    def words(self):
        return self._words

    @property
    def word_count(self):
        return len(self._words)

class SuggestItem(object):
    def __init__(self, term, distance, count):
        self._term = term
        self._distance = distance
        self._count = count

    def __eq__(self, other):
        if self._distance == other.distance:
            return self._count == other.count
        else:
            return self._distance == other.distance

    def __lt__(self, other):
        if self._distance == other.distance:
            return self._count > other.count
        else:
            return self._distance < other.distance

    def __str__(self):
        return "{}, {}, {}".format(self._term, self._distance, self._count)

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, term):
        self._term = term

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, distance):
        self._distance = distance

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, count):
        self._count = count

Composition = namedtuple("Composition",
                         ["segmented_string", "corrected_string",
                          "distance_sum", "log_prob_sum"])
Composition.__new__.__defaults__ = (None,) * len(Composition._fields)
Composition.__doc__ = """Used by :meth:`word_segmentation`

**NOTE**: "Parameters" is used instead "Attributes" due to a bug which
overwrites attribute descriptions.

Parameters
----------
segmented_string : str
    The word segmented string.
corrected_string : str
    The spelling corrected string.
distance_sum : int
    The sum of edit distance between input string and corrected string
log_prob_sum : float
    The sum of word occurrence probabilities in log scale (a measure of
    how common and probable the corrected segmentation is).
"""

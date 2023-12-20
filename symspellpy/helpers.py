from difflib import SequenceMatcher
from itertools import zip_longest
import re

def null_distance_results(string1, string2, max_distance):
    if string1 is None:
        if string2 is None:
            return 0
        else:
            return len(string2) if len(string2) <= max_distance else -1
    return len(string1) if len(string1) <= max_distance else -1

def prefix_suffix_prep(string1, string2):
    len1 = len(string1)
    len2 = len(string2)
    while len1 != 0 and string1[len1 - 1] == string2[len2 - 1]:
        len1 -= 1
        len2 -= 1
    start = 0
    while start != len1 and string1[start] == string2[start]:
        start += 1
    if start != 0:
        len1 -= start
        len2 -= start
    return len1, len2, start

def to_similarity(distance, length):
  
    return -1 if distance < 0 else 1.0 - distance / length

def try_parse_int64(string):
    try:
        ret = int(string)
    except ValueError:
        return None
    return None if ret < -2 ** 64 or ret >= 2 ** 64 else ret

def parse_words(phrase, preserve_case=False):
    if preserve_case:
        return re.findall(r"([^\W_]+['’]*[^\W_]*)", phrase)
    else:
        return re.findall(r"([^\W_]+['’]*[^\W_]*)", phrase.lower())

def is_acronym(word):
    return re.match(r"\b[A-Z0-9]{2,}\b", word) is not None

def transfer_casing_for_matching_text(text_w_casing, text_wo_casing):
    if len(text_w_casing) != len(text_wo_casing):
        raise ValueError("The 'text_w_casing' and 'text_wo_casing' "
                         "don't have the same length, "
                         "so you can't use them with this method, "
                         "you should be using the more general "
                         "transfer_casing_similar_text() method.")

    return ''.join([y.upper() if x.isupper() else y.lower()
                    for x, y in zip(text_w_casing, text_wo_casing)])

def transfer_casing_for_similar_text(text_w_casing, text_wo_casing):
    if not text_wo_casing:
        return text_wo_casing
    if not text_w_casing:
        raise ValueError("We need 'text_w_casing' to know what "
                         "casing to transfer!")
    _sm = SequenceMatcher(None, text_w_casing.lower(),
                          text_wo_casing)
    c = ''
    for tag, i1, i2, j1, j2 in _sm.get_opcodes():
        if tag == 'insert':
            if i1 == 0 or text_w_casing[i1 - 1] == ' ':
                if text_w_casing[i1] and text_w_casing[i1].isupper():
                    c += text_wo_casing[j1:j2].upper()
                else:
                    c += text_wo_casing[j1:j2].lower()
            else:
                if text_w_casing[i1 - 1].isupper():
                    c += text_wo_casing[j1:j2].upper()
                else:
                    c += text_wo_casing[j1:j2].lower()
        elif tag == 'delete':
            pass
        elif tag == 'equal':
            c += text_w_casing[i1:i2]
        elif tag == 'replace':
            _w_casing = text_w_casing[i1:i2]
            _wo_casing = text_wo_casing[j1:j2]
            if len(_w_casing) == len(_wo_casing):
                c += transfer_casing_for_matching_text(
                    text_w_casing=_w_casing, text_wo_casing=_wo_casing)
            else:
                _last = 'lower'
                for w, wo in zip_longest(_w_casing, _wo_casing):
                    if w and wo:
                        if w.isupper():
                            c += wo.upper()
                            _last = 'upper'
                        else:
                            c += wo.lower()
                            _last = 'lower'
                    elif not w and wo:
                        c += wo.upper() if _last == 'upper' else wo.lower()
    return c
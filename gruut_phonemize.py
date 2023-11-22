from gruut import sentences


def gphonemize(text):
    phonemes = ''
    for sent in sentences(text, lang="en-us"):
        for word in sent:
            if word.phonemes:
                phonemes += ''.join(word.phonemes)
    return phonemes
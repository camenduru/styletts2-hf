from gruut import sentences


text = input("> ")
phonemes = ''
for sent in sentences(text, lang="en-us"):
    for word in sent:
        if word.phonemes:
            print(word.text + ":" + ''.join(word.phonemes))
            phonemes += ''.join(word.phonemes)
        else:
            print(word.text + ": NoPhonemes")
print("--")
print(phonemes)

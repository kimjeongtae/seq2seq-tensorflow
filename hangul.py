NUM_CHOSUNGS = 19
NUM_JUNGSUNGS = 21
NUM_JONGSUNGS = 28

FIRST_HANGUL_UNICODE = ord('가')
LAST_HANGUL_UNICODE = ord('힣')


CHOSUNGS = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ',
            'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

JUNGSUNGS = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
             'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

# ' ': ' ', 'ㄱ': '!', 'ㄴ': '@', 'ㄹ': '#', 'ㅁ': '$', 'ㅂ': '%', 'ㅅ': '^', 'ㅇ': '&'
JONGSUNGS = [' ', '!', 'ㄲ', 'ㄳ', '@', 'ㄵ', 'ㄶ', 'ㄷ', '#', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ',
             'ㄿ', 'ㅀ', '$', '%', 'ㅄ', '^', 'ㅆ', '&', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

'''
JONGSUNGS = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ',
             'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
'''


def jaso_to_syllable(chosung, jungsung, jongsung=' '):
    chosung_index = CHOSUNGS.index(chosung)
    jungsung_index = JUNGSUNGS.index(jungsung)
    jongsung_index = JONGSUNGS.index(jongsung)

    syllable_code = (((chosung_index * NUM_JUNGSUNGS) + jungsung_index) * NUM_JONGSUNGS) + \
                    jongsung_index + FIRST_HANGUL_UNICODE

    return chr(syllable_code)


def syllable_to_jaso(syllable):
    code = ord(syllable) - FIRST_HANGUL_UNICODE
    jongsung_index = code % NUM_JONGSUNGS
    code //= NUM_JONGSUNGS
    jungsung_index = code % NUM_JUNGSUNGS
    code //= NUM_JUNGSUNGS
    chosung_index = code

    return CHOSUNGS[chosung_index] + JUNGSUNGS[jungsung_index] + JONGSUNGS[jongsung_index]


def word_to_jaso(word):
    return ''.join(syllable_to_jaso(syllable) for syllable in word)


def jaso_to_word(jaso_list):
    return ''.join(jaso_to_syllable(*syllable) for syllable in zip(jaso_list[::3], jaso_list[1::3], jaso_list[2::3]))


def is_hangul(syllables):
    for syllable in syllables:
        if not FIRST_HANGUL_UNICODE <= ord(syllable) <= LAST_HANGUL_UNICODE:
            return False
    return True


if __name__ == '__main__':
    print(jaso_to_syllable(*'ㄱㅣ '))
    print(syllable_to_jaso(*'김'))
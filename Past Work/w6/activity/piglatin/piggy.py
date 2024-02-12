def is_consonant1(letter):

    if letter.lower() not in 'aeiou':

        return True

    else:

        return False


def to_piglatin1(word):

    c=''

    # move all starting consonant to the end of the word

    for i in word.lower():

        if is_consonant(i)==False:

            break

        else:

            c+=i

    output = word.lower().replace(c,'')+c+'ay'

    return output[0].upper()+output[1::]

def is_consonant2(letter):

    ''' function is_consonant that takes a character and returns True if it is a consonant.'''

    return (False if letter in 'aeiou' else True)

 

def to_piglatin2(word):

    ''' function to_piglatin that takes a word, moves all starting consonants (all consonants before the first 

    vowel) to the end of the word, then adds ay to the end and returns the result. '''

    cons =[]

    pig_latin=[]

    word_list = list(word.lower())

        

    while (len(word_list) > 0):

        letter = word_list.pop(0)

 

        if is_consonant(letter):

            cons.append(letter)

        else:

            pig_latin.insert(0,letter)

            pig_latin.append(''.join(word_list))

            break

 

    cons.append('ay')

    pig_latin.append(''.join(cons))

    return (''.join(pig_latin).capitalize())
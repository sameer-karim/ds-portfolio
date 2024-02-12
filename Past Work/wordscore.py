def score_word(word, wild_positions):
    """
    Compute the score for the word based on the provided scoring dictionary.

    Parameters:
    - word: str, the word to compute the score for.

    Returns:
    int, the score of the word.
    """
    scores = {"a": 1, "c": 3, "b": 3, "e": 1, "d": 2, "g": 2,
              "f": 4, "i": 1, "h": 4, "k": 5, "j": 8, "m": 3,
              "l": 1, "o": 1, "n": 1, "q": 10, "p": 3, "s": 1,
              "r": 1, "u": 1, "t": 1, "w": 4, "v": 4, "y": 4,
              "x": 8, "z": 10}

    return sum([0 if idx in wild_positions else scores[char] for idx, char in enumerate(word.lower())])
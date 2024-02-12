import itertools
from wordscore import score_word

def run_scrabble(rack):
    """
    Compute the valid Scrabble words that can be constructed from the given rack.

    Parameters:
    - rack: str, the tiles given by the user.

    Returns:
    tuple, a list of valid Scrabble words and their scores, and the total number of valid words.
    """

    # Validate input
    if not 2 <= len(rack) <= 7:
        return "Invalid input length. Rack should have between 2 and 7 tiles.",

    # Check for more than two wildcards
    if rack.count("?") + rack.count("*") > 2:
        return "Invalid input. Only up to two wildcards ('?' and '*') are allowed.",

    # Read valid Scrabble words
    with open("sowpods.txt", "r") as infile:
        raw_input = infile.readlines()
        valid_words = [datum.strip('\n').upper() for datum in raw_input]

    # Generate all possible words
    possible_words = set()

    for i in range(2, len(rack) + 1):
        for combo in itertools.permutations(rack, i):
            # Handle wildcards
            word_combos = [(combo, [])]  # Each word along with its wildcards' positions
            if "?" in combo:
                word_combos = [(list(combo), [idx for idx, char in enumerate(combo) if char == '?'])]
                for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    new_combo = [char if c == '?' else c for c in combo]
                    word_combos.append((new_combo, [idx for idx, char in enumerate(combo) if char == '?']))
            if "*" in combo:
                new_word_combos = []
                for word, wild_positions in word_combos:
                    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        new_wild_positions = [idx for idx, char in enumerate(combo) if char == '*'] + wild_positions
                        new_word_combos.append(([char if c == '*' else c for c in word], new_wild_positions))
                word_combos = new_word_combos
            for word, wild_positions in word_combos:
                if "".join(word).upper() in valid_words:
                    possible_words.add(("".join(word).upper(), tuple(wild_positions)))

    # Score the possible words
    scored_words = []
    for word, wild_positions in possible_words:
        score = score_word(word, wild_positions)  # Now we also send the wildcards' positions
        scored_words.append((score, word))
    scored_words.sort(key=lambda x: (-x[0], x[1]))

    return scored_words, len(scored_words)

run_scrabble('has*')
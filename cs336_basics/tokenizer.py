from collections import Counter
from collections.abc import Sequence

# count adjacent pairs in one token sequence -> Counter[pair, count]
def pair_counter(tokens: Sequence[bytes]) -> Counter[tuple[bytes, bytes]]:
  
  counts: Counter[tuple[bytes, bytes]] = Counter()
  for i in range(0, len(tokens) - 1):
      pair = (tokens[i], tokens[i+1])
      counts[pair] += 1
  return counts


# count pairs across the whole corpus,
# enable training to choose the most frequent pair
def count_corpus_pairs(sequences: list[list[bytes]]) -> Counter[tuple[bytes, bytes]]:
  
  counts: Counter[tuple[bytes, bytes]] = Counter()
  for sequence in sequences:
    counts += pair_counter(sequence)
  return counts

# merge one chosen pair in one sequence.
def merge_pair(tokens: Sequence[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:

  merge_list: list[bytes] = []
  i = 0
  while i < len(tokens):
    if i+1 >= len(tokens):
      merge_list.append(tokens[i])
      i += 1
    elif (tokens[i], tokens[i+1]) == pair:
      merge_list.append(tokens[i] + tokens[i+1])
      i += 2
    else:
      merge_list.append(tokens[i])
      i += 1
  return merge_list

# apply one chosen merge pair across all token sequences in the training corpus
def apply_merge(
  
  sequences: list[list[bytes]], pair: tuple[bytes, bytes]) -> list[list[bytes]]:
  merged_sequences: list[list[bytes]] = []
  for sequence in sequences:
    merged_sequences.append(merge_pair(sequence, pair))
  return merged_sequences
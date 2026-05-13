from collections import Counter
from collections.abc import Sequence
import os
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def run_train_bpe(
  input_path: str | os.PathLike, 
  vocab_size: int, 
  special_tokens: list[str],
  **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  
  with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

  corpus = text_to_byte_sequences(text, special_tokens)

  vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

  for special_token in special_tokens:
    token_bytes = special_token.encode("utf-8")
    if token_bytes not in set(vocab.values()):
      vocab[len(vocab)] = token_bytes
  
  merges: list[tuple[bytes, bytes]] = []
  pair_counts, pair_to_sequences = build_pair_index(corpus)
  while len(vocab) < vocab_size:
      if not pair_counts:
          break
      
      best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
      merges.append(best_pair)
      corpus = apply_merge(corpus, best_pair)
      vocab[len(vocab)] = best_pair[0] + best_pair[1]

  return vocab, merges

# take raw text and split it into meaningful chunks before BPE sees any bytes
def pretokenize(text: str) -> list[str]:
  return [match.group(0) for match in re.finditer(PAT, text)]

def pretoken_to_byte_tokens(pretoken: str) -> list[bytes]:
  return to_byte_tokens(pretoken)

# convert each pretoken string into a list of single-byte bytes objects
def to_byte_tokens(text: str) -> list[bytes]:
  return [bytes([b]) for b in text.encode("utf-8")]

def text_to_byte_sequences(
  text: str,
  special_tokens: list[str] | None = None,
) -> dict[tuple[bytes, ...], int]:
  counts: dict[tuple[bytes, ...], int] = {}

  parts = [text]
  if special_tokens:
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    special_pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
    parts = re.split(special_pattern, text)

  for part in parts:
    if part == "":
      continue
    if special_tokens and part in special_tokens:
      continue
    
    for pretoken in pretokenize(part):
      sequence = tuple(pretoken_to_byte_tokens(pretoken))
      counts[sequence] = counts.get(sequence, 0) + 1

  return counts

# count adjacent pairs in one token sequence -> Counter[pair, count]
def pair_counter(tokens: Sequence[bytes]) -> Counter[tuple[bytes, bytes]]:
  counts: Counter[tuple[bytes, bytes]] = Counter()
  for i in range(0, len(tokens) - 1):
      pair = (tokens[i], tokens[i+1])
      counts[pair] += 1
  return counts

def sequence_pairs(sequence: tuple[bytes, ...]) -> set[tuple[bytes, bytes]]:
  pairs: set[tuple[bytes, bytes]] = set()
  for i in range(len(sequence) - 1):
    pairs.add((sequence[i], sequence[i + 1]))
  return pairs

def sequence_pair_counts(
  sequence: tuple[bytes, ...],
  frequency: int,
) -> Counter[tuple[bytes, bytes]]:
  counts: Counter[tuple[bytes, bytes]] = Counter()
  for i in range(len(sequence) - 1):
    counts[(sequence[i], sequence[i + 1])] += frequency
  return counts

def build_pair_index(
  corpus: dict[tuple[bytes, ...], int],
) -> tuple[
  Counter[tuple[bytes, bytes]],
  dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
]:
  pair_counts: Counter[tuple[bytes, bytes]] = Counter()
  pair_to_sequences: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

  for sequence, frequency in corpus.items():
    for pair, count in sequence_pair_counts(sequence, frequency).items():
      pair_counts[pair] += count
      pair_to_sequences.setdefault(pair, set()).add(sequence)

  return pair_counts, pair_to_sequences

# # count pairs across the whole corpus,
# # enable training to choose the most frequent pair
# def count_corpus_pairs(
#   sequences: dict[tuple[bytes, ...], int]
# ) -> Counter[tuple[bytes, bytes]]:
#   counts: Counter[tuple[bytes, bytes]] = Counter()

#   for sequence, frequency in sequences.items():
#     for i in range(len(sequence) - 1):
#       pair = (sequence[i], sequence[i + 1])
#       counts[pair] += frequency

#   return counts

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

def sequence_has_pair(
  sequence: Sequence[bytes],
  pair: tuple[bytes, bytes],
) -> bool:
  for i in range(len(sequence) - 1):
    if (sequence[i], sequence[i + 1]) == pair:
      return True
  return False

# apply one chosen merge pair across all token sequences in the training corpus
def apply_merge(
  sequences: dict[tuple[bytes, ...], int], 
  pair: tuple[bytes, bytes]
) -> dict[tuple[bytes, ...], int]:
  merged_sequences: dict[tuple[bytes, ...], int] = {}
  
  for sequence, frequency in sequences.items():
    if sequence_has_pair(sequence, pair):
      merged_sequence = tuple(merge_pair(sequence, pair))
    else: 
      merged_sequence = sequence

    merged_sequences[merged_sequence] = merged_sequences.get(merged_sequence, 0) + frequency
  
  return merged_sequences

class BPETokenizer:
  def __init__(
    self,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
  ):
    self.vocab = vocab
    self.merges = merges
    self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
    self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
    self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
    self.special_token_to_id = {
      token: self.token_to_id[token.encode("utf-8")]
      for token in self.special_tokens
    }

  def encode_pretoken(self, pretoken: str) -> list[int]:
    tokens = to_byte_tokens(pretoken)

    while True:
      pairs = pair_counter(tokens)
      ranked_pairs = [
        (self.merge_ranks[pair], pair)
        for pair in pairs
        if pair in self.merge_ranks
      ]
      if not ranked_pairs:
        break
      
      _, best_pair = min(ranked_pairs)
      tokens = merge_pair(tokens, best_pair)
    return [self.token_to_id[token] for token in tokens]

  def encode(self, text: str) -> list[int]:
    if not self.special_tokens:
      ids: list[int] = []
      for pretoken in pretokenize(text):
        ids.extend(self.encode_pretoken(pretoken))
      return ids
    
    ids: list[int] = []
    special_pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
    parts = re.split(special_pattern, text)
    for part in parts: 
      if part == "":
        continue
      if part in self.special_token_to_id:
        ids.append(self.special_token_to_id[part])
      else:
        for pretoken in pretokenize(part):
          ids.extend(self.encode_pretoken(pretoken))
    return ids

  def encode_iterable(self, iterable):
    for text in iterable:
        yield from self.encode(text)

  def decode(self, ids: list[int]) -> str:
    token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
    return token_bytes.decode("utf-8", errors="replace")
  
def get_tokenizer(
  vocab: dict[int, bytes],
  merges: list[tuple[bytes, bytes]],
  special_tokens: list[str] | None = None,
) -> BPETokenizer:
  return BPETokenizer(vocab, merges, special_tokens)

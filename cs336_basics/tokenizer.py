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

  corpus = text_to_byte_sequences(text)

  vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

  for special_token in special_tokens:
    token_bytes = special_token.encode("utf-8")
    if token_bytes not in set(vocab.values()):
      vocab[len(vocab)] = token_bytes
  
  merges = []
  while len(vocab) < vocab_size:
      pair_counts = count_corpus_pairs(corpus)
      if not pair_counts:
          break
      
      best_pair, _ = pair_counts.most_common(1)[0]
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

def text_to_byte_sequences(text: str) -> list[list[bytes]]:
  return [pretoken_to_byte_tokens(tok) for tok in pretokenize(text)]

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
    self.special_tokens = special_tokens or []
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

  def decode(self, ids: list[int]) -> str:
    token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
    return token_bytes.decode("utf-8", errors="replace")
  
def get_tokenizer(
  vocab: dict[int, bytes],
  merges: list[tuple[bytes, bytes]],
  special_tokens: list[str] | None = None,
) -> BPETokenizer:
  return BPETokenizer(vocab, merges, special_tokens)


    





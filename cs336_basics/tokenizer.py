from collections import Counter
from collections.abc import Sequence

def pair_counts(
  tokens: Sequence[bytes]
  ) -> Counter[tuple[bytes, bytes]]:
  counts: Counter[tuple[bytes, bytes]] = Counter()
  for i in range(0, len(tokens) - 1):
      pair = (tokens[i], tokens[i+1])
      counts[pair] += 1
  return counts

def merge_pair(tokens: Sequence[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:

  while 
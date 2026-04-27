import json
import random
import sys
sys.path.insert(0, ".")

from src.tokenizer import ABCTokenizer, tokenize_abc

with open("data/tunes.json") as f:
    tunes = json.load(f)

tokenizer = ABCTokenizer.from_tunes(tunes)
print(f"Vocab size: {len(tokenizer):,}\n")

for tune in random.sample(tunes, 3):
    print(f"Name:   {tune['name']}")
    print(f"Type:   {tune['type']}  Key: {tune['mode']}  Meter: {tune['meter']}")
    print(f"ABC:    {tune['abc'][:80]}...")

    tokens = tokenize_abc(tune["abc"])
    print(f"Tokens ({len(tokens)}): {tokens[:20]} ...")

    ids = tokenizer.encode(tune["abc"], tune["type"], tune["mode"], tune["meter"])
    print(f"IDs    ({len(ids)}): {ids[:20]} ...")
    print(f"Prefix: {tokenizer.decode_to_tokens(ids)[:5]}")

    reconstructed = tokenizer.decode_to_abc(ids)
    print(f"Decode: {reconstructed[:80]}...")
    print()

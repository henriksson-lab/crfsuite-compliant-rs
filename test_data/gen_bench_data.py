#!/usr/bin/env python3
"""Generate a large synthetic dataset in IWA format for benchmarking CRFsuite."""
import random
import sys

random.seed(42)

LABELS = ["B-NP", "I-NP", "B-VP", "I-VP", "B-PP", "B-ADJP", "I-ADJP", "B-ADVP", "O"]
WORDS = [
    "the", "a", "an", "cat", "dog", "bird", "fish", "man", "woman", "child",
    "sat", "ran", "flew", "swam", "walked", "jumped", "kicked", "threw",
    "on", "in", "at", "over", "under", "near", "by", "with", "from", "to",
    "big", "small", "red", "blue", "green", "fast", "slow", "old", "new",
    "quickly", "slowly", "very", "quite", "often", "always", "never",
    "house", "park", "tree", "car", "road", "river", "mountain", "city",
    "happy", "sad", "tall", "short", "bright", "dark", "warm", "cold",
]
POS_TAGS = ["NN", "NNS", "VB", "VBD", "VBG", "IN", "DT", "JJ", "RB", "CC"]

def gen_sequence(min_len=3, max_len=20):
    length = random.randint(min_len, max_len)
    items = []
    for t in range(length):
        label = random.choice(LABELS)
        w_cur = random.choice(WORDS)
        pos = random.choice(POS_TAGS)
        attrs = [f"w[0]={w_cur}", f"pos[0]={pos}"]
        if t > 0:
            w_prev = random.choice(WORDS)
            attrs.append(f"w[-1]={w_prev}")
        if t < length - 1:
            w_next = random.choice(WORDS)
            attrs.append(f"w[1]={w_next}")
        # Add some bigram features
        attrs.append(f"w[0]|pos[0]={w_cur}|{pos}")
        if random.random() < 0.3:
            attrs.append(f"prefix2={w_cur[:2]}")
        if random.random() < 0.3:
            attrs.append(f"suffix2={w_cur[-2:]}")
        items.append(f"{label}\t" + "\t".join(attrs))
    return "\n".join(items)

def main():
    n_sequences = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    outfile = sys.argv[2] if len(sys.argv) > 2 else "-"

    if outfile == "-":
        f = sys.stdout
    else:
        f = open(outfile, "w")

    for i in range(n_sequences):
        f.write(gen_sequence() + "\n\n")

    if outfile != "-":
        f.close()
        total_items = sum(1 for line in open(outfile) if line.strip() and not line.startswith("\n"))
        print(f"Generated {n_sequences} sequences, ~{total_items} items", file=sys.stderr)

if __name__ == "__main__":
    main()

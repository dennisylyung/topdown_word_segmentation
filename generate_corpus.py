import os
import time
from random import shuffle

dir = 'threads'
files = os.listdir(dir)
print(f'{len(files)} threads')
s = time.time()
threads = []
unique_replies = set()
for i, file in enumerate(files):
    if i % (len(files) // 20) == 0:
        print(f'reading {i}th thread ({i * 100 / len(files):.2f}%) in {time.time() - s:.2f}s')
    try:
        int(file)
    except ValueError:
        continue
    with open(os.path.join(dir, file), 'r') as f:
        thread = f.read()
    replies = set([sentence for sentence in thread.replace('。', '\n').split('\n') if len(sentence) > 10])
    dedup_replies = replies - unique_replies
    unique_replies.update(dedup_replies)
    threads.append("\n".join(dedup_replies))
    with open(f'dedup_threads/{i}', 'w') as f:
        f.write("\n".join(dedup_replies))

print(f'{len(threads)} threads')
sentences = sum([thread.count("\n") + 1 for thread in threads])
print(f'{sentences} sentences')
print(f'{sum([len(thread) for thread in threads])} characters')
print(f'{len(set("".join(threads)))} unique characters')

shuffle(threads)
print(f'threads shuffled')

with open(f'corpus_full.txt', 'w') as f:
    f.write("\n".join(threads))
print(f'full corpus written to corpus_full.txt')

lines = [line for thread in threads for line in thread.split('\n')]
sample_size = 2048

with open(f'corpus_train.txt', 'w') as f:
    f.write("\n".join(lines[sample_size:]))
print(f'training corpus of {len(lines[sample_size:])} lines written to corpus_train.txt')

with open(f'corpus_eval.txt', 'w') as f:
    f.write("\n".join(lines[:sample_size]))
print(f'training corpus of {len(lines[:sample_size])} lines written to corpus_train.txt')

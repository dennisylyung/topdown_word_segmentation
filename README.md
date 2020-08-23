# Top Down Word Segmentation

This project tries to perform word segmentation on Cantonese text with semantic awareness from a language model.

## Background

To train a language model such as BERT, text must first be tokenized. 
This is relatively straightforward with English and other alphabet based languages, as words are clearly delimited with spaces. 

>  "Flowers grow in the field behind the house"
>
> Flowers / grow / in / the / field / behind / the / house

However in languages such as Chinese and Japanese, words are made up of characters, and are not clearly separated.

>  "花生長在屋後的田裏"
>
> **花** / 生長 / 在 / 屋 / 後 / 的 / 田 / 裏
>
> (**Flowers** grow in the field behind the house)
>
> **花生** / 長 / 在 / 屋 / 後 / 的 / 田 / 裏
>
> (**Peanuts** grow in the field behind the house)

Example from [this paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4214861/)

Frequently, word boundaries can be ambiguous, such as the example above, suggesting very different meanings. 

### Analogy for English speakers

Each chinese character had a single syllable, 
thus the ambiguity of Chinese word segmentation has similarities to grouping syllables from spoken English into words.

Consider this example:

> Anna Mary Candy Lights Since Imp Pulp Lay Things

If you read this out, you may find that these syllables represent an actual legit sentence. 

(The example is taken from my [psychology textbook](https://scholar.google.com.hk/scholar?cluster=13207650529714687855&hl=en&oi=scholarr) about perception)

### Common approaches

A common Chinese text tokenizer is the pretrained [jieba](https://github.com/fxsjy/jieba)
Newer models such as Baidu's [LAC](https://github.com/baidu/lac) address ambiguous word boundaries by training on large labelled corpus
Some universal tokenization model such as [sentencepiece](https://github.com/google/sentencepiece) also have features to accomodate non-space languages

### Objective of the project

While there are many Chinese tokenizers to choose from, the language I speak, Cantonese, is different from Chinese. 

The language is mainly spoken in Hong Kong, and southeastern China.

While Cantonese and Mandarin Chinese are very similar structurally, the vocabulary is quite different.

Therefore, Chinese tokenizers like jieba does not work very well on Cantonese text. 

Developing a tokenizer can be very costly. For example Baidu uses a lot of manually segmented sentences to train its LAC model. 

Acquiring such data for a smaller language like Cantonese is much harder. 

I wonder if we can achieve similar results with semi-supervised learning, training solely on the large corpus from social platforms. 

The idea also stems from the idea of "top-down processing" that I learned in psychology. 

Psychology studies found that humans do not process language "bottom-up", step after step. 

i.e. Word segmentation -> Word meaning -> sentence meaning

In contrast, we perform earlier steps with the help from later steps. 

Therefore, the reason why we can understand ambiguous sentences effortlessly is the word boundaries are quite apparent if we consider the whole sentence. 

### Scope

Although the project objective sounds quite academic, I do this project just for fun. 

## The Project

My theory is that when we read a ambiguous sentence, the conflicting "candidates" for how to segment it is activated in our brain.
Afterwards our brain pick the candidate that is most coherence. 

I first pretrained a language model (Albert) using a corpus collected online.
Then, a sentence coherence model is trained with the pretrained LM with a ranking loss.
Afterwards, I try ranking segmentation candidates generated from a unigram model with the coherence model.

### 1. Corpus acquitision

I have wrote a bot to collect threads from a popular discussion forum in Hong Kong. 

After removing short and duplicate messages, the corpus is around 1GB, with more than 300B characters

However, the language model was trained halfway, with a smaller corpus. 

the corpus and the bot is not included in git due to copyright.

### 2. Tokenization

The corpus tokenized with sentencepiece with the unigram model.

As I want to rank the candidate tokens ultimately, I want to use a larger vocabulary for this step.

So I have chosen a size of 65536

The tokenizer is trained in [`train_tokenizer.py`](train_tokenizer.py)

### 2. Language Model

I trained a language model using Albert, whose author reported a good performance on a Chinese corpus using few parameters.

However, due to the large vocab size, the model still uses 9M parameters. 

A modification for this project is enabling tokenization sampling during training. 

The unigram tokenizer class in Huggingface do not have sampling enabled. 

A custom class was written to override this behaviour.

The LM model is trained in [`lm_training.ipynb`](/lm_training.ipynb)

### 3. Coherence model

My believe is that suboptimal segmentation of ambiguous sentences will contain tokens that are less coherent with the whole sentence.

For the semi-supervised learning, I generated negative samples by replacing tokens in the sentences.

Since the "correct" samples will also contain suboptimal segmentations, it is hard to classify whether the sentence a "right" or "wrong"

Therefore, I have chosen to train on a ranking loss, where the tokenized sentences from the corpus are assumed to be at least more coherent than those with tokens replaced.

It is hoped that the relationship learned by this negative sampling could also be generalized to rank segmentation candidates from the same sentence

The coherence model is trained in [`coherence_training.ipynb`](coherence_training.ipynb)

### 4. Word segmentation

Finally, word segmentation is performed by:

1. generate the n best tokenization candidates using the unigram model

2. calculate the coherence score on each of the candidates using the coherence model

3. rank the candidates according to the score

The code can be found in [`word_segmentation.ipynb`](/word_segmentation.ipynb)

## Results

Results can be found [here](/word_segmentation.ipynb).

Results using 2 ambiguous sentences well known for their indecent alternatives:

```python
nws.get('兒子生性病母倍感安慰', 10)

>>  [('▁/兒子/生/性病/母/倍/感/安慰', -0.52289635, 1),
     ('▁/兒子/生性/病/母/倍/感/安慰', -0.62185246, 2),
     ('▁/兒/子/生/性/病/母/倍/感/安慰', -0.67673385, 8),
     ('▁/兒/子/生/性病/母/倍/感/安慰', -0.6780272, 4),
     ('▁/兒/子/生性/病/母/倍/感/安慰', -0.75988257, 5)]
```

The top result, which is the funny but incorrect one:

> son / contract / STD / mother / very / feel / relief

> Mother feels relieved that her son contracted STD

The second result, which has the correct meaning (of the real news headline)

> son / mature / sick / mother / very / feel / relief

> Sick mother feels relieved as her son matures

However the unigram tokenizer also suffers from this problem, as seen in the original rank shown in the third value in the tuples.

---

```python
nws.get('獅子山下體現香港精神', 10)

>>  [('▁/獅子山/下體/現/香港/精神', 0.110433176, 5),
     ('▁/獅子山/下/體/現/香港/精神', 0.016519196, 8),
     ('▁/獅子/山/下/體現/香港/精神', -0.036805652, 7),
     ('▁/獅子山下/體/現/香港/精神', -0.14911321, 3),
     ('▁/獅子山下/體現/香港/精神', -0.15852737, 1),
     ('▁/獅子山/下/體現/香港/精神', -0.20499955, 2),
     ('▁/獅子山下/體/現/香港/精/神', -0.20978086, 10),
     ('▁/獅子山/下/體現/香港/精/神', -0.22237615, 9),
     ('▁/獅子山下/體現/香港/精/神', -0.3028409, 4),
     ('▁/獅子山下/體現/香/港/精神', -0.37603474, 6)]
```

The top result, which is incorrect one:

> Lion Rock (a hill in HK) / genitals / show / Hong Kong / spirit

> The genitals of the Lion Rock show the spirit of Hong Kong

The fifth result, which is ranked first by the unigram tokenizer, has the correct meaning

> Bottom of Lion Rock / demonstrate / Hong Kong / spirit

> Demonstrating the spirit of Hong Kong at the bottom of Lion Rock


One possible reason is that the model, being trained on the threads from a passively moderated discussion board, 
do not find random occurrences of sexual references incoherent......

---

Now, let's try with examples from the academic paper

```python
nws.get('花生長在屋後的田裡', 10)

>>  [('▁/花/生長/在/屋/後的/田/裡', 0.08546053, 3),
     ('▁/花/生長/在/屋/後/的/田/裡', -0.15445843, 4),
     ('▁/花/生/長/在/屋/後/的/田/裡', -0.22414203, 6),
     ('▁/花/生/長/在/屋/後的/田/裡', -0.47418493, 5),
     ('▁/花生/長/在/屋/後的/田/裡', -0.81693584, 1),
     ('▁/花生/長/在/屋/後/的/田/裡', -0.92735624, 2)]
```

The top result, which is correct one:

> Flowers / grow / at / house / behind / field / inside

> Flowers grow in the field behind the house

The fifth result, which is ranked first by the unigram tokenizer, is incorrect

> Peanuts / grow / at / house / behind / field / inside

> Peanuts grow in the field behind the house


---

```python
nws.get('照顧客嘅要求設計產品', 10)

>>  [('▁/照/顧客/嘅/要求/設計/產品', 0.0071468055, 3),
     ('▁/照/顧/客/嘅/要求/設計/產品', -0.06025382, 9),
     ('▁/照/顧客/嘅/要/求/設計/產品', -0.220116, 6),
     ('▁照/顧/客/嘅/要求/設計/產品', -0.22720684, 7),
     ('▁照/顧客/嘅/要/求/設計/產品', -0.8018307, 5),
     ('▁照/顧客/嘅/要求/設計/產品', -0.835267, 2),
     ('▁/照顧/客/嘅/要求/設計/產品', -0.8555057, 1),
     ('▁/照顧/客/嘅/要/求/設計/產品', -0.95422095, 4),
     ('▁/照顧/客/嘅/要求/設計/產/品', -0.9880646, 10),
     ('▁/照顧/客/嘅/要求/設/計/產品', -1.0709113, 8)]
```

The top result, which is correct one:

> According to / client / 's / request / design / product

> Design the product according to the client's request

The seventh result, which is ranked first by the unigram tokenizer, is incorrect

> Take care of / guest / 's / request / design / product

> Design products taking care of the guest's product

While the sentence seem to have similar meaning, it is not really coherent. 

The word "照顧" strictly mean taking care of the well being. 

### Summary

I'd say that the results are not conclusive yet. 
It sometimes performs better, sometimes worse. This could be due to chance. 

I am going to fine tune the model, and evaluate more systematically. 

For example, I might try evaluating on a small manually segmented corpus
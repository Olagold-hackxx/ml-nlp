# ML NLP

> Sequence models and language models, from word2vec to retrieval-augmented agents.

This repository is Phase 3 of my Applied ML track. It follows Stanford **CS224N** (NLP with Deep Learning), **CS224U** (Natural Language Understanding) and **CME 295** (Transformers and Large Language Models).

It builds the modern NLP stack in the order it was invented: word vectors, parsers, recurrent language models, attention, the Transformer written from scratch, pretraining and fine-tuning, the internals of current LLMs (RoPE, the KV cache, efficient attention), and finally retrieval and tool-using agents. Each system is evaluated with the metric its field uses: perplexity, UAS, BLEU, exact match, recall@k or task success.

## At a glance

| | |
|---|---|
| Track | Applied ML — Phase 3 — Sequence Models & NLP |
| Projects | 10, about 31 days of work |
| Tasks | 71 (59 code, 12 written; 10 advanced) |
| Verification | 40 automated tests, 31 metric targets |
| Courses | CS224N NLP with Deep Learning, CME295 Transformers & LLMs, CS224U Natural Language Understanding |
| Textbooks | *Foundations of Machine Learning (2nd ed.)* (Mohri, Rostamizadeh & Talwalkar) |

## By the end of this repository I will be able to

- Train word2vec with negative sampling and evaluate embeddings on analogy and similarity tasks
- Build a transition-based dependency parser and recurrent language models with LSTMs
- Implement seq2seq with attention and the full Transformer from scratch
- Fine-tune BERT, GPT and T5-style models and explain what pretraining objective each uses
- Implement RoPE, a KV cache and an efficient attention variant, and measure what they buy
- Build retrieval-augmented question answering and semantic search, evaluated with recall@k and exact match
- Build an LLM agent with tool use, structured outputs and an evaluation harness

## Projects

| Directory | Project | Days | Lectures | Tasks | Status |
|---|---|---|---|---|---|
| [`0x00-word2vec`](./0x00-word2vec) | [Word Vectors: word2vec from Scratch](#0x00-word2vec--word-vectors-word2vec-from-scratch) | 3 | CS224N Lec 1–2 | 7 + 1 advanced | todo |
| [`0x01-dependency_parsing`](./0x01-dependency_parsing) | [Neural Dependency Parsing (Arc-Standard)](#0x01-dependency_parsing--neural-dependency-parsing-arc-standard) | 3 | CS224N Lec 3–4 | 7 + 1 advanced | todo |
| [`0x02-rnn_lstm`](./0x02-rnn_lstm) | [RNN & LSTM Language Models](#0x02-rnn_lstm--rnn--lstm-language-models) | 3 | CS224N Lec 5–6 | 5 + 1 advanced | todo |
| [`0x03-seq2seq_attention`](./0x03-seq2seq_attention) | [Seq2Seq with Attention](#0x03-seq2seq_attention--seq2seq-with-attention) | 3 | CS224N Lec 7–8 | 6 + 1 advanced | todo |
| [`0x04-transformer`](./0x04-transformer) | [The Transformer from Scratch](#0x04-transformer--the-transformer-from-scratch) | 5 | CS224N Lec 9; CME295 Lec 1–2 | 8 + 1 advanced | todo |
| [`0x05-pretraining_finetuning`](./0x05-pretraining_finetuning) | [Pretraining & Fine-tuning: BERT, GPT, T5](#0x05-pretraining_finetuning--pretraining--fine-tuning-bert-gpt-t5) | 3 | CS224N Lec 10–11 | 6 + 1 advanced | todo |
| [`0x06-rope_kvcache`](./0x06-rope_kvcache) | [Modern Transformer Internals: RoPE, KV-Cache, Efficient Attention](#0x06-rope_kvcache--modern-transformer-internals-rope-kv-cache-efficient-attention) | 3 | CME295 Lec 3–4 | 5 + 1 advanced | todo |
| [`0x07-rag_qa`](./0x07-rag_qa) | [Retrieval-Augmented Question Answering](#0x07-rag_qa--retrieval-augmented-question-answering) | 3 | CS224N Lec 12–13; CS224U Intro lectures | 6 + 1 advanced | todo |
| [`0x08-semantic_search`](./0x08-semantic_search) | [Sentence Embeddings & Semantic Search](#0x08-semantic_search--sentence-embeddings--semantic-search) | 2 | CS224U Sentence meaning, grounded language, retrieval & embeddings lectures | 6 + 1 advanced | todo |
| [`0x09-agents`](./0x09-agents) | [LLM Agents & Tool Use](#0x09-agents--llm-agents--tool-use) | 3 | CME295 Agents, tool use & multimodal lectures | 5 + 1 advanced | todo |

Status moves from `todo` to `in progress` to `done` as each project is submitted and passes its checks.

## Repository layout

```
ml-nlp/
├── README.md
├── 0x00-word2vec/
│   ├── README.md
│   ├── 0-softmax_loss.py
│   ├── 1-neg_sampling.py
│   ├── 2-skipgram.py
│   ├── 3-analogy.py
│   ├── 4-train.py
│   ├── results.json
│   └── … 4 more
├── 0x01-dependency_parsing/
│   ├── README.md
│   ├── 0-partial_parse.py
│   ├── 1-minibatch_parse.py
│   ├── 2-oracle.py
│   ├── 3-parser_model.py
│   ├── parser_utils.py
│   ├── 4-uas.py
│   └── … 5 more
├── 0x02-rnn_lstm/
│   ├── README.md
│   ├── 0-lstm_cell.py
│   ├── 1-lstm_forward.py
│   ├── 2-lm_utils.py
│   ├── 3-char_lstm.py
│   ├── results.json
│   ├── samples.txt
│   └── … 3 more
├── 0x03-seq2seq_attention/
│   ├── README.md
│   ├── 0-dot_attention.py
│   ├── 1-additive_attention.py
│   ├── 2-bleu.py
│   ├── 3-dates.py
│   ├── 4-seq2seq.py
│   ├── results.json
│   └── … 3 more
├── 0x04-transformer/
│   ├── README.md
│   ├── 0-attention.py
│   ├── 1-masks.py
│   ├── 2-multi_head_attention.py
│   ├── 3-components.py
│   ├── 4-transformer.py
│   ├── test_transformer.py
│   └── … 6 more
├── 0x05-pretraining_finetuning/
│   ├── README.md
│   ├── 0-mlm_masking.py
│   ├── 1-span_corruption.py
│   ├── 2-finetune_distilbert.py
│   ├── results.json
│   ├── 3-scratch_baseline.py
│   ├── written/4-pretraining_report.md
│   └── … 4 more
├── 0x06-rope_kvcache/
│   ├── README.md
│   ├── 0-rope.py
│   ├── 1-kv_cache.py
│   ├── 2-tiled_attention.py
│   ├── 3-model.py
│   ├── 3-generate.py
│   ├── results.json
│   └── … 2 more
├── 0x07-rag_qa/
│   ├── README.md
│   ├── 0-bm25.py
│   ├── 1-metrics.py
│   ├── 2-chunk.py
│   ├── 3-retrieval_eval.py
│   ├── results.json
│   ├── recall_at_k.png
│   └── … 4 more
├── 0x08-semantic_search/
│   ├── README.md
│   ├── 0-cosine_top_k.py
│   ├── 1-ranking_metrics.py
│   ├── 2-contrastive_loss.py
│   ├── 3-finetune.py
│   ├── results.json
│   ├── 4-search.py
│   └── … 2 more
└── 0x09-agents/
    ├── README.md
    ├── 0-tool_parser.py
    ├── 1-python_tool.py
    ├── 2-agent.py
    ├── tasks.jsonl
    ├── 3-eval.py
    ├── results.json
    └── … 2 more
```

## Conventions

- Python 3.10+. Every file starts with `#!/usr/bin/env python3`, ends with a newline, and has a docstring on every function. Nothing runs on import; demos live under `if __name__ == "__main__":`.
- Each task ships a `N-main.py` style example in its project README, so any function can be run on its own.
- Training scripts write their final metrics and configuration to `results.json` in the project directory; notebooks are committed executed, with outputs visible.
- Written answers live in `written/` as Markdown with LaTeX.
- Every project directory has its own `README.md`: what it contains, one line per file, and how to run it.

## Setup

```bash
git clone https://github.com/Olagold-hackxx/ml-nlp.git
cd ml-nlp
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib pandas jupyter torch torchvision
cd 0x00-word2vec   # each project README shows how to run its examples
```

## Project details

### 0x00-word2vec — Word Vectors: word2vec from Scratch

**3 days** · CS224N Lec 1–2

**Covers:** Intro & word vectors, word2vec details (skip-gram, negative sampling), GloVe, evaluation of word vectors.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| Softmax and the naive-softmax skip-gram loss | mandatory · code | `0-softmax_loss.py` |
| Negative-sampling loss and gradients | mandatory · code | `1-neg_sampling.py` |
| Skip-gram over a context window | mandatory · code | `2-skipgram.py` |
| Analogies and nearest neighbours | mandatory · code | `3-analogy.py` |
| Train word2vec on a real corpus | mandatory · code | `4-train.py`, `results.json` |
| Visualise the embedding space | mandatory · code | `5-visualize.ipynb`, `analogies.png` |
| Derivation: word2vec gradients | mandatory · written | `written/6-word2vec_gradients.pdf` |
| Noise distribution and frequent-word subsampling | advanced · code | `100-sampling.py` |

**Targets it has to hit**

- 5 automated Python tests against reference implementations and closed-form answers
- `corpus_tokens` ≥ 1000000 (Train word2vec on a real corpus), read from `results.json`
- `final_avg_loss` < 3 (Train word2vec on a real corpus), read from `results.json`
- `analogy_top10_accuracy` ≥ 0.08 (Train word2vec on a real corpus), read from `results.json`

**What I will be able to do**

- Explain why the naive softmax skip-gram objective is expensive and how negative sampling approximates it
- Derive the gradients of the naive-softmax and negative-sampling losses w.r.t. the center and outside vectors
- Explain why word2vec keeps two vectors per word and what you do with them after training
- Explain the role of the 3/4 power in the noise distribution and of frequent-word subsampling
- Solve analogies of the form a : b :: c : ? with cosine similarity and explain why a, b, c are excluded
- Read a 2-D projection of embeddings critically (what PCA preserves and what it hides)

**Key concepts:** Distributional hypothesis · Skip-gram vs CBOW · Naive softmax and its O(|V|) cost · Negative sampling and the unigram^0.75 noise distribution · Subsampling of frequent words · Center vs outside vectors · 3CosAdd word analogies and cosine similarity

<details><summary>Rules and resources</summary>

Rules:

- Python 3.10+, NumPy only for tasks 0–4 and 7 (no PyTorch, TensorFlow, gensim or scikit-learn in those files)
- Matplotlib and scikit-learn are allowed only in the visualization notebook
- All randomness seeded (np.random.default_rng(seed))

Resources:

- [CS224N course site (notes, assignments)](https://web.stanford.edu/class/cs224n/)
- [Mikolov et al. 2013 — Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Mikolov et al. 2013 — Distributed Representations of Words and Phrases (negative sampling, subsampling)](https://arxiv.org/abs/1310.4546)
- [Goldberg & Levy — word2vec Explained: deriving the negative-sampling objective](https://arxiv.org/abs/1402.3722)
- [D2L — Word Embedding (word2vec) & Approximate Training](https://d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html)

</details>

### 0x01-dependency_parsing — Neural Dependency Parsing (Arc-Standard)

**3 days** · CS224N Lec 3–4

**Covers:** Backprop & neural nets for NLP, dependency grammar, transition-based parsing, neural dependency parsing.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| Arc-standard partial parse | mandatory · code | `0-partial_parse.py` |
| Minibatch parsing | mandatory · code | `1-minibatch_parse.py` |
| Static oracle | mandatory · code | `2-oracle.py` |
| Neural parser model (PyTorch) | mandatory · code | `3-parser_model.py`, `parser_utils.py` |
| Unlabeled attachment score | mandatory · code | `4-uas.py` |
| Train and evaluate the parser | mandatory · code | `5-run.py`, `results.json`, `train_log.txt` |
| Written: transitions, dropout, Adam and error analysis | mandatory · written | `written/6-parsing.pdf` |
| Projectivity check | advanced · code | `100-projective.py` |

**Targets it has to hit**

- 5 automated Python tests against reference implementations and closed-form answers
- `dev_uas` ≥ 0.85 (Train and evaluate the parser), read from `results.json`
- `test_uas` ≥ 0.85 (Train and evaluate the parser), read from `results.json`

**What I will be able to do**

- Run the arc-standard system by hand on a sentence and explain why it needs exactly 2n transitions
- Write a static oracle that turns a projective gold tree into a transition sequence
- Explain why arc-standard cannot produce non-projective trees
- Describe the Chen & Manning feature set and network (embeddings → hidden ReLU → softmax over transitions)
- Explain how minibatching works when parses finish at different times
- Compute and interpret UAS, and classify common parsing errors (PP attachment, coordination, …)

**Key concepts:** Dependency trees, heads and dependents · Projectivity · Arc-standard transition system (SHIFT, LEFT-ARC, RIGHT-ARC) · Static oracle · Feature templates from stack/buffer configurations · Embedding lookup as matrix indexing · Unlabeled attachment score (UAS)

<details><summary>Rules and resources</summary>

Rules:

- Tasks 0–2, 4 and 7: pure Python 3.10+ (NumPy allowed), no parsing libraries (no spaCy, Stanza, NLTK parsers)
- Tasks 3 and 5: PyTorch; nn.Linear and nn.Embedding are forbidden in the model (use nn.Parameter)

Resources:

- [CS224N course site (Assignment on neural dependency parsing)](https://web.stanford.edu/class/cs224n/)
- [Chen & Manning 2014 — A Fast and Accurate Dependency Parser using Neural Networks](https://aclanthology.org/D14-1082/)
- [Nivre 2008 — Algorithms for Deterministic Incremental Dependency Parsing](https://aclanthology.org/J08-4003/)
- [Universal Dependencies (treebanks & guidelines)](https://universaldependencies.org/)
- [Kingma & Ba — Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

</details>

### 0x02-rnn_lstm — RNN & LSTM Language Models

**3 days** · CS224N Lec 5–6

**Covers:** Language models & RNNs, perplexity, LSTMs, vanishing/exploding gradients, gradient clipping.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| LSTM cell forward (NumPy) | mandatory · code | `0-lstm_cell.py` |
| LSTM over a sequence | mandatory · code | `1-lstm_forward.py` |
| Perplexity and temperature sampling | mandatory · code | `2-lm_utils.py` |
| Character-level LSTM on Shakespeare | mandatory · code | `3-char_lstm.py`, `results.json`, `samples.txt`, `loss_curve.png` |
| Derivation: vanishing gradients in RNNs | mandatory · written | `written/4-vanishing_gradients.pdf` |
| LSTM cell backward (NumPy) | advanced · code | `100-lstm_backward.py` |

**Targets it has to hit**

- 4 automated Python tests against reference implementations and closed-form answers
- `val_perplexity` ≤ 5 (Character-level LSTM on Shakespeare), read from `results.json`
- `val_bits_per_char` ≤ 2.33 (Character-level LSTM on Shakespeare), read from `results.json`

**What I will be able to do**

- Write the LSTM equations from memory and explain the role of each gate
- Derive why gradients vanish or explode in a vanilla RNN in terms of the recurrent Jacobian
- Explain why the LSTM cell state mitigates vanishing gradients (and why it does not fully solve it)
- Compute perplexity from per-token log-probabilities and relate it to cross-entropy and bits-per-character
- Train a character-level LSTM language model and sample from it at different temperatures
- Explain truncated BPTT and gradient-norm clipping

**Key concepts:** Autoregressive language modelling · Perplexity and bits-per-character · Backpropagation through time · Vanishing and exploding gradients · LSTM gates and the additive cell-state path · Gradient clipping · Temperature sampling

<details><summary>Rules and resources</summary>

Rules:

- Tasks 0–2 and 5: NumPy only
- Task 3: PyTorch (torch.nn.LSTM is allowed there)

Resources:

- [CS224N course site (RNN & LSTM lecture notes)](https://web.stanford.edu/class/cs224n/)
- [Karpathy — The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Olah — Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Pascanu, Mikolov & Bengio — On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063)
- [D2L — Long Short-Term Memory (LSTM)](https://d2l.ai/chapter_recurrent-modern/lstm.html)

</details>

### 0x03-seq2seq_attention — Seq2Seq with Attention

**3 days** · CS224N Lec 7–8

**Covers:** Machine translation, sequence-to-sequence, the bottleneck problem, attention, BLEU, beam search.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| Dot-product attention with masking | mandatory · code | `0-dot_attention.py` |
| Additive (Bahdanau) attention | mandatory · code | `1-additive_attention.py` |
| Corpus BLEU | mandatory · code | `2-bleu.py` |
| Date-normalisation dataset | mandatory · code | `3-dates.py` |
| Train seq2seq with and without attention | mandatory · code | `4-seq2seq.py`, `results.json`, `attention.png` |
| Report: what attention buys you | mandatory · written | `written/5-report.md` |
| Beam search | advanced · code | `100-beam_search.py` |

**Targets it has to hit**

- 5 automated Python tests against reference implementations and closed-form answers
- `test_exact_match` ≥ 0.97 (Train seq2seq with and without attention), read from `results.json`
- `test_char_bleu` ≥ 0.95 (Train seq2seq with and without attention), read from `results.json`

**What I will be able to do**

- Explain the bottleneck problem and how attention removes it
- Implement dot-product and additive attention with masking, and state their cost
- Compute corpus-level BLEU by hand and explain clipping and the brevity penalty
- Train an attention-based seq2seq model and compare it to the same model without attention
- Interpret an attention heatmap as a soft alignment
- Explain beam search, its stopping criteria and why it is not exact

**Key concepts:** Encoder–decoder architecture · Information bottleneck of a fixed context vector · Dot-product (Luong) vs additive (Bahdanau) attention · Masking padded positions · Teacher forcing and exposure bias · Greedy vs beam search decoding · Corpus BLEU: clipped n-gram precision and brevity penalty

<details><summary>Rules and resources</summary>

Rules:

- Tasks 0–3 and 6: NumPy / standard library only
- Task 4: PyTorch; attention must be your own code (no nn.MultiheadAttention, no F.scaled_dot_product_attention)

Resources:

- [Sutskever, Vinyals & Le — Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Bahdanau, Cho & Bengio — Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Luong, Pham & Manning — Effective Approaches to Attention-based NMT](https://arxiv.org/abs/1508.04025)
- [Papineni et al. — BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
- [D2L — Attention Mechanisms and Transformers (Bahdanau attention)](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html)

</details>

### 0x04-transformer — The Transformer from Scratch

**5 days** · CS224N Lec 9; CME295 Lec 1–2

**Covers:** Self-attention and Transformers; Transformer architecture deep-dive: attention, multi-head attention, positional encodings, encoder/decoder, layer norm, residuals.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| Scaled dot-product attention | mandatory · code | `0-attention.py` |
| Causal and padding masks | mandatory · code | `1-masks.py` |
| Multi-head attention (NumPy) | mandatory · code | `2-multi_head_attention.py` |
| LayerNorm, positional encoding, FFN and the Noam schedule | mandatory · code | `3-components.py` |
| The full Transformer in PyTorch | mandatory · code | `4-transformer.py`, `test_transformer.py` |
| Train on Multi30k (German → English) | mandatory · code | `5-train.py`, `results.json`, `loss_curve.png`, `samples.txt` |
| Annotated notes: Attention Is All You Need | mandatory · written | `notes/attention_is_all_you_need.md` |
| Label-smoothed cross-entropy | advanced · code | `100-label_smoothing.py` |

**Targets it has to hit**

- 5 automated Python tests against reference implementations and closed-form answers
- `test_bleu` ≥ 30 (Train on Multi30k (German → English)), read from `results.json`
- `reverse_task_accuracy` ≥ 0.99 (Train on Multi30k (German → English)), read from `results.json`

**What I will be able to do**

- Derive scaled dot-product attention as a differentiable soft dictionary lookup and justify the 1/√d_k scaling with a variance argument
- Implement causal and padding masks and prove that a causal decoder cannot see the future
- Explain what multiple heads buy compared to one head of the same total width, and count the parameters of every sub-layer
- Build a complete encoder–decoder Transformer in PyTorch without nn.Transformer / nn.MultiheadAttention
- Explain the O(T²·d) cost of self-attention and when it dominates the O(T·d²) cost of the FFN
- Explain why the Transformer needs positional information and how sinusoidal encodings make relative offsets linear
- Train the model to a competitive BLEU on Multi30k and debug it with unit tests

**Key concepts:** Queries, keys and values · Scaled dot-product attention and the 1/√d_k factor · Causal and padding masks · Multi-head attention (split / combine heads) · Sinusoidal positional encodings · Residual connections and layer normalisation (Post-LN vs Pre-LN) · Position-wise feed-forward network · Label smoothing and the warm-up (Noam) learning-rate schedule

<details><summary>Rules and resources</summary>

Rules:

- Tasks 0–3 and 8: NumPy only
- Tasks 4–5: PyTorch ≥ 2.0; forbidden: nn.Transformer*, nn.MultiheadAttention, F.scaled_dot_product_attention, nn.LayerNorm (write your own)
- A single consumer GPU or Colab T4 is enough for task 5 (≈ 1–2 h of training)

Resources:

- [Vaswani et al. 2017 — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [D2L — The Transformer Architecture](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)
- [Xiong et al. 2020 — On Layer Normalization in the Transformer Architecture (Pre-LN vs Post-LN)](https://arxiv.org/abs/2002.04745)
- [Ba, Kiros & Hinton — Layer Normalization](https://arxiv.org/abs/1607.06450)

</details>

### 0x05-pretraining_finetuning — Pretraining & Fine-tuning: BERT, GPT, T5

**3 days** · CS224N Lec 10–11

**Covers:** Pretraining (encoders, decoders, encoder–decoders: BERT, GPT, T5), fine-tuning, prompting & RLHF preview.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| BERT masked-LM corruption | mandatory · code | `0-mlm_masking.py` |
| T5 span corruption | mandatory · code | `1-span_corruption.py` |
| Fine-tune DistilBERT | mandatory · code | `2-finetune_distilbert.py`, `results.json` |
| From-scratch Transformer baseline | mandatory · code | `3-scratch_baseline.py`, `results.json` |
| Report: is pretraining worth it? | mandatory · written | `written/4-pretraining_report.md` |
| Annotated notes: BERT, GPT-2, T5 | mandatory · written | `notes/bert.md`, `notes/gpt2.md`, `notes/t5.md` |
| Few-shot prompting baseline | advanced · code | `100-fewshot.py`, `results.json` |

**Targets it has to hit**

- 2 automated Python tests against reference implementations and closed-form answers
- `distilbert_test_accuracy` ≥ 0.91 (Fine-tune DistilBERT), read from `results.json`
- `learning_curve.distilbert_1pct` > 0 (Fine-tune DistilBERT), read from `results.json`
- `scratch_test_accuracy` ≥ 0.8 (From-scratch Transformer baseline), read from `results.json`
- `learning_curve.scratch_1pct` > 0 (From-scratch Transformer baseline), read from `results.json`
- `fewshot_test_accuracy_k4` > 0 (Few-shot prompting baseline, advanced), read from `results.json`

**What I will be able to do**

- Explain the MLM objective and why BERT replaces only 80% of selected tokens with [MASK]
- Implement T5-style span corruption and explain why it shortens targets
- Compare encoder-only, decoder-only and encoder–decoder pretraining objectives and what each is good at
- Fine-tune DistilBERT for text classification with the Hugging Face stack
- Measure the benefit of pretraining against a from-scratch Transformer at several training-set sizes
- Read and summarise the BERT, GPT-2 and T5 papers critically

**Key concepts:** Masked language modelling (80/10/10 corruption) · Causal language modelling · Span corruption with sentinel tokens (T5) · Encoder-only vs decoder-only vs encoder–decoder · Subword tokenisation (WordPiece, BPE, SentencePiece) · Fine-tuning vs training from scratch: sample efficiency · Knowledge distillation (DistilBERT) · In-context (few-shot) learning

<details><summary>Rules and resources</summary>

Rules:

- Tasks 0–1: NumPy / standard library only
- Tasks 2–3 and 6: PyTorch + Hugging Face transformers/datasets; dataset: AG News or IMDb (pick one, use the official test split)

Resources:

- [Devlin et al. — BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Radford et al. — Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Raffel et al. — Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](https://arxiv.org/abs/1910.10683)
- [Sanh et al. — DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers — Fine-tuning a pretrained model](https://huggingface.co/docs/transformers/training)

</details>

### 0x06-rope_kvcache — Modern Transformer Internals: RoPE, KV-Cache, Efficient Attention

**3 days** · CME295 Lec 3–4

**Covers:** Scaling, positional encodings deep-dive (absolute, relative, RoPE), KV-cache, MQA/GQA, efficient attention (FlashAttention).

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| Rotary position embeddings | mandatory · code | `0-rope.py` |
| KV-cache: prefill and incremental decoding | mandatory · code | `1-kv_cache.py` |
| Tiled attention with online softmax | mandatory · code | `2-tiled_attention.py` |
| RoPE + KV-cache in your PyTorch Transformer | mandatory · code | `3-model.py`, `3-generate.py`, `results.json` |
| Derivation: RoPE, KV-cache memory and IO complexity | mandatory · written | `written/4-rope_kvcache.pdf` |
| Grouped-query attention | advanced · code | `100-gqa.py` |

**Targets it has to hit**

- 4 automated Python tests against reference implementations and closed-form answers
- `val_loss_rope` ≤ 1.6 (RoPE + KV-cache in your PyTorch Transformer), read from `results.json`
- `max_abs_logit_diff` ≤ 0.001 (RoPE + KV-cache in your PyTorch Transformer), read from `results.json`
- `speedup` ≥ 2 (RoPE + KV-cache in your PyTorch Transformer), read from `results.json`

**What I will be able to do**

- Prove that RoPE makes the query–key dot product depend only on the relative offset m − n
- Implement both RoPE conventions and show they are the same up to a fixed permutation of dimensions
- Explain why autoregressive decoding without a cache costs O(T²) per token and with a cache O(T)
- Compute the KV-cache size of a real model from its config and explain how GQA shrinks it
- Derive the online-softmax recurrence and implement exact attention in blocks without materialising the T×T matrix
- Measure the speed-up of cached generation and verify that it is numerically equivalent

**Key concepts:** Absolute vs relative position information · Rotary position embeddings (complex-number view) · Interleaved vs rotate-half RoPE conventions · Prefill vs decode phases of inference · KV-cache memory footprint · Online (streaming) softmax · IO-aware tiled attention · Multi-query and grouped-query attention

<details><summary>Rules and resources</summary>

Rules:

- Tasks 0–2 and 5: NumPy only
- Task 3: PyTorch, building on your A24 code (decoder-only variant)

Resources:

- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Dao et al. — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Milakov & Gimelshein — Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- [Shazeer — Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150)
- [Ainslie et al. — GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)

</details>

### 0x07-rag_qa — Retrieval-Augmented Question Answering

**3 days** · CS224N Lec 12–13; CS224U Intro lectures · Reading: FoML Ch 10

**Covers:** Question answering, natural language generation; Course overview, NLU tasks, retrieval-augmented in-context learning.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| BM25 from scratch | mandatory · code | `0-bm25.py` |
| Retrieval and QA metrics | mandatory · code | `1-metrics.py` |
| Chunking with overlap | mandatory · code | `2-chunk.py` |
| Sparse vs dense retrieval on SQuAD | mandatory · code | `3-retrieval_eval.py`, `results.json`, `recall_at_k.png` |
| RAG pipeline with an open LLM | mandatory · code | `4-rag.py`, `results.json`, `predictions.jsonl` |
| Error analysis report | mandatory · written | `written/5-error_analysis.md` |
| Hybrid retrieval with reciprocal rank fusion | advanced · code | `100-rrf.py` |

**Targets it has to hit**

- 4 automated Python tests against reference implementations and closed-form answers
- `n_queries` ≥ 1000 (Sparse vs dense retrieval on SQuAD), read from `results.json`
- `bm25_recall_at_5` ≥ 0.65 (Sparse vs dense retrieval on SQuAD), read from `results.json`
- `dense_recall_at_5` ≥ 0.7 (Sparse vs dense retrieval on SQuAD), read from `results.json`
- `n_eval` ≥ 200 (RAG pipeline with an open LLM), read from `results.json`
- `rag_f1` ≥ 0.5 (RAG pipeline with an open LLM), read from `results.json`

**What I will be able to do**

- Write the BM25 scoring function from memory and explain the role of k1 and b
- Implement and evaluate sparse and dense retrievers with recall@k
- Build a RAG pipeline (chunk → retrieve → prompt → generate) around an open LLM
- Evaluate generated answers with SQuAD EM/F1 and compare against closed-book generation
- Attribute errors to retrieval vs generation and propose fixes
- Explain when hybrid (sparse + dense) retrieval helps

**Key concepts:** Sparse retrieval: TF-IDF and BM25 (term saturation k1, length normalisation b) · Dense bi-encoder retrieval · Chunking with overlap · Recall@k as the ceiling for the reader · Hybrid retrieval and reciprocal rank fusion · Closed-book vs open-book (retrieval-augmented) QA · SQuAD exact match and token F1

<details><summary>Rules and resources</summary>

Rules:

- Tasks 0–2 and 6: pure Python (NumPy allowed); no rank_bm25, Pyserini, scikit-learn or LangChain in those files
- Tasks 3–4: sentence-transformers / transformers allowed; any open-weights LLM ≤ 8B parameters (no closed APIs)

Resources:

- [Lewis et al. — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Karpukhin et al. — Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906)
- [Rajpurkar et al. — SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)
- [Khattab & Zaharia — ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832)
- [CS224U course site](https://web.stanford.edu/class/cs224u/)

</details>

### 0x08-semantic_search — Sentence Embeddings & Semantic Search

**2 days** · CS224U Sentence meaning, grounded language, retrieval & embeddings lectures · Reading: FoML Ch 10

**Covers:** Sentence representations, contrastive learning, neural IR, evaluation of retrieval.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| Cosine top-k search | mandatory · code | `0-cosine_top_k.py` |
| MRR@k and nDCG@k | mandatory · code | `1-ranking_metrics.py` |
| Multiple-negatives ranking loss | mandatory · code | `2-contrastive_loss.py` |
| Fine-tune SBERT on your own corpus | mandatory · code | `3-finetune.py`, `results.json` |
| Search interface | mandatory · code | `4-search.py` |
| Report: what did fine-tuning change? | mandatory · written | `written/5-report.md` |
| Hard-negative mining | advanced · code | `100-hard_negatives.py` |

**Targets it has to hit**

- 4 automated Python tests against reference implementations and closed-form answers
- `n_corpus_docs` ≥ 1000 (Fine-tune SBERT on your own corpus), read from `results.json`
- `n_test_queries` ≥ 100 (Fine-tune SBERT on your own corpus), read from `results.json`
- `ndcg_gain` ≥ 0.02 (Fine-tune SBERT on your own corpus), read from `results.json`

**What I will be able to do**

- Explain why raw BERT [CLS] vectors are poor sentence embeddings and how SBERT fixes it
- Implement top-k cosine search and the MRR/nDCG metrics
- Derive the multiple-negatives ranking loss as a softmax over in-batch candidates
- Fine-tune a sentence-embedding model on your own corpus and measure the gain over the off-the-shelf model
- Explain the risk of false negatives in hard-negative mining

**Key concepts:** Bi-encoders and mean pooling · Cosine similarity and normalised embeddings · Contrastive (InfoNCE / multiple-negatives ranking) loss · In-batch and hard negatives · MRR@k and nDCG@k · Domain adaptation of embedding models

<details><summary>Rules and resources</summary>

Rules:

- Tasks 0–2 and 6: NumPy only
- Task 3–4: sentence-transformers / PyTorch

Resources:

- [Reimers & Gurevych — Sentence-BERT](https://arxiv.org/abs/1908.10084)
- [Sentence-Transformers documentation (training, losses, evaluation)](https://www.sbert.net/)
- [Henderson et al. — Efficient Natural Language Response Suggestion (in-batch negatives)](https://arxiv.org/abs/1705.00652)
- [Thakur et al. — BEIR: a Heterogeneous Benchmark for Zero-shot IR](https://arxiv.org/abs/2104.08663)
- [Gao, Yao & Chen — SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)

</details>

### 0x09-agents — LLM Agents & Tool Use

**3 days** · CME295 Agents, tool use & multimodal lectures

**Covers:** Tool calling, ReAct, planning, retrieval as a tool, agent evaluation, vision–language models.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| Tool-call and final-answer parser | mandatory · code | `0-tool_parser.py` |
| Python tool with timeout | mandatory · code | `1-python_tool.py` |
| The ReAct agent loop | mandatory · code | `2-agent.py` |
| Evaluate the agent on 20 tasks | mandatory · code | `tasks.jsonl`, `3-eval.py`, `results.json`, `traces.jsonl` |
| Medium write-up: building a small agent | mandatory · written | — |
| Second tool: retrieval | advanced · code | `100-search_tool.py`, `results.json` |

**Targets it has to hit**

- 2 automated Python tests against reference implementations and closed-form answers
- `n_tasks` = 20 (Evaluate the agent on 20 tasks), read from `results.json`
- `agent_accuracy` ≥ 0.6 (Evaluate the agent on 20 tasks), read from `results.json`
- `agent_accuracy_two_tools` ≥ 0 (Second tool: retrieval, advanced), read from `results.json`
- `tool_choice_accuracy` ≥ 0 (Second tool: retrieval, advanced), read from `results.json`

**What I will be able to do**

- Explain the ReAct pattern and how tool observations are fed back into the context
- Write a robust parser for model-emitted tool calls and final answers
- Implement an agent loop with a step budget and graceful handling of malformed calls, unknown tools and tool exceptions
- Run Python code from an LLM safely enough for local experiments (subprocess, timeout, output limits) and explain what is still unsafe
- Design a small evaluation set and measure the effect of tools on accuracy
- Communicate results and failure modes in a public write-up

**Key concepts:** Chat templates and tool-calling formats · The ReAct loop (thought → action → observation) · Structured output parsing and error recovery · Sandboxing tool execution · Step budgets and termination · Agent evaluation: task success, steps, failure taxonomy · Workflows vs autonomous agents

<details><summary>Rules and resources</summary>

Rules:

- Tasks 0 and 2: pure Python standard library; 2-agent.py must not import any LLM library at module level
- Tasks 3 and 5: an open-weights LLM ≤ 8B with tool-calling ability (e.g. Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct) via transformers, vLLM, llama.cpp or Ollama — no closed APIs

Resources:

- [Yao et al. — ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Schick et al. — Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [Gao et al. — PAL: Program-aided Language Models](https://arxiv.org/abs/2211.10435)
- [Anthropic — Building effective agents](https://www.anthropic.com/research/building-effective-agents)
- [Weng — LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)

</details>

## Part of a larger programme

This repository is one of seven that together cover a Stanford-curriculum machine learning programme, which I am working through alongside a BSc in Applied Mathematics and Computer Science at the University of Lagos:

- [ML Foundations](https://github.com/Olagold-hackxx/ml-foundations): Classical machine learning implemented from scratch in NumPy, from least squares to value iteration.
- [ML Deep Learning](https://github.com/Olagold-hackxx/ml-deep-learning): Deep learning from layer internals to a deployed vision application.
- [ML NLP](https://github.com/Olagold-hackxx/ml-nlp) (this repository): Sequence models and language models, from word2vec to retrieval-augmented agents.
- [ML Generative Models & RL](https://github.com/Olagold-hackxx/ml-generative-rl): Generative models and reinforcement learning, from PixelCNN and DDPM to PPO, RLHF and DPO.
- [ML Frontier](https://github.com/Olagold-hackxx/ml-frontier): Language models from scratch: tokenizers, architectures, training systems, scaling laws, post-training and evaluation.
- [ML Capstone](https://github.com/Olagold-hackxx/ml-capstone): A research-grade capstone project: proposal, milestone and final report.
- [ML Theory](https://github.com/Olagold-hackxx/ml-theory): Statistical learning theory with full proofs, from PAC learning to double descent.

Author: [Abdulqoyyum Aileru](https://github.com/Olagold-hackxx) · [LinkedIn](https://www.linkedin.com/in/abdulqoyyum-aileru/)

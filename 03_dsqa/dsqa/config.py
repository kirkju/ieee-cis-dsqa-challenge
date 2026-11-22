"""The numbers the entry runs with, and where each comes from.

They were set on the development corpus, with retrieval measured against
reference passages and no model run (docs/journal.md in our repository). A
number that a run with the base model has not confirmed says so.
"""

from pathlib import Path

# The embedding model, shipped in the zip next to main.py: the evaluation has
# no internet to fetch it by name. It is the organizers' example's,
# sentence-transformers/all-MiniLM-L6-v2.
EMBEDDER = Path(__file__).resolve().parent.parent / "embed_model"

# What that embedder reads of a chunk: 256 tokens, two of them its own
# markers. A chunk is cut to this, so that all of it is in its embedding.
CHUNK_TOKENS = 254
# A table may take this much of a chunk, more than the embedder reads: it
# stays in one piece under its caption, and the rows past the embedder's
# reach are still matched by their words.
TABLE_TOKENS = 500

# Tokens of context given to the model with a question, counted with the base
# model's tokenizer. The organizers' own figure: prompts anywhere near the
# model's window ran the GPU out of memory, and long contexts with long
# answers took minutes a question. With the base model on an H100 this much
# is read in 3.3 s and peaks at 64.5 GiB of the card's 79.6; 30,000 tokens
# peak at 73.4 and 38,500 do not fit. Whether more answers better has not
# been tried.
CONTEXT_TOKENS = 10_000
# The document of the best chunk goes into the context whole when it takes no
# more than this share of it: an answer that needs two passages of a short
# paper then has both.
WHOLE = 0.7
# After the GPU ran out of memory on a prompt, the context is halved for that
# question and for the ones after it, down to this.
LEAST_CONTEXT_TOKENS = 1_250

# The most tokens an answer may take. The reference answers of the sample run
# from a sentence to several paragraphs; the organizers' examples allow
# 2,048. The base model writes 14 tokens a second on an H100, so the time a
# question may take ends an answer before this does.
ANSWER_TOKENS = 1_024
# Seconds of a question's time kept back from its working, to have the answer
# asked for apart when the working was cut before it stated one: of the 98
# answers of the first run with the base model, six ran into the time a
# question may take. A question with less than three times this to spend
# gives all of it to the working.
FINAL_SECONDS = 8
# The most tokens the answer may take when it is asked for apart.
FINAL_TOKENS = 96

# The time a run has, as the rules give it: an hour for the 50 questions of
# the public leaderboard and two for the 100 of the private one, counted from
# when our code starts running. So every 50 questions have an hour, and the
# run is to be done MARGIN_SECONDS before each hour is up.
HOUR_SECONDS = 3_600
QUESTIONS_AN_HOUR = 50
MARGIN_SECONDS = 360
# The most a single question may take, whatever time is left.
QUESTION_SECONDS = 50
# With less than this left for a question, the model is not asked: the
# passage found for it is the answer.
LEAST_SECONDS = 4
# Reading the corpus stops after this long, and what was read is indexed: a
# corpus larger than we planned for must not cost the whole run.
READING_SECONDS = 900
# The base model as a judge of relevance (relevance.py), for llm-rank,
# llm-gate, llm-doc, llm-pick and llm-wide; in this version only a question
# answered the dones-plain way is judged (route.py). How many of the first
# chunks of a ranking it reads, each in a prompt of its own: 40 of about 360
# tokens are 14,000 tokens, read in about 4.6 s a question on the H100 pod
# (not measured); with llm-wide, 80.
RELEVANCE_N = 40
RELEVANCE_WIDE_N = 80
# llm-gate: a chunk the model gives less than this P(Yes) is left out, but
# for the best this many, which are kept whatever their score.
RELEVANCE_GATE = 0.2
RELEVANCE_KEEP = 3
# llm-doc: a document scores the sum of the scores of its best this many chunks.
RELEVANCE_DOC_CHUNKS = 3
# llm-pick: the model is asked to choose when the first two documents score
# within this much of each other. A first guess, to be set on the tuning set.
RELEVANCE_PICK_MARGIN = 0.5
# The most tokens a passage is scored with, and of each passage of a document
# that llm-pick shows.
RELEVANCE_BODY_TOKENS = 600
RELEVANCE_CARD_TOKENS = 350
# The most tokens engine.score() reads in one pass, padding counted.
SCORE_TOKENS = 8_000
# With less than this left of a question's time, its passages are not scored.
RELEVANCE_SECONDS = 15
# The chunks are scored this many at a time, and no slice is begun that would
# end past this many seconds of scoring, or half of what is left of the
# question's time. 80 chunks are about 28,000 tokens: about 9 s on the H100
# pod and 17 s on an A100, which reads about 1.8 times slower (not measured).
RELEVANCE_SLICE = 20
RELEVANCE_MOST_SECONDS = 30
# With dones-example, the organizers' RAG example makes its index of the
# corpus, Docling's reading and MiniLM's embedding on the CPU, in a process
# of its own before the base model is loaded (example.py), and that process
# is stopped this many seconds after our code started. Then there is no
# index, and the questions on IFMIF-DONES are answered the entry's way.
#
# Measured on the laptop with the example's own code: Docling reads 0.074 to
# 0.10 s a page on its GPU, and MiniLM embeds 94 of the example's chunks a
# second on 16 threads, the 13,975 of the development corpus's 2,943 pages
# in 148 s. A corpus two or three times ours would take 450 to 940 s to
# read and 300 to 450 s to embed at that pace. The example's own run on the
# evaluation's machine took 2,080 s with its 50 answers and the model's
# loading, so its index took no more than about 1,500 s there.
#
# Ours after the index, at the 28 tokens a second of the evaluation's
# machine: the model loaded in up to 220 s, our index in about 150 s, and
# 50 questions in about 600 s; done by 2,770 s when the index ends at 1,800,
# within the 3,240 the hour leaves after MARGIN_SECONDS. On a machine that
# writes at 14 tokens a second, as the rented pod did, the questions would
# want twice that: the share of the hour each question is given (budget.py)
# and the emergency limit of the RAG example's replies (route.py) cut the
# answers short before the hour is passed. With 100 questions and two hours
# the index is made once, in the first hour, and the second hour is the
# next 50 questions' alone.
EXAMPLE_INDEX_SECONDS = 1_800

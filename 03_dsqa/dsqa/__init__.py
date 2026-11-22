"""Our entry: what main.py's answer() is made of.

    config     the numbers the entry runs with, and where each comes from
    text       a text with the marks of its layout undone
    tables     the tables of a PDF page, read as rows of cells
    parse      a PDF as text: its title, its blocks and its tables
    chunk      that text in chunks that follow its structure
    rank       the chunks ranked for a question by their words, and rankings fused
    context    the chunks that go into the context of a question
    retrieval  the corpus read, chunked and indexed: from a folder to a context
    prompt     what the model is told, and the passages laid out for it to read
    engine     what generates: the base model, behind one boundary
    budget     how long a question may take
    answering  a question answered, and what is said when a part fails

Everything here ships in the submission zip, next to main.py, and nothing
here imports from outside it but the pinned packages.
"""

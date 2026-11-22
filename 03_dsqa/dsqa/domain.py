"""The domain a question is about, told by code: IFMIF-DONES or AI papers.

The evaluation does not say which of its two domains a question is from,
and the entry may tell the model different things for each (by-domain,
variant.py). So the domain is told here from words, with no model: those
of the question, and those that open the document of the passage ranked
first for it, its title and its first chunk. Each is counted against two
lists of words, one for each domain, and the domain is the one with more;
when neither has more, the domain is not told and nothing is said for it.

The words are ones that the other domain does not use, or not in that
sense. The literature of a neutron source speaks of beams, targets,
accelerators, activation, models, layers, losses, images and the accuracy
of a measurement, and AI papers of beam search, targets, hardware
accelerators and activation functions: those words are in neither list,
or only in phrases that are the domain's own. On the 150 documents of the
development corpus, words of the IFMIF-DONES list take from 5.6 to 66 of
every thousand words of a DONES document and at most 0.42 of an AI paper.
"""

import re

DONES, AI = "DONES", "AI"

# Words of the IFMIF-DONES literature, as whole words, in any case.
_DONES_WORDS = re.compile(
    r"\b(?:"
    r"ifmif|dones|eveda|lipac|deuterons?|neutrons?|neutronics?|lithium|li\(d, ?xn\)|"
    r"irradiat\w*|tritium|dpa|hftm|test cell|test modules?|rfq|srf|linacs?|hebt|mebt|lebt|"
    r"cryomodules?|tokamaks?|plasma|radiolog\w*|radioprotection|radioactiv\w*|"
    r"msv|usv|sv/h|[gt]?bq|eurofusion|f4e|fusion (?:materials?|reactors?|power|devices?|"
    r"facilit(?:y|ies)|plants?|energy)|neutron sources?|"
    r"(?:proton|deuteron|ion|particle|accelerator) beams?|beam ?lines?|"
    r"beam (?:currents?|dumps?|loss(?:es)?|halo|profiles?|diagnostics|instrumentation|"
    r"energ(?:y|ies)|power|optics)|accelerator(?:-based| facilit(?:y|ies)| systems?)|"
    r"(?:linear|particle) accelerators?|emittance|bunch lengths?|otr|transition radiation|"
    r"activation products|activated corrosion products|corrosion products|neutron activation|"
    r"shielding|dose rates?|helium|(?:rf|srf|superconducting|accelerating) cavit(?:y|ies)|"
    r"solenoids?|remote handling|hot cells?"
    r")\b",
    re.IGNORECASE,
)
# Words of the AI papers, among them those of the mathematics they have.
_AI_WORDS = re.compile(
    r"\b(?:"
    r"neural|deep learning|machine learning|learning|transformers?|datasets?|benchmarks?|"
    r"fine-?tun\w*|pre-?train\w*|llms?|language models?|gpus?|embeddings?|classifiers?|"
    r"convolutional|cnns?|rnns?|lstms?|hyperparameters?|epochs?|backpropagation|"
    r"self-attention|attention (?:heads?|layers?|mechanisms?|weights|maps?)|encoders?|"
    r"decoders?|tokens?|prompts?|fuzzy|cognitive maps?|theorems?|lemmas?|banach|hilbert|"
    r"training (?:sets?|data|loss)|test sets?|learning rates?|reinforcement|chatbots?"
    r")\b",
    re.IGNORECASE,
)


def counted(text: str) -> tuple[int, int]:
    """How many words of a text are of each list: IFMIF-DONES's, then AI's."""
    return len(_DONES_WORDS.findall(text)), len(_AI_WORDS.findall(text))


def of(question: str, opening: str = "") -> str | None:
    """The domain of a question, told by its words and by those that open
    the document of its best passage; None when they tell neither."""
    dones, ai = (sum(pair) for pair in zip(counted(question), counted(opening), strict=True))
    if dones > ai:
        return DONES
    if ai > dones:
        return AI
    return None

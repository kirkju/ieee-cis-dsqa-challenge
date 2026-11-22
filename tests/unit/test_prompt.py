"""dsqa/prompt.py: what the model is told and shown."""

from dsqa import prompt

LOOP = "A study of the lithium loop"
TRAP = "Notes on the nitrogen trap"
# Chunks as the entry makes them, each under the title of its document:
# three of the first document and two of the second.
CHUNKS = [
    f"{LOOP}\n\nThe loop carries the lithium to the target.",
    f"{LOOP}\n\nIt is driven by an electromagnetic pump.",
    f"{LOOP}\n\nTable 1. Flows.\n\n| Main loop | 97.5 |",
    f"{TRAP}\n\nThe trap is in the dump tank.",
    f"{TRAP}\n\nIt is operated at 500 to 600 degrees.",
]
OF = [0, 0, 0, 1, 1]


def placed(*numbers):
    return [(number, CHUNKS[number]) for number in numbers]


def test_a_chunk_is_the_title_of_its_document_and_the_rest():
    assert prompt.title_and_body(CHUNKS[2]) == (LOOP, "Table 1. Flows.\n\n| Main loop | 97.5 |")
    assert prompt.title_and_body("a title alone") == ("a title alone", "")


def test_the_chunks_of_a_document_are_shown_together_under_its_title_said_once():
    # Placed as ranked: the trap's second chunk first, then the loop's.
    shown = prompt.context_of(placed(4, 0, 1, 3), OF)
    assert shown == (
        f"Document: {LOOP}\n\n"
        "The loop carries the lithium to the target.\n\n"
        "It is driven by an electromagnetic pump.\n\n\n"
        f"Document: {TRAP}\n\n"
        "The trap is in the dump tank.\n\n"
        "It is operated at 500 to 600 degrees."
    )
    assert shown.count(LOOP) == shown.count(TRAP) == 1
    # The document of the best passage is the last one, next to the question.
    assert prompt.context_of(placed(0, 4), OF).index(LOOP) > 0


def test_a_mark_stands_where_two_chunks_do_not_follow_one_another():
    shown = prompt.context_of(placed(2, 0), OF)
    assert shown == (
        f"Document: {LOOP}\n\n"
        "The loop carries the lithium to the target.\n\n"
        "[...]\n\n"
        "Table 1. Flows.\n\n| Main loop | 97.5 |"
    )
    assert prompt.context_of([], OF) == ""
    # A chunk cut to fit, with nothing under its title, is shown as it is.
    assert prompt.context_of([(0, "A study of")], OF) == "Document: A study of\n\nA study of"


def test_the_model_is_given_the_instructions_then_the_passages_and_the_question():
    system, user = prompt.messages("  Where is the trap? ", "the passages")
    assert system == {"role": "system", "content": prompt.SYSTEM}
    assert user["role"] == "user"
    assert user["content"].startswith("Passages retrieved from the document collection:\n\n")
    assert user["content"].endswith("the passages\n\n---\n\nQuestion: Where is the trap?")
    # Without passages, the question alone.
    assert prompt.messages("Where is the trap?")[1]["content"] == "Where is the trap?"
    # The instructions ask for what the judge and the experts look for, the
    # working first and the answer after it, once.
    for asked_for in ("calculation", "contradict", "State no answer before"):
        assert asked_for in prompt.SYSTEM
    assert prompt.SYSTEM.index("Work the question out first") < prompt.SYSTEM.index(prompt.FINAL)


def test_a_reply_is_its_working_and_the_answer_it_states_last():
    working = "The main loop carries 97.5 l/s and the bypass 2.015.\n2.015 / 97.5 = 0.021."
    assert prompt.split(f"{working}\nFinal answer: About 2%.") == (working, "About 2%.")
    # Under the emphasis a model puts on the line, and in its own capitals.
    assert prompt.split("It decays by half in 53 days.\n\n**Final Answer:** 53.22 days") == (
        "It decays by half in 53 days.",
        "53.22 days",
    )
    assert prompt.split("### final answer: yes") == ("", "yes")
    # The last such line counts, and all that follows it is the answer.
    stated_twice = "Final answer: 32\nNo, the table says 81.\nFinal answer: 81,\nfrom Table 2."
    assert prompt.split(stated_twice) == (
        "Final answer: 32\nNo, the table says 81.",
        "81,\nfrom Table 2.",
    )
    # A reply that states none, or was cut as it came to it.
    assert prompt.split("So the final answer is 2%.") is None
    assert prompt.split("The main loop carries 97.5 l/s.\nFinal answer:") is None
    assert prompt.split("") is None


def test_what_is_returned_of_a_reply_puts_the_answer_before_its_working():
    reply = "2.015 / 97.5 = 0.021.\nFinal answer: About 2%."
    assert prompt.as_answer(reply) == "About 2%.\n\n2.015 / 97.5 = 0.021."
    assert prompt.as_answer("Final answer: About 2%.") == "About 2%."
    # One that states no answer is returned as it is.
    assert prompt.as_answer("About 2%, by the table.") == "About 2%, by the table."


def test_a_reply_that_states_no_answer_is_shown_back_and_its_answer_asked_for():
    system, user, said, asked = prompt.closing("Where is the trap?", "the passages", "It is in the")
    assert [system, user] == prompt.messages("Where is the trap?", "the passages")
    assert said == {"role": "assistant", "content": "It is in the"}
    assert asked == {"role": "user", "content": prompt.CLOSE}
    assert prompt.FINAL in prompt.CLOSE

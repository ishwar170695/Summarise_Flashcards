import genanki
import io
import random
import hashlib

def build_anki_deck(questions, deck_name="StudySummarizer Deck"):
    model_id = int(hashlib.sha1(deck_name.encode("utf-8")).hexdigest(), 16) % (10 ** 10)
    model = genanki.Model(
        model_id,
        'StudySummarizerModel',
        fields=[{"name": "Question"}, {"name": "Answer"}],
        templates=[{
            "name": "Card 1",
            "qfmt": "{{Question}}",
            "afmt": "{{FrontSide}}<hr id='answer'>{{Answer}}"
        }]
    )

    deck_id = random.randint(1_000_000, 9_999_999)
    deck = genanki.Deck(deck_id, deck_name)

    for question, answer in questions:
        deck.add_note(genanki.Note(
            model=model,
            fields=[question.strip(), answer.strip()]
        ))

    output = io.BytesIO()
    genanki.Package(deck).write_to_file(output)
    return output.getvalue()

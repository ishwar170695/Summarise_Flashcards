import genanki

def create_summary_deck(summaries, deck_name="AI Summaries", output_path="ai_summaries.apkg"):
    model = genanki.Model(
        1607392319,
        'Simple Model',
        fields=[
            {'name': 'Question'},
            {'name': 'Answer'}
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Question}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
            },
        ]
    )

    deck = genanki.Deck(
        2059400110,
        deck_name
    )

    for i, summary in enumerate(summaries, 1):
        note = genanki.Note(
            model=model,
            fields=[f"Summary {i}", summary]
        )
        deck.add_note(note)

    genanki.Package(deck).write_to_file(output_path)
    print(f"✅ Deck saved as '{output_path}' with {len(summaries)} summaries.")

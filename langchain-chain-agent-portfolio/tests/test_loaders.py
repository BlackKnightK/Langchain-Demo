from chain_question_generator.loaders import load_document


def test_load_txt(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Hello   world.\n\n\nSecond paragraph.", encoding="utf-8")
    text = load_document(path)
    assert "Hello world." in text
    assert "Second paragraph." in text

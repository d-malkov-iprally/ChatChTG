"""Room counting without an LLM: read the printed labels, count in code.

``questions`` recognises a room-count question, ``ocr`` reads the labels, ``lexicon`` maps
them to the question, ``geometry`` counts apartments and storeys from shapes, and ``router``
picks the method for each kind of question.
"""

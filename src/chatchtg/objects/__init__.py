"""Object counting without an LLM: count the symbols that a trained detector found.

``questions`` recognises an object-count question about a Finnish (CubiCasa5K) plan,
``counter`` counts the stored detections of the drawing, ``importer`` stores the detections
of a batch prediction (``object_detector/modal_predict.py``) as extractions, and ``tune``
and ``report`` choose the minimum scores and score the counter on the benchmark.
"""

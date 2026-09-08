__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import codecs
import os
import tempfile
from html import escape

from lexnlp.extract.common.annotations.text_annotation import TextAnnotation
from lexnlp.extract.common.base_path import lexnlp_test_path


def load_resource_document(doc_path: str, encoding: str = "ascii") -> str:
    """
    Load a resource file from the package test data directory and return its contents as a string.

    Parameters:
        doc_path (str): Path to the file relative to the package's test data root.
        encoding (str): Encoding used to read the file (default: "ascii").

    Returns:
        str: The file contents.
    """
    full_path = os.path.join(lexnlp_test_path, doc_path)
    with codecs.open(full_path, encoding=encoding, mode="r") as fr:
        data = fr.read()
    return data


def save_test_document(doc_path: str, text: str, encoding: str = "utf-8") -> str:
    """Write annotated debug output and return where it went.

    This used to write into the tracked ``test_data`` tree, so every test run
    left the working copy dirty with output nothing asserts on. Output now goes
    to a scratch directory: ``LEXNLP_TEST_OUTPUT_DIR`` when set, otherwise a
    per-user directory under the system temp location.
    """
    output_root = os.environ.get("LEXNLP_TEST_OUTPUT_DIR") or os.path.join(
        tempfile.gettempdir(), "lexnlp-test-output"
    )
    full_path = os.path.join(output_root, doc_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with codecs.open(full_path, encoding=encoding, mode="w") as fw:
        fw.write(text)
    return full_path


def annotate_text(text: str, annotations: list[TextAnnotation]) -> str:
    """
    :param text: source text
    :param annotations: text annotations
    :return: html, where annotations are replaced by HREFs + annotations' list in the end of the document
    """
    annotations.sort(key=lambda a: a.coords[1])
    for i, ant in enumerate(annotations):
        ant.index = i + 1
    result = """
<html>
<head>
  <meta charset="UTF-8">
</head>
<body>
  <p>
    """
    end = 0
    for ant in annotations:
        part = text[end : ant.coords[0]]
        result += escape(part).replace("\n", "<br/>")

        title = "[%d] " % ant.index + escape(ant.text).replace('"', "'")
        rf = '<a href="#" title="%s">' % title
        link_title = escape(text[ant.coords[0] : ant.coords[1]])
        if ant.index > 0:
            link_title = "REFR#%d " % ant.index + link_title
        rf += link_title
        rf += "</a>"
        result += rf

        end = ant.coords[1]
        if end >= len(text):
            break
    if end < len(text):
        part = text[end:-1]
        result += escape(part).replace("\n", "<br/>")

    result += "</p> <ol>"
    for ant in annotations:
        item = escape(ant.name)
        explanation = escape(ant.get_extracted_text(text))
        result += "<li><b>%s</b>: %s</li>" % (item, explanation)
    result += """
  </ol>
</body>
</html>"""
    return result

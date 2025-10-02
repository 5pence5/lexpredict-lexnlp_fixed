__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import os


lexnlp_base_path = os.path.abspath(os.path.dirname(__file__) + '/../../../')

_test_path_override = os.environ.get('LEXNLP_TEST_DATA_PATH') or os.environ.get('LEXNLP_TEST_DATA')
if _test_path_override:
    lexnlp_test_path = os.path.abspath(_test_path_override)
else:
    lexnlp_test_path = os.path.join(lexnlp_base_path, 'test_data')

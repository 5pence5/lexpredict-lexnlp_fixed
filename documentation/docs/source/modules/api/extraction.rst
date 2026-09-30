Additional extraction APIs
==========================

The locale-specific guides cover the most commonly used extractors. This page
documents the cross-locale dispatch layer and the maintained extractors that do
not yet have dedicated guides.

Cross-locale dispatch
---------------------

.. automodule:: lexnlp.extract.all_locales.amounts
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.all_locales.citations
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.all_locales.copyrights
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.all_locales.court_citations
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.all_locales.courts
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.all_locales.dates
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.all_locales.definitions
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.all_locales.durations
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.all_locales.geoentities
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.all_locales.languages
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.all_locales.money
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.all_locales.percents
   :members:
   :show-inheritance:

English
-------

.. automodule:: lexnlp.extract.en.addresses.addresses
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.en.contracts.predictors
   :members:
   :show-inheritance:

.. currentmodule:: lexnlp.extract.en.dict_entities

.. autoclass:: DictionaryEntryAlias

.. autoclass:: DictionaryEntry

.. autoclass:: AliasBanRecord

.. autoclass:: AliasBanList

.. autoclass:: SearchResultPosition

.. autoclass:: DictionaryEntity

.. autofunction:: normalize_text

.. autofunction:: alias_is_banlisted

.. autofunction:: conflicts_take_first_by_id

.. autofunction:: conflicts_top_by_priority

.. autofunction:: prepare_alias_banlist_dict

.. automodule:: lexnlp.extract.en.entities.nltk_re
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.en.entities.nltk_tokenizer
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.en.entities.stanford_ner
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.en.preprocessing.span_tokenizer
   :members:
   :show-inheritance:

German
------

.. automodule:: lexnlp.extract.de.copyrights
   :members:
   :show-inheritance:

.. currentmodule:: lexnlp.extract.de.court_citations

.. autofunction:: get_court_citation_annotations

.. autofunction:: get_court_citation_annotation_list

.. autofunction:: get_court_citations

.. autofunction:: get_court_citation_list

.. automodule:: lexnlp.extract.de.courts
   :members:
   :show-inheritance:

.. currentmodule:: lexnlp.extract.de.definitions

.. autofunction:: get_definition_annotations

.. autofunction:: get_definition_annotation_list

.. autofunction:: get_definitions

.. autofunction:: get_definition_list

.. automodule:: lexnlp.extract.de.geoentities
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.de.laws
   :members:
   :show-inheritance:

.. currentmodule:: lexnlp.extract.de.money

.. autofunction:: get_money

.. autofunction:: get_money_list

.. autofunction:: get_money_annotations

.. autofunction:: get_money_annotation_list

Spanish
-------

.. automodule:: lexnlp.extract.es.copyrights
   :members:
   :show-inheritance:

.. automodule:: lexnlp.extract.es.courts
   :members:
   :show-inheritance:

.. currentmodule:: lexnlp.extract.es.definitions

.. autofunction:: get_definition_annotations

.. autofunction:: get_definition_annotation_list

.. autofunction:: get_definitions

.. autofunction:: get_definition_list

.. currentmodule:: lexnlp.extract.es.regulations

.. autofunction:: get_regulation_annotations

.. autofunction:: get_regulation_annotation_list

.. autofunction:: get_regulations

.. autofunction:: get_regulation_list

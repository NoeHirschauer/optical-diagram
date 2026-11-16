API Reference
=============


This page contains auto-generated API reference documentation [#f1]_.

.. toctree::
   :titlesonly:
   :hidden:

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}

.. autoapisummary::

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.id }}
   {% endfor %}

.. [#f1] Created with `sphinx-autoapi <https://github.com/readthedocs/sphinx-autoapi>`_

.. raw:: html

    <div class="prename">{{ module }}.</div>
    <div class="empty"></div>

``{{ name }}``
{{ '=' * (name|length + 4) }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: {{ _('Attributes') }}
    
    .. autosummary::
    {% for item in attributes %}
        {%- if not item.startswith('_') %}
        ~{{ name }}.{{ item }}
        {%- endif -%}
    {%- endfor %}
    {% endif %}
    {% endblock %}
    
    {% block methods %}
    {%- if methods or inherited_members %}
    .. rubric:: {{ _('Methods') }}
    
    .. autosummary::
    {% for item in methods %}
        {%- if not item.startswith('_') and not item in attributes %}
        ~{{ name }}.{{ item }}
        {%- endif -%}
    {%- endfor %}
    
    {% for item in inherited_members %}
        {%- if not item.startswith('_') and not item in attributes %}
        ~{{ name }}.{{ item }}
        {%- endif -%}
    {%- endfor %}
    {% endif %}
    {% endblock %}

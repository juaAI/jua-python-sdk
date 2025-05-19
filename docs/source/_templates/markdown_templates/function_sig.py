"""
Custom function signature formatting for markdown output
"""

from sphinx.application import Sphinx


def patch_markdown_builder(app):
    """Patch the markdown builder's translator for function signature formatting"""
    if not hasattr(app, "builder") or app.builder.name != "markdown":
        return

    # Get the original visit_desc_parameterlist method
    from sphinx_markdown_builder.translator import MarkdownTranslator

    original_visit_desc_parameterlist = MarkdownTranslator.visit_desc_parameterlist
    original_depart_desc_parameterlist = MarkdownTranslator.depart_desc_parameterlist
    original_visit_desc_parameter = MarkdownTranslator.visit_desc_parameter
    original_depart_desc_parameter = MarkdownTranslator.depart_desc_parameter

    # Patch it to format long signatures with parameters on new lines
    def patched_visit_desc_parameterlist(self, node):
        # Check if the parameter list is long
        # (more than 3 parameters or has complex types)
        param_count = len(node.children)
        complex_types = any("|" in child.astext() for child in node.children)
        is_long_signature = param_count > 3 or complex_types

        if is_long_signature:
            # Start with opening parenthesis
            self.add("(", suffix_eol=1)
            # Indent parameters
            self._push_context(self._make_indent_context("    "))
        else:
            # Use the original method for short signatures
            original_visit_desc_parameterlist(self, node)

    def patched_depart_desc_parameterlist(self, node):
        param_count = len(node.children)
        complex_types = any("|" in child.astext() for child in node.children)
        is_long_signature = param_count > 3 or complex_types

        if is_long_signature:
            # Pop the indent context
            self._pop_context()
            # Add closing parenthesis
            self.add(")", prefix_eol=1)
        else:
            # Use the original method for short signatures
            original_depart_desc_parameterlist(self, node)

    def patched_visit_desc_parameter(self, node):
        param_count = len(node.parent.children)
        complex_types = any("|" in child.astext() for child in node.parent.children)
        is_long_signature = param_count > 3 or complex_types

        if is_long_signature:
            param_text = node.astext()
            # Add the parameter followed by a comma
            if node is not node.parent.children[-1]:  # If not the last parameter
                param_text += ","
            self.add(param_text, suffix_eol=1)
        else:
            # Use the original method for short signatures
            original_visit_desc_parameter(self, node)

    def patched_depart_desc_parameter(self, node):
        param_count = len(node.parent.children)
        complex_types = any("|" in child.astext() for child in node.parent.children)
        is_long_signature = param_count > 3 or complex_types

        if not is_long_signature:
            # Use the original method for short signatures
            original_depart_desc_parameter(self, node)

    # Add a helper method to create indent context if it doesn't exist
    if not hasattr(MarkdownTranslator, "_make_indent_context"):
        from sphinx_markdown_builder.contexts import IndentContext, SubContextParams

        def _make_indent_context(self, prefix):
            return IndentContext(
                prefix,
                only_first=False,
                support_multi_line_break=True,
                params=SubContextParams(0, 0),
            )

        MarkdownTranslator._make_indent_context = _make_indent_context

    # Apply the patches
    MarkdownTranslator.visit_desc_parameterlist = patched_visit_desc_parameterlist
    MarkdownTranslator.depart_desc_parameterlist = patched_depart_desc_parameterlist
    MarkdownTranslator.visit_desc_parameter = patched_visit_desc_parameter
    MarkdownTranslator.depart_desc_parameter = patched_depart_desc_parameter


def setup(app: Sphinx):
    """Setup function for Sphinx extension"""
    app.connect("builder-inited", patch_markdown_builder)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

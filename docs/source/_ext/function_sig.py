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

    # Use a completely different approach for long signatures
    def patched_visit_desc_parameterlist(self, node):
        # Check if the parameter list is long
        # more than 3 parameters or has complex types)
        param_count = len(node.children)
        complex_types = any("|" in child.astext() for child in node.children)
        is_long_signature = param_count > 3 or complex_types

        # Store information in the node for later use
        node._is_long_signature = is_long_signature

        if is_long_signature:
            # For long signatures, we'll handle everything in depart_desc_parameterlist
            # Just collect parameter texts for now
            node._param_texts = []
            for param_node in node.children:
                node._param_texts.append(param_node.astext())

            # Just output the opening parenthesis for now
            self.add("(")
        else:
            # Use the original method for short signatures
            original_visit_desc_parameterlist(self, node)

    def patched_depart_desc_parameterlist(self, node):
        # Check if we marked this as a long signature
        if hasattr(node, "_is_long_signature") and node._is_long_signature:
            # Add HTML <br/> tag after the opening parenthesis
            self.add("<br/>")

            # Add each parameter on its own line with proper indentation
            # using HTML entities
            for i, param_text in enumerate(node._param_texts):
                # Use HTML non-breaking spaces for indentation (4 spaces)
                line = "&nbsp;&nbsp;&nbsp;&nbsp;" + param_text
                if i < len(node._param_texts) - 1:
                    line += ","
                # Add <br/> for line break
                self.add(line + "<br/>")

            # Add closing parenthesis
            self.add(")")
        else:
            # Use the original method for short signatures
            original_depart_desc_parameterlist(self, node)

    def patched_visit_desc_parameter(self, node):
        # Skip normal parameter processing for long signatures
        if (
            hasattr(node.parent, "_is_long_signature")
            and node.parent._is_long_signature
        ):
            # We've already collected and processed these in visit_desc_parameterlist
            from docutils import nodes

            raise nodes.SkipNode
        else:
            # Use the original method for short signatures
            original_visit_desc_parameter(self, node)

    # Add a helper method to visit desc_returns if needed
    original_visit_desc_returns = getattr(
        MarkdownTranslator, "visit_desc_returns", None
    )

    def patched_visit_desc_returns(self, node):
        # Ensure there's a space before the return arrow
        self.add(" ")
        if original_visit_desc_returns:
            original_visit_desc_returns(self, node)
        else:
            self.add("→ ")

    # Apply the patches
    MarkdownTranslator.visit_desc_parameterlist = patched_visit_desc_parameterlist
    MarkdownTranslator.depart_desc_parameterlist = patched_depart_desc_parameterlist
    MarkdownTranslator.visit_desc_parameter = patched_visit_desc_parameter

    if original_visit_desc_returns:
        MarkdownTranslator.visit_desc_returns = patched_visit_desc_returns


def setup(app: Sphinx):
    """Setup function for Sphinx extension"""
    app.connect("builder-inited", patch_markdown_builder)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

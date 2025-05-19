"""
Custom function signature formatting for markdown output
"""

from sphinx.application import Sphinx


def patch_markdown_builder(app):
    """Patch the markdown builder's translator for function signature formatting"""
    if not hasattr(app, "builder") or app.builder.name != "markdown":
        return

    # Get the original visit_desc_parameterlist method
    from docutils import nodes
    from sphinx_markdown_builder.translator import MarkdownTranslator

    original_visit_desc_parameterlist = MarkdownTranslator.visit_desc_parameterlist
    original_depart_desc_parameterlist = MarkdownTranslator.depart_desc_parameterlist
    original_visit_desc_parameter = MarkdownTranslator.visit_desc_parameter

    def process_node_with_references(self, node):
        """Process a node, preserving links in references"""
        result = ""

        if isinstance(node, nodes.reference) and "refuri" in node.attributes:
            # Handle references with links
            link_text = node.astext()
            refuri = node["refuri"]
            result += f"[{link_text}]({refuri})"
        elif isinstance(node, nodes.Text):
            # Plain text node
            result += node.astext()
        else:
            # Process this node's text
            if hasattr(node, "astext"):
                # Check if this node has children to process
                if node.children:
                    # Process all children and combine their results
                    for child in node.children:
                        result += process_node_with_references(self, child)
                else:
                    # No children, just use the text
                    result += node.astext()
            else:
                # Fallback for nodes without astext method
                result += str(node)

        return result

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
                # Process the parameter node with reference preservation
                param_text = process_node_with_references(self, param_node)
                node._param_texts.append(param_text)

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
            # Handle return type with references
            self.add("→ ")
            if node.children:
                return_text = process_node_with_references(self, node)
                self.add(return_text)

    # Apply the patches
    MarkdownTranslator.visit_desc_parameterlist = patched_visit_desc_parameterlist
    MarkdownTranslator.depart_desc_parameterlist = patched_depart_desc_parameterlist
    MarkdownTranslator.visit_desc_parameter = patched_visit_desc_parameter

    if original_visit_desc_returns:
        MarkdownTranslator.visit_desc_returns = patched_visit_desc_returns
    else:
        MarkdownTranslator.visit_desc_returns = patched_visit_desc_returns


def setup(app: Sphinx):
    """Setup function for Sphinx extension"""
    app.connect("builder-inited", patch_markdown_builder)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

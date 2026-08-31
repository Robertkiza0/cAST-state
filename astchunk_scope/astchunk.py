import tree_sitter as ts

from .preprocessing import ByteRange, get_nws_count_direct


class ASTChunk():
    """
    A chunk of code represented by a list of ASTNodes.

    This class provides additional information for each chunk, including:
        - chunk_text: rebuilt code text from the list of ASTNodes
        - chunk_size: size of the chunk (in non-whitespace characters)
        - chunk_ancestors: ancestors of the chunk (list of ancestor names)
        - metadata: additional metadata for the chunk (e.g., file path, class path, etc.)

    Attributes:
        - ast_window: list of ASTNode objects
        - max_chunk_size: maximum size for each AST chunk, using non-whitespace character count by default.
        - language: programming language
        - metadata_template: type of metadata to store (e.g., start/end line number, path to file, etc.)
    """
    def __init__(
        self, ast_window: list, max_chunk_size: int, language: str, metadata_template: str,
        ancestor_annotation_cache: dict | None = None,
    ):
        self.ast_window = ast_window
        self.max_chunk_size = max_chunk_size
        self.language = language
        self.metadata_template = metadata_template
        # Memoizes a class/function ancestor node's cAST-Scope header (id(node) ->
        # formatted string) across every ASTChunk built from the same tree, so a
        # class with many chunks nested in it doesn't re-walk its whole subtree
        # once per chunk. Caller (ASTChunkBuilder) shares one dict per chunkify()
        # call; falls back to a private one-off dict if used standalone.
        self._ancestor_annotation_cache = ancestor_annotation_cache if ancestor_annotation_cache is not None else {}
        assert len(self.ast_window) > 0, "Expect ASTChunk to be non-empty"

        self.chunk_text = self.rebuild_code(self.ast_window)
        self.chunk_size = get_nws_count_direct(self.chunk_text)

        # build chunk ancestors using the ancestors of the first ASTNode in the window
        self.chunk_ancestors = self.build_chunk_ancestors(self.ast_window[0].ancestors)

    @property
    def strcode(self):
        return self.chunk_text

    @property
    def brange(self):
        return ByteRange(self.ast_window[0].brange.start, self.ast_window[-1].brange.stop)

    @property
    def start_line(self):
        return self.ast_window[0].start_line

    @property
    def end_line(self):
        return self.ast_window[-1].end_line

    @property
    def size(self):
        """
        Define size as the number of non-whitespace characters.
        """
        return self.chunk_size

    @property
    def length(self):
        """
        Define length as the number of lines covered by the chunk.
        """
        return self.end_line - self.start_line + 1

    def rebuild_code(self, ast_window: list) -> str:
        """
        Rebuild source code from a list of ASTNodes.

        The code text stored in each ASTNode is inherited from the tree-sitter Node object, which omits
        leading and trailing spaces and newlines between nodes. Therefore, this function restores the
        original code by adding the necessary newlines and spaces.

        Args:
            ast_window: list of ASTNode objects

        Returns:
            Rebuilt source code string
        """
        if len(ast_window) == 0:
            return ""

        current_line, current_col = ast_window[0].start_line, ast_window[0].start_col
        code = " " * current_col

        for node in ast_window:
            # If we need to jump to a new line, add newline(s)
            if  node.start_line > current_line:
                # Add as many newlines as needed.
                code += "\n" * (node.start_line - current_line)
                current_line =  node.start_line
                # Reset the column since we are at a new line.
                current_col = 0
            # If we are on the correct line but need to add indentation spaces:
            if  node.start_col > current_col:
                code += " " * (node.start_col - current_col)
                current_col =  node.start_col
            # Append the node_text
            code += node.strcode
            # Update our cursor position to the given end coordinate.
            # (We trust that the given end coordinate is consistent with the node_text.)
            current_line, current_col =  node.end_line,  node.end_col

        return code

    # ------------------------------------------------------------------ #
    #   cAST-Scope: scope-aware ancestor extraction (Limitation #1 fix)  #
    # ------------------------------------------------------------------ #
    # The original build_chunk_ancestors() only kept the header's first
    # source line (`node.text.decode("utf8").split("\n")[0]`). That loses
    # two things a reader needs to understand a chunk nested deep in a
    # class/function without pulling in the whole enclosing body:
    #   1. A class's instance state (the self.* attributes it carries),
    #      which chunk_expansion is the only place cheap enough to surface
    #      it (recomputing it from the whole class body per chunk would be
    #      redundant work already paid for once here).
    #   2. A decorated function's decorators (@app.get(...), @staticmethod),
    #      which change what the signature *means* (e.g. a route vs a plain
    #      method) but live as siblings of function_definition under
    #      decorated_definition, not inside it — so a plain first-line split
    #      of the function node's own text silently drops them.

    @staticmethod
    def _self_attribute_name(assignment_target: ts.Node) -> str | None:
        """If assignment_target is (or wraps, via tuple/pattern-list unpacking)
        an attribute access `self.<name>`, return `<name>`; else None."""
        if assignment_target.type == "attribute":
            obj = assignment_target.child_by_field_name("object")
            attr = assignment_target.child_by_field_name("attribute")
            if obj is not None and attr is not None and obj.text == b"self":
                return attr.text.decode("utf8")
        return None

    def _extract_self_attributes(self, class_node: ts.Node) -> list[str]:
        """Collect every `self.<name>` assigned anywhere in class_node's body
        (across all its methods, including tuple-unpacking and augmented
        assignment), e.g. {config, db, logger}. Does not descend into a
        nested class_definition — that class's `self` refers to a different
        instance, so its attributes aren't this class's state."""
        attrs: set[str] = set()

        def walk(node: ts.Node) -> None:
            if node.type == "class_definition" and node is not class_node:
                return  # different instance's self — do not attribute its state to us
            if node.type in ("assignment", "augmented_assignment"):
                target = node.child_by_field_name("left")
                if target is not None:
                    if target.type in ("pattern_list", "tuple_pattern", "expression_list"):
                        for element in target.children:
                            name = self._self_attribute_name(element)
                            if name is not None:
                                attrs.add(name)
                    else:
                        name = self._self_attribute_name(target)
                        if name is not None:
                            attrs.add(name)
            for child in node.children:
                walk(child)

        walk(class_node)
        return sorted(attrs)

    @staticmethod
    def _decorator_lines(function_node: ts.Node) -> list[str]:
        """If function_node is wrapped in a decorated_definition (its parent
        in the tree — decorators are siblings of function_definition, not
        children of it), return its decorator source lines in order."""
        parent = function_node.parent
        if parent is None or parent.type != "decorated_definition":
            return []
        return [
            child.text.decode("utf8").strip()
            for child in parent.children
            if child.type == "decorator"
        ]

    def build_chunk_ancestors(self, node_ancestors: list) -> list[str]:
        '''
        Build the class/function path to the chunk. The path is built from the ancestors of the first
        ASTNode in the window. We only keep the ancestors that are class or function definitions.

        The intuition is that we want to record where the chunk is located in the AST. This can be useful
        for downstream tasks such as code retrieval (e.g., disambiguating between different functions with the same name).
        For each ancestor that is a class or function definition, we extract the first line in the ancestor's text.
        This simple heuristic is also commonly used in software patching tasks, such as generating GitHub issue fixes,
        where identifying the location of a change is an essential part of the patch.

        cAST-Scope extension: a class ancestor's header is annotated with the
        `self.*` instance state assigned anywhere in its body (e.g.
        "class DataProcessor: (State: self.config, self.db, self.logger)"),
        and a function ancestor's header is prefixed with its decorators if
        any (e.g. "@app.get('/x') def route(self):") — both computed directly
        from the already-parsed tree-sitter node, no re-parsing, so this adds
        no measurable overhead over the original.

        Args:
            node_ancestors: list of tree-sitter nodes that are ancestors of the first ASTNode in the window

        Returns:
            List of ancestors that are class or function definitions
        '''
        chunk_ancestors = []

        for node in node_ancestors:
            cache_key = id(node)
            cached = self._ancestor_annotation_cache.get(cache_key)
            if cached is not None:
                chunk_ancestors.append(cached)
                continue

            if node.type == "class_definition":
                header = node.text.decode("utf8").split("\n")[0].rstrip()
                state_attrs = self._extract_self_attributes(node)
                if state_attrs:
                    header += f" (State: {', '.join(f'self.{name}' for name in state_attrs)})"
            elif node.type == "function_definition":
                header = node.text.decode("utf8").split("\n")[0].rstrip()
                decorators = self._decorator_lines(node)
                if decorators:
                    header = " ".join(decorators) + " " + header
            else:
                continue

            self._ancestor_annotation_cache[cache_key] = header
            chunk_ancestors.append(header)

        return chunk_ancestors

    def build_metadata(self, repo_level_metadata: dict):
        """
        Build metadata for the chunk.

        Args:
            repo_level_metadata: repository-level metadata (e.g., repo name, file path)
        """
        if self.metadata_template == "none":
            self.metadata = {}
        elif self.metadata_template == "default":
            filepath = repo_level_metadata.get("filepath", "")
            self.metadata = {
                "filepath": filepath,
                "chunk_size": self.chunk_size,
                "line_count": self.length,
                "start_line_no": self.start_line,
                "end_line_no": self.end_line,
                "node_count": len(self.ast_window),
            }
        elif self.metadata_template == "coderagbench-repoeval":
            fpath_tuple = repo_level_metadata.get("fpath_tuple", [])
            repo = repo_level_metadata.get("repo", "")
            self.metadata = {
                "fpath_tuple": fpath_tuple,
                "repo": repo,
                "chunk_size": self.chunk_size,
                "line_count": self.length,
                "start_line_no": self.start_line,
                "end_line_no": self.end_line,
                "node_count": len(self.ast_window),
            }
        elif self.metadata_template == "coderagbench-swebench-lite":
            instance_id = repo_level_metadata.get("instance_id", "")
            filename = repo_level_metadata.get("filename", "")
            self.metadata = {
                "_id": f"{instance_id}_{self.start_line}-{self.end_line}",
                "title": filename,
            }
        else:
            raise ValueError(f"Unsupported Metadata Template Name: {self.metadata_template}!")

    def apply_chunk_expansion(self):
        """
        Apply chunk expansion to the chunk. Chunk expansion is the process of adding chunk expansion metadata
        (e.g., file path, class path) to the beginning of each chunk.
        """
        self.chunk_expansion_metadata = {
            "filepath": "",
            "ancestors": "\n".join(["\t" * i + ancestor for i, ancestor in enumerate(self.chunk_ancestors)]),
        }
        if self.metadata_template == "default":
            self.chunk_expansion_metadata["filepath"] = self.metadata["filepath"]
        elif self.metadata_template == "coderagbench-repoeval":
            self.chunk_expansion_metadata["filepath"] = "/".join(self.metadata["fpath_tuple"])
        elif self.metadata_template == "coderagbench-swebench-lite":
            self.chunk_expansion_metadata["filepath"] = self.metadata["title"]

        chunk_expansion = "'''\n"
        chunk_expansion += f"{self.chunk_expansion_metadata['filepath']}\n" if self.chunk_expansion_metadata["filepath"] else ""
        chunk_expansion += f"{self.chunk_expansion_metadata['ancestors']}\n" if self.chunk_expansion_metadata["ancestors"] else ""
        chunk_expansion += "'''"

        self.chunk_text = f"{chunk_expansion}\n{self.chunk_text}"

    def to_code_window(self) -> dict:
        """
        Convert the ASTChunk object into a code window for downstream integration.
        """
        if self.metadata_template == "coderagbench-swebench-lite":
            code_window = {
                "_id": self.metadata["_id"],
                "title": self.metadata['title'],
                "text": self.chunk_text
            }
        else:
            code_window = {
                "content": self.chunk_text,
                "metadata": self.metadata
            }

        return code_window

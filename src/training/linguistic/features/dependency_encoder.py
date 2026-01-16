"""
Dependency feature encoder.

Encodes dependency parse trees as feature vectors for MT models.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import logging
import numpy as np

from src.training.linguistic.parsers.base import ParseTree

logger = logging.getLogger(__name__)


@dataclass
class DependencyFeatures:
    """
    Extracted features from dependency parse.

    Attributes:
        deprel_ids: Dependency relation IDs
        pos_ids: POS tag IDs
        tree_depths: Depth in tree for each word
        head_distances: Distance to head for each word
        is_root: Whether word is root
        num_children: Number of children for each word
    """
    deprel_ids: List[int]
    pos_ids: List[int]
    tree_depths: List[int]
    head_distances: List[int]
    is_root: List[bool]
    num_children: List[int]


class DependencyEncoder:
    """
    Encode dependency parse trees as feature vectors.

    Supports multiple feature types and optional GNN encoding.
    """

    # Common UD dependency relations
    UD_DEPRELS = [
        "root", "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp",
        "obl", "vocative", "expl", "dislocated", "advcl", "advmod",
        "discourse", "aux", "cop", "mark", "nmod", "appos", "nummod",
        "acl", "amod", "det", "clf", "case", "conj", "cc", "fixed",
        "flat", "compound", "list", "parataxis", "orphan", "goeswith",
        "reparandum", "punct", "dep"
    ]

    # Common UD POS tags
    UD_POS = [
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
        "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
        "VERB", "X"
    ]

    def __init__(
        self,
        encoding_dim: int = 128,
        use_graph_encoder: bool = False,
        feature_types: Optional[List[str]] = None,
    ):
        """
        Initialize dependency encoder.

        Args:
            encoding_dim: Dimension of encoded features
            use_graph_encoder: Whether to use GNN for encoding
            feature_types: Types of features to extract
        """
        self.encoding_dim = encoding_dim
        self.use_graph_encoder = use_graph_encoder

        if feature_types is None:
            self.feature_types = [
                "dependency_labels",
                "pos_tags",
                "tree_depth",
                "head_distance",
            ]
        else:
            self.feature_types = feature_types

        # Build vocabularies
        self.deprel_vocab = {rel: i for i, rel in enumerate(self.UD_DEPRELS)}
        self.pos_vocab = {pos: i for i, pos in enumerate(self.UD_POS)}

        # Feature dimensions
        self.deprel_dim = len(self.UD_DEPRELS)
        self.pos_dim = len(self.UD_POS)

        logger.info(
            f"DependencyEncoder initialized: dim={encoding_dim}, "
            f"features={self.feature_types}, gnn={use_graph_encoder}"
        )

    def extract_features(self, parse: ParseTree) -> DependencyFeatures:
        """
        Extract features from parse tree.

        Args:
            parse: Parse tree

        Returns:
            DependencyFeatures object
        """
        n = len(parse.words)

        # Dependency relation IDs
        deprel_ids = [
            self.deprel_vocab.get(rel, len(self.deprel_vocab))
            for rel in parse.deprels
        ]

        # POS tag IDs
        pos_ids = [
            self.pos_vocab.get(pos, len(self.pos_vocab))
            for pos in parse.pos_tags
        ]

        # Tree depths
        tree_depths = [parse.get_tree_depth(i) for i in range(n)]

        # Head distances
        head_distances = [
            abs(parse.heads[i] - (i + 1)) if parse.heads[i] != 0 else 0
            for i in range(n)
        ]

        # Is root
        is_root = [head == 0 for head in parse.heads]

        # Number of children
        num_children = [len(parse.get_children(i)) for i in range(n)]

        return DependencyFeatures(
            deprel_ids=deprel_ids,
            pos_ids=pos_ids,
            tree_depths=tree_depths,
            head_distances=head_distances,
            is_root=is_root,
            num_children=num_children,
        )

    def encode_parse_tree(
        self,
        parse: ParseTree,
        return_tensor: bool = True
    ):
        """
        Encode parse tree as feature vectors.

        Args:
            parse: Parse tree
            return_tensor: Whether to return torch.Tensor (else numpy array)

        Returns:
            Feature vectors of shape [seq_len, encoding_dim]
        """
        # Extract features
        features = self.extract_features(parse)

        # Build feature matrix
        feature_vectors = []

        for i in range(len(parse.words)):
            vec = []

            # Dependency relation (one-hot)
            if "dependency_labels" in self.feature_types:
                deprel_onehot = np.zeros(self.deprel_dim + 1)  # +1 for UNK
                deprel_onehot[features.deprel_ids[i]] = 1.0
                vec.extend(deprel_onehot)

            # POS tag (one-hot)
            if "pos_tags" in self.feature_types:
                pos_onehot = np.zeros(self.pos_dim + 1)  # +1 for UNK
                pos_onehot[features.pos_ids[i]] = 1.0
                vec.extend(pos_onehot)

            # Tree depth (normalized)
            if "tree_depth" in self.feature_types:
                max_depth = max(features.tree_depths)
                depth_norm = features.tree_depths[i] / (max_depth + 1e-8)
                vec.append(depth_norm)

            # Head distance (normalized)
            if "head_distance" in self.feature_types:
                max_dist = max(features.head_distances)
                dist_norm = features.head_distances[i] / (max_dist + 1e-8)
                vec.append(dist_norm)

            # Is root (binary)
            if "is_root" in self.feature_types:
                vec.append(1.0 if features.is_root[i] else 0.0)

            # Number of children (normalized)
            if "num_children" in self.feature_types:
                max_children = max(features.num_children)
                children_norm = features.num_children[i] / (max_children + 1e-8)
                vec.append(children_norm)

            feature_vectors.append(np.array(vec))

        feature_matrix = np.stack(feature_vectors)

        # Project to encoding_dim if needed
        if feature_matrix.shape[1] != self.encoding_dim:
            # Simple linear projection (in practice, use learnable projection)
            # For now, pad or truncate
            if feature_matrix.shape[1] < self.encoding_dim:
                # Pad with zeros
                padding = np.zeros((
                    feature_matrix.shape[0],
                    self.encoding_dim - feature_matrix.shape[1]
                ))
                feature_matrix = np.concatenate([feature_matrix, padding], axis=1)
            else:
                # Truncate
                feature_matrix = feature_matrix[:, :self.encoding_dim]

        # Optionally use graph encoder
        if self.use_graph_encoder:
            feature_matrix = self._apply_graph_encoder(feature_matrix, parse)

        # Convert to tensor if requested
        if return_tensor:
            try:
                import torch
                return torch.from_numpy(feature_matrix).float()
            except ImportError:
                logger.warning("PyTorch not available, returning numpy array")
                return feature_matrix

        return feature_matrix

    def _apply_graph_encoder(
        self,
        features: np.ndarray,
        parse: ParseTree
    ) -> np.ndarray:
        """
        Apply graph neural network encoder.

        Args:
            features: Initial feature matrix [seq_len, dim]
            parse: Parse tree (for graph structure)

        Returns:
            Encoded features [seq_len, dim]
        """
        # TODO: Implement GNN encoder (GCN, GAT, etc.)
        # For now, return features as-is
        logger.debug("GNN encoding not yet implemented, returning features")
        return features

    def encode_batch(
        self,
        parses: List[ParseTree],
        return_tensor: bool = True
    ):
        """
        Encode batch of parse trees.

        Args:
            parses: List of parse trees
            return_tensor: Whether to return torch.Tensor

        Returns:
            Batch of feature vectors (list or padded tensor)
        """
        encoded = [
            self.encode_parse_tree(parse, return_tensor=False)
            for parse in parses
        ]

        if return_tensor:
            try:
                import torch
                from torch.nn.utils.rnn import pad_sequence

                # Convert to tensors
                tensors = [torch.from_numpy(e).float() for e in encoded]

                # Pad to same length
                padded = pad_sequence(tensors, batch_first=True, padding_value=0.0)

                return padded
            except ImportError:
                logger.warning("PyTorch not available, returning list")
                return encoded

        return encoded


def compute_parse_similarity(parse1: ParseTree, parse2: ParseTree) -> float:
    """
    Compute similarity between two parse trees.

    Uses tree edit distance and label similarity.

    Args:
        parse1: First parse tree
        parse2: Second parse tree

    Returns:
        Similarity score in [0, 1]
    """
    # Simple similarity based on shared dependency relations
    deprels1 = set(parse1.deprels)
    deprels2 = set(parse2.deprels)

    if not deprels1 and not deprels2:
        return 1.0

    intersection = len(deprels1 & deprels2)
    union = len(deprels1 | deprels2)

    return intersection / union if union > 0 else 0.0

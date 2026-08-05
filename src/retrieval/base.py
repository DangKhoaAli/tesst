"""
Abstract Base Retriever for AIC Video Retrieval System.

All single-domain retrievers (visual, text, ocr, object) implement this interface,
enabling the fusion layer to combine results uniformly via RRF or weighted sum.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.common.types import SearchResult


class BaseRetriever(ABC):
    """
    Interface that every single-domain retriever must implement.

    A retriever takes a query (text, vector, or structured) and returns
    a ranked list of SearchResult objects. The fusion layer then combines
    results from multiple retrievers.
    """

    @abstractmethod
    def retrieve(self, query: object, top_k: int = 100) -> List[SearchResult]:
        """
        Execute the search and return top_k results.

        Args:
            query:  Domain-specific query object (text string, numpy vector, etc.)
            top_k:  Maximum number of results to return

        Returns:
            List of SearchResult sorted by score descending.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in SearchResult.retriever_source."""
        ...

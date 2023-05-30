from __future__ import annotations

import asyncio
from typing import Any, Optional

import numpy as np

from raffle_ds_research.tools.index_tools import retrieval_data_type as rtypes
from raffle_ds_research.tools.index_tools import search_server


class MultiSearchClient(search_server.SearchClient):
    """A client to interact with a search server."""

    clients: dict[str, search_server.SearchClient]

    def __init__(self, clients: dict[str, search_server.SearchClient]) -> None:
        self.clients = clients

    @property
    def requires_vectors(self) -> bool:
        """Whether the client requires vectors to be passed to the search method."""
        return any(client.requires_vectors for client in self.clients.values())

    def ping(self) -> bool:
        """Ping the server."""
        return all(client.ping() for client in self.clients.values())

    def search(
        self,
        *,
        text: list[str],
        vector: Optional[np.ndarray] = None,
        label: Optional[list[str | int]] = None,
        top_k: int = 3,
    ) -> dict[str, rtypes.RetrievalBatch[np.ndarray]]:
        """Search the server given a batch of text and/or vectors."""
        return {
            name: client.search(
                vector=vector,
                text=text,
                label=label,
                top_k=top_k,
            )
            for name, client in self.clients.items()
        }

    async def async_search(
        self,
        *,
        text: list[str],
        vector: Optional[rtypes.Ts] = None,
        label: Optional[list[str | int]] = None,
        top_k: int = 3,
    ) -> dict[str, rtypes.RetrievalBatch[np.ndarray]]:
        """Search the server given a batch of text and/or vectors."""

        def search_fn(args: dict[str, Any]) -> rtypes.RetrievalBatch[np.ndarray]:
            client = args.pop("client")
            return client.search(**args)

        names = list(self.clients.keys())
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                None,
                search_fn,
                {
                    "client": self.clients[name],
                    "vector": vector,
                    "text": text,
                    "label": label,
                    "top_k": top_k,
                },
            )
            for name in names
        ]

        results = await asyncio.gather(*futures)
        return dict(zip(names, results))


class MultiSearchMaster:
    """Handle multiple search servers."""

    servers: dict[str, search_server.SearchMaster]

    def __init__(self, servers: dict[str, search_server.SearchMaster], skip_setup: bool = False):
        """Initialize the search master."""
        self.skip_setup = skip_setup
        self.servers = servers

    def __enter__(self) -> MultiSearchMaster:
        """Start the servers."""
        for server in self.servers.values():
            server.__enter__()

        return self

    def __exit__(self, *args, **kwargs) -> None:  # noqa: ANN, ARG
        """Stop the servers."""
        for server in self.servers.values():
            server.__exit__(*args, **kwargs)

    def get_client(self) -> MultiSearchClient:
        """Get the client for interacting with the Faiss server."""
        return MultiSearchClient(clients={name: server.get_client() for name, server in self.servers.items()})
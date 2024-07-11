import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any
import os

class RAGManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def save_info(self, content: str, tags: List[str]) -> str:
        raise NotImplementedError("This method should be implemented by subclasses")

    def retrieve_info(self, query: str, tags: List[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError("This method should be implemented by subclasses")

class ChromaDBManager(RAGManager):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.db_path = config.get('vector_db', {}).get('path', './chroma_db')
        self.client = chromadb.Client(Settings(persist_directory=self.db_path))
        self.collection = self.client.get_or_create_collection("default_collection")

    def save_info(self, content: str, tags: List[str]) -> str:
        doc_id = str(self.collection.count() + 1)
        self.collection.add(
            documents=[content],
            metadatas=[{"tags": tags}],
            ids=[doc_id]
        )
        return doc_id

    def retrieve_info(self, query: str, tags: List[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        filter_dict = {"tags": {"$in": tags}} if tags else None
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_dict
        )

        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
            formatted_results.append({
                "content": doc,
                "tags": metadata['tags'],
                "distance": distance
            })

        return formatted_results

class RAGSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_manager = self._initialize_rag_manager()

    def _initialize_rag_manager(self):
        custom_manager_path = self.config.get('custom_rag_manager')
        if custom_manager_path:
            try:
                module_name, class_name = custom_manager_path.rsplit('.', 1)
                module = __import__(module_name, fromlist=[class_name])
                CustomManagerClass = getattr(module, class_name)
                return CustomManagerClass(self.config)
            except (ImportError, AttributeError) as e:
                print(f"Error loading custom RAG manager: {e}. Falling back to default ChromaDBManager.")

        return ChromaDBManager(self.config)

    def save_info(self, content: str, tags: List[str]) -> str:
        return self.rag_manager.save_info(content, tags)

    def retrieve_info(self, query: str, tags: List[str] = None) -> List[Dict[str, Any]]:
        return self.rag_manager.retrieve_info(query, tags)
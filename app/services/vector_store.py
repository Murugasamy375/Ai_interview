import uuid
import datetime
import logging
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config

logger = logging.getLogger("app.services.vector_store")


class SentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    Lazy-loaded SentenceTransformer embedding function.

    The model is NOT loaded when FastAPI starts.
    It is loaded only when Chroma actually needs embeddings.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info(
                "Loading embedding model: %s",
                self.model_name
            )

            self.model = SentenceTransformer(
                self.model_name,
                device="cpu"
            )

            logger.info("Embedding model loaded successfully")

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        self._load_model()

        embeddings = self.model.encode(
            input,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embeddings.tolist()


class VectorStoreService:
    """
    ChromaDB service with lazy embedding-model loading.
    """

    def __init__(self):
        logger.info("Initializing ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIR
        )

        # IMPORTANT:
        # SentenceTransformer is NOT loaded here.
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            config.EMBEDDING_MODEL_NAME
        )

        self.resume_collection = self.client.get_or_create_collection(
            name="resumes",
            embedding_function=self.embedding_fn
        )

        self.jd_collection = self.client.get_or_create_collection(
            name="job_descriptions",
            embedding_function=self.embedding_fn
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

        logger.info("ChromaDB initialized successfully")

    def add_resume(
        self,
        filename: str,
        text: str
    ) -> Dict[str, Any]:

        return self._add_document(
            self.resume_collection,
            "resume",
            filename,
            text
        )

    def add_jd(
        self,
        filename: str,
        text: str
    ) -> Dict[str, Any]:

        return self._add_document(
            self.jd_collection,
            "jd",
            filename,
            text
        )

    def _add_document(
        self,
        collection,
        doc_type: str,
        filename: str,
        text: str
    ) -> Dict[str, Any]:

        chunks = self.splitter.split_text(text)

        if not chunks:
            return {
                "filename": filename,
                "chunks_added": 0
            }

        doc_id = str(uuid.uuid4())

        timestamp = datetime.datetime.utcnow().isoformat()

        ids = [
            f"{doc_id}_chunk_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "document_id": doc_id,
                "document_name": filename,
                "type": doc_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "upload_timestamp": timestamp
            }
            for i in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas
        )

        return {
            "document_id": doc_id,
            "filename": filename,
            "type": doc_type,
            "chunks_added": len(chunks),
            "timestamp": timestamp
        }

    def list_documents(self) -> Dict[str, List[Dict[str, Any]]]:

        return {
            "resumes": self._get_collection_info(
                self.resume_collection
            ),
            "job_descriptions": self._get_collection_info(
                self.jd_collection
            )
        }

    def _get_collection_info(
        self,
        collection
    ) -> List[Dict[str, Any]]:

        results = collection.get()

        if not results or not results["ids"]:
            return []

        docs = {}

        for i, doc_id in enumerate(results["ids"]):

            metadata = results["metadatas"][i]

            document_id = metadata.get("document_id")

            if document_id not in docs:

                document = results["documents"][i]

                docs[document_id] = {
                    "document_id": document_id,
                    "filename": metadata.get("document_name"),
                    "type": metadata.get("type"),
                    "chunks_count": metadata.get(
                        "total_chunks",
                        0
                    ),
                    "upload_timestamp": metadata.get(
                        "upload_timestamp"
                    ),
                    "sample_text": (
                        document[:100] + "..."
                        if document
                        else ""
                    )
                }

        return list(docs.values())

    def get_document_chunks(
        self,
        collection_name: str,
        doc_identifier: str
    ) -> List[Dict[str, Any]]:

        collection = (
            self.resume_collection
            if collection_name == "resumes"
            else self.jd_collection
        )

        results = collection.get(
            include=[
                "documents",
                "metadatas",
                "embeddings"
            ]
        )

        if not results or not results["ids"]:
            return []

        docs = {}

        for i, chunk_id in enumerate(results["ids"]):

            metadata = results["metadatas"][i]

            document_id = metadata.get("document_id")

            filename = metadata.get("document_name")

            if not document_id:
                continue

            if document_id not in docs:

                docs[document_id] = {
                    "document_id": document_id,
                    "filename": filename,
                    "upload_timestamp": metadata.get(
                        "upload_timestamp",
                        ""
                    ),
                    "chunks": []
                }

            docs[document_id]["chunks"].append({
                "chunk_id": chunk_id,
                "text": results["documents"][i],
                "embedding": results["embeddings"][i],
                "chunk_index": metadata.get(
                    "chunk_index",
                    0
                )
            })

        for doc in docs.values():

            doc["chunks"].sort(
                key=lambda x: x["chunk_index"]
            )

        sorted_docs = sorted(
            docs.values(),
            key=lambda x: x.get(
                "upload_timestamp",
                ""
            )
        )

        # UUID
        if doc_identifier in docs:
            return docs[doc_identifier]["chunks"]

        # Filename
        for doc in sorted_docs:

            if (
                doc["filename"]
                and doc_identifier.lower()
                in doc["filename"].lower()
            ):
                return doc["chunks"]

        # Numeric index
        try:

            val = int(doc_identifier)

            if 0 <= val < len(sorted_docs):
                return sorted_docs[val]["chunks"]

            if 1 <= val <= len(sorted_docs):
                return sorted_docs[val - 1]["chunks"]

        except ValueError:
            pass

        return []

    def reset_db(self):

        try:
            self.client.delete_collection("resumes")
        except Exception:
            pass

        try:
            self.client.delete_collection("job_descriptions")
        except Exception:
            pass

        self.resume_collection = (
            self.client.get_or_create_collection(
                name="resumes",
                embedding_function=self.embedding_fn
            )
        )

        self.jd_collection = (
            self.client.get_or_create_collection(
                name="job_descriptions",
                embedding_function=self.embedding_fn
            )
        )

        logger.info("Vector database reset successfully")
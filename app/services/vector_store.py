import uuid
import datetime
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app import config

class SentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom embedding function wrapper for Chroma DB using SentenceTransformers."""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

class VectorStoreService:
    """Service to handle document chunking, embedding, and storage in Chroma DB."""
    def __init__(self):
        # Initialize chroma persistent client
        self.client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        
        # Initialize embedding function
        self.embedding_fn = SentenceTransformerEmbeddingFunction(config.EMBEDDING_MODEL_NAME)
        
        # Initialize collections
        self.resume_collection = self.client.get_or_create_collection(
            name="resumes",
            embedding_function=self.embedding_fn
        )
        self.jd_collection = self.client.get_or_create_collection(
            name="job_descriptions",
            embedding_function=self.embedding_fn
        )
        
        # Initialize text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

    def add_resume(self, filename: str, text: str) -> Dict[str, Any]:
        """Parse, chunk, embed, and store a resume in the resume collection."""
        return self._add_document(self.resume_collection, "resume", filename, text)

    def add_jd(self, filename: str, text: str) -> Dict[str, Any]:
        """Parse, chunk, embed, and store a job description in the jd collection."""
        return self._add_document(self.jd_collection, "jd", filename, text)
    
    def _add_document(self, collection, doc_type: str, filename: str, text: str) -> Dict[str, Any]:
        # Split text into chunks
        chunks = self.splitter.split_text(text)
        
        if not chunks:
            return {"filename": filename, "chunks_added": 0}
            
        doc_id = str(uuid.uuid4())
        timestamp = datetime.datetime.utcnow().isoformat()
        
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
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
        """List all documents currently stored in the DB collections."""
        resumes_info = self._get_collection_info(self.resume_collection)
        jds_info = self._get_collection_info(self.jd_collection)
        return {
            "resumes": resumes_info,
            "job_descriptions": jds_info
        }

    def _get_collection_info(self, collection) -> List[Dict[str, Any]]:
        results = collection.get()
        if not results or not results["ids"]:
            return []
            
        docs = {}
        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            document_id = metadata.get("document_id")
            if document_id not in docs:
                docs[document_id] = {
                    "document_id": document_id,
                    "filename": metadata.get("document_name"),
                    "type": metadata.get("type"),
                    "chunks_count": metadata.get("total_chunks", 0),
                    "upload_timestamp": metadata.get("upload_timestamp"),
                    "sample_text": results["documents"][i][:100] + "..." if results["documents"][i] else ""
                }
        return list(docs.values())

    def get_document_chunks(self, collection_name: str, doc_identifier: str) -> List[Dict[str, Any]]:
        """
        Retrieve chunks, metadata, and embeddings for a document in the collection.
        Resolves identifier by checking:
        1. Exact document_id (UUID)
        2. Filename (case-insensitive substring)
        3. Integer index (0-based or 1-based upload order)
        """
        collection = self.resume_collection if collection_name == "resumes" else self.jd_collection
        
        # Get all entries with their embeddings and metadatas
        results = collection.get(include=["documents", "metadatas", "embeddings"])
        if not results or not results["ids"]:
            return []
            
        # Group chunks by document_id
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
                    "upload_timestamp": metadata.get("upload_timestamp", ""),
                    "chunks": []
                }
                
            docs[document_id]["chunks"].append({
                "chunk_id": chunk_id,
                "text": results["documents"][i],
                "embedding": results["embeddings"][i],
                "chunk_index": metadata.get("chunk_index", 0)
            })
            
        # Sort chunks by their chunk_index
        for doc in docs.values():
            doc["chunks"].sort(key=lambda x: x["chunk_index"])
            
        # Sort documents by upload timestamp to guarantee stable indexing
        sorted_docs = sorted(docs.values(), key=lambda x: x.get("upload_timestamp", ""))
        
        # 1. Exact UUID Match
        if doc_identifier in docs:
            return docs[doc_identifier]["chunks"]
            
        # 2. Substring Filename Match (case-insensitive)
        for doc in sorted_docs:
            if doc["filename"] and doc_identifier.lower() in doc["filename"].lower():
                return doc["chunks"]
                
        # 3. Numeric Index Match (0-based or 1-based)
        try:
            val = int(doc_identifier)
            # Try 0-based first
            if 0 <= val < len(sorted_docs):
                return sorted_docs[val]["chunks"]
            # Try 1-based
            elif 1 <= val <= len(sorted_docs):
                return sorted_docs[val - 1]["chunks"]
        except ValueError:
            pass
            
        return []

    def reset_db(self):
        """Reset (delete and recreate) the resume and job description collections."""
        try:
            self.client.delete_collection("resumes")
        except Exception:
            pass
        try:
            self.client.delete_collection("job_descriptions")
        except Exception:
            pass
            
        self.resume_collection = self.client.get_or_create_collection(
            name="resumes",
            embedding_function=self.embedding_fn
        )
        self.jd_collection = self.client.get_or_create_collection(
            name="job_descriptions",
            embedding_function=self.embedding_fn
        )

"""ChromaDB storage utility for YouTube transcript chunks."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from datetime import datetime
from tabulate import tabulate  # For pretty printing tables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('chroma_storage.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChromaStorage:
    """Handles storage and retrieval of transcript chunks in ChromaDB."""
    
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "youtube_transcripts",
        collection_metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize ChromaDB storage.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
            collection_metadata: Optional metadata for the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata or {
            "hnsw:space": "cosine",
            "description": "YouTube transcript chunks with channel and date filtering"
        }
        
        # Initialize ChromaDB client with settings
        logger.info(f"Initializing ChromaDB client with persist directory: {persist_directory}")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one."""
        try:
            # First try to get the collection
            try:
                collection = self.client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Retrieved existing collection '{self.collection_name}'")
                return collection
            except ValueError:
                # Collection doesn't exist, create it
                collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata=self.collection_metadata
                )
                logger.info(f"Created new collection '{self.collection_name}'")
                return collection
        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}")
            raise

    def store_processed_file(self, file_path: Path) -> Dict[str, int]:
        """Store chunks from a processed JSON file into ChromaDB.
        
        Args:
            file_path: Path to the processed JSON file
            
        Returns:
            Dict with counts of added and skipped items
        """
        logger.info(f"Processing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return {"added": 0, "skipped": 0, "errors": 1}
            
        # Get existing IDs to avoid duplicates
        existing_ids = set(self.collection.get(include=[])["ids"])
        
        # Prepare batches for ChromaDB
        documents_batch = []
        embeddings_batch = []
        metadatas_batch = []
        ids_batch = []
        
        # Get metadata from the file
        file_metadata = data.get("meta_data", {})
        
        # Process chunks
        items_to_process = data.get("chunks", [])
        if not items_to_process and isinstance(data, list):
            items_to_process = data
        elif not items_to_process and isinstance(data, dict) and all(k in data for k in ["id", "chunk", "embeddings", "metadata"]):
            items_to_process = [data]
            
        if not items_to_process:
            logger.warning(f"No valid chunks found in {file_path}")
            return {"added": 0, "skipped": 0, "errors": 0}
            
        added_count = 0
        skipped_count = 0
        error_count = 0
        
        for item in items_to_process:
            try:
                chunk_id = item.get("id")
                chunk_text = item.get("chunk")
                embedding = item.get("embeddings")
                
                # Create metadata with new structure
                metadata = {
                    "channel_id": file_metadata.get("channel_id", "unknown"),
                    "video_id": file_metadata.get("video_id", "unknown"),
                    "published_at": file_metadata.get("published", file_metadata.get("upload_date", "unknown")),
                    "channel_name": file_metadata.get("channel_name", "unknown"),
                    "video_title": file_metadata.get("video_title", file_metadata.get("title", "unknown")),
                    "chunk_index": item.get("chunk_index", -1)
                }
                
                # Validate required fields
                if not all([chunk_id, chunk_text, embedding]):
                    logger.warning(f"Skipping item due to missing required fields in {file_path}")
                    error_count += 1
                    continue
                    
                # Skip if already exists
                if chunk_id in existing_ids:
                    logger.debug(f"Chunk {chunk_id} already exists, skipping")
                    skipped_count += 1
                    continue
                    
                # Add to batches
                documents_batch.append(chunk_text)
                embeddings_batch.append(embedding)
                metadatas_batch.append(metadata)
                ids_batch.append(chunk_id)
                existing_ids.add(chunk_id)
                
            except Exception as e:
                logger.error(f"Error processing chunk in {file_path}: {e}")
                error_count += 1
                continue
                
        # Add batches to ChromaDB if any
        if documents_batch:
            try:
                self.collection.add(
                    ids=ids_batch,
                    embeddings=embeddings_batch,
                    documents=documents_batch,
                    metadatas=metadatas_batch
                )
                added_count = len(ids_batch)
                logger.info(f"Added {added_count} new chunks from {file_path}")
            except Exception as e:
                logger.error(f"Failed to add chunks to ChromaDB: {e}")
                error_count += len(ids_batch)
                added_count = 0
                
        return {
            "added": added_count,
            "skipped": skipped_count,
            "errors": error_count
        }
        
    def _ensure_metadata_fields(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure metadata has all required fields with defaults if missing."""
        required_fields = {
            "video_id": "unknown_video",
            "title": "unknown_title",
            "channel_name": "unknown_channel",
            "upload_date": "unknown_date",
            "chunk_index": -1
        }
        
        for field, default in required_fields.items():
            if field not in metadata:
                metadata[field] = default
                
        return metadata
        
    def store_directory(self, directory_path: Path) -> Dict[str, int]:
        """Store all processed JSON files from a directory.
        
        Args:
            directory_path: Path to directory containing processed JSON files
            
        Returns:
            Dict with total counts of added, skipped, and error items
        """
        logger.info(f"Processing directory: {directory_path}")
        
        total_stats = {"added": 0, "skipped": 0, "errors": 0}
        
        for file_path in directory_path.glob("*.json"):
            stats = self.store_processed_file(file_path)
            for key in total_stats:
                total_stats[key] += stats[key]
                
        logger.info(f"Directory processing complete. Stats: {total_stats}")
        return total_stats
        
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection."""
        try:
            count = self.collection.count()
            # Get unique video IDs and channel names
            all_metadata = self.collection.get(include=["metadatas"])["metadatas"]
            unique_videos = len(set(m.get("video_id") for m in all_metadata))
            unique_channels = len(set(m.get("channel_name") for m in all_metadata))
            
            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "unique_videos": unique_videos,
                "unique_channels": unique_channels,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "error": str(e)
            }

    def list_all_documents(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """List all documents in the collection with their metadata.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of dictionaries containing document data
        """
        try:
            results = self.collection.get(
                limit=limit,
                offset=offset,
                include=["documents", "metadatas", "embeddings"]
            )
            
            documents = []
            for i in range(len(results["ids"])):
                doc = {
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "embedding_length": len(results["embeddings"][i])
                }
                documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    def search_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search documents by metadata filters.
        
        Args:
            metadata_filter: Dictionary of metadata fields to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents with their metadata
        """
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=limit
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"])):
                formatted_results.append({
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i]
                })
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search by metadata: {e}")
            return []

    def get_video_chunks(self, video_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific video.
        
        Args:
            video_id: The video ID to search for
            
        Returns:
            List of chunks from the specified video
        """
        return self.search_by_metadata({"video_id": video_id})

    def get_channel_chunks(
        self,
        channel_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get chunks from a specific channel within a date range.
        
        Args:
            channel_id: Channel ID to filter by
            start_date: Optional start date (Unix timestamp)
            end_date: Optional end date (Unix timestamp)
            limit: Maximum number of results to return
            
        Returns:
            List of matching chunks with their metadata
        """
        where = {"channel_id": channel_id}
        
        if start_date and end_date:
            where["published_at"] = {"$gte": start_date, "$lte": end_date}
        elif start_date:
            where["published_at"] = {"$gte": start_date}
        elif end_date:
            where["published_at"] = {"$lte": end_date}
            
        return self.search_by_metadata(where, limit)

    def print_collection_summary(self):
        """Print a summary of the collection contents."""
        try:
            # Get collection stats
            stats = self.get_collection_stats()
            print("\n=== Collection Summary ===")
            print(json.dumps(stats, indent=2))
            
            # Get sample documents
            print("\n=== Sample Documents ===")
            results = self.collection.get(limit=5)
            
            # Print sample documents in a table format
            table_data = []
            for i in range(len(results["ids"])):
                metadata = results["metadatas"][i]
                # Convert Unix timestamp to readable date if available
                published_at = metadata.get("published_at", "unknown")
                if published_at != "unknown":
                    try:
                        # Try parsing ISO format first
                        try:
                            published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        except ValueError:
                            # If that fails, try parsing as Unix timestamp
                            published_at = datetime.fromtimestamp(int(published_at))
                        published_at = published_at.strftime('%Y-%m-%d %H:%M:%S')
                    except (ValueError, TypeError):
                        pass
                        
                table_data.append([
                    results["ids"][i],
                    metadata.get("video_id", "unknown"),
                    metadata.get("channel_id", "unknown"),
                    metadata.get("channel_name", "unknown"),
                    published_at,
                    metadata.get("video_title", "unknown"),
                    len(results["documents"][i])
                ])
            
            headers = ["ID", "Video ID", "Channel ID", "Channel Name", "Published At", "Title", "Text Length"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            
        except Exception as e:
            logger.error(f"Failed to print collection summary: {e}")
            print(f"Error printing collection summary: {e}")

    def print_video_summary(self, video_id: str):
        """Print a summary of chunks for a specific video."""
        try:
            # Get all chunks for the video
            results = self.collection.get(
                where={"video_id": video_id}
            )
            
            if not results["ids"]:
                print(f"No chunks found for video ID: {video_id}")
                return
                
            # Get metadata from first chunk (should be same for all chunks of same video)
            metadata = results["metadatas"][0]
            
            # Convert Unix timestamp to readable date if available
            published_at = metadata.get("published_at", "unknown")
            if published_at != "unknown":
                try:
                    # Try parsing ISO format first
                    try:
                        published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    except ValueError:
                        # If that fails, try parsing as Unix timestamp
                        published_at = datetime.fromtimestamp(int(published_at))
                    published_at = published_at.strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError):
                    pass
            
            print(f"\n=== Video Summary for {video_id} ===")
            print(f"Title: {metadata.get('video_title', 'unknown')}")
            print(f"Channel: {metadata.get('channel_name', 'unknown')} (ID: {metadata.get('channel_id', 'unknown')})")
            print(f"Published: {published_at}")
            print(f"Total Chunks: {len(results['ids'])}")
            
            # Print chunks in a table
            table_data = []
            for i in range(len(results["ids"])):
                table_data.append([
                    results["ids"][i],
                    results["metadatas"][i].get("chunk_index", -1),
                    len(results["documents"][i])
                ])
            
            print("\nChunks:")
            print(tabulate(
                table_data,
                headers=["Chunk ID", "Index", "Text Length"],
                tablefmt="grid"
            ))
            
        except Exception as e:
            logger.error(f"Failed to print video summary: {e}")
            print(f"Error printing video summary: {e}")

    def clear_collection(self) -> bool:
        """Clear all data from the collection."""
        try:
            # Get all document IDs
            all_ids = self.collection.get(include=[])["ids"]
            if all_ids:
                # Delete documents in batches of 166 (ChromaDB's limit)
                batch_size = 166
                for i in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[i:i + batch_size]
                    self.collection.delete(ids=batch_ids)
            logger.info(f"Cleared collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

def main():
    """Example usage of ChromaStorage."""
    # Initialize storage
    storage = ChromaStorage(
        persist_directory="chroma_db",
        collection_name="youtube_transcripts"
    )
    
    # Load data from the latest processed folder
    output_dir = Path("outputs/youtube")
    if output_dir.exists():
        # Get the latest folder
        latest_folder = max((d for d in output_dir.iterdir() if d.is_dir()), key=lambda x: x.stat().st_mtime)
        processed_folder = latest_folder / "processed"
        print(f"\nLoading data from: {processed_folder}")
        if processed_folder.exists():
            # Store all JSON files from the processed folder
            stats = storage.store_directory(processed_folder)
            print("\nStorage Statistics:")
            print(f"Added: {stats['added']}")
            print(f"Skipped: {stats['skipped']}")
            print(f"Errors: {stats['errors']}")
        else:
            print(f"Processed folder does not exist: {processed_folder}")
    
    # Print collection summary
    print("\n=== Collection Summary ===")
    storage.print_collection_summary()
    
    # Example: Print summary for a specific video (if any exists)
    if storage.get_collection_stats()["total_chunks"] > 0:
        # Get the first video ID from the collection
        first_chunk = storage.list_all_documents(limit=1)[0]
        video_id = first_chunk["metadata"]["video_id"]
        print(f"\n=== Example Video Summary: {video_id} ===")
        storage.print_video_summary(video_id)
        
        # Example: Search by metadata
        print("\n=== Example Search Results ===")
        # Search for chunks from the same channel
        channel_name = first_chunk["metadata"]["channel_name"]
        channel_chunks = storage.search_by_metadata(
            {"channel_name": channel_name},
            limit=3
        )
        for chunk in channel_chunks:
            print(f"\nVideo: {chunk['metadata']['video_title']}")
            print(f"Channel: {chunk['metadata']['channel_name']}")
            print(f"Text: {chunk['text'][:200]}...")

if __name__ == "__main__":
    main() 
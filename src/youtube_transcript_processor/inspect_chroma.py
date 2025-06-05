"""Inspect and display data inside ChromaDB."""

from pathlib import Path
from tabulate import tabulate
import argparse
from chroma_storage import ChromaStorage

def inspect_chroma_data():
    # Initialize ChromaStorage
    storage = ChromaStorage(
        persist_directory="chroma_db",
        collection_name="youtube_transcripts_v1"
    )
    
    # Get all documents from the collection
    docs = storage.list_all_documents(limit=1000)  # Adjust limit as needed
    
    if not docs:
        print("No documents found in the collection.")
        return
    
    # Prepare table data
    table_data = []
    for doc in docs:
        table_data.append([
            doc["id"],
            doc["metadata"].get("video_id", "unknown"),
            doc["metadata"].get("channel_name", "unknown"),
            doc["metadata"].get("upload_date", "unknown"),
            doc["metadata"].get("title", "unknown"),
            len(doc["text"])
        ])
    
    # Print table
    headers = ["ID", "Video ID", "Channel", "Upload Date", "Title", "Text Length"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def clear_chroma_data():
    # Initialize ChromaStorage
    storage = ChromaStorage(
        persist_directory="chroma_db",
        collection_name="youtube_transcripts_v1"
    )
    
    # Clear all documents from the collection
    storage.clear_collection()
    print("All documents have been cleared from the collection.")

def main():
    parser = argparse.ArgumentParser(description='ChromaDB data inspection and management tool')
    parser.add_argument('--data', action='store_true', help='Display all data in the collection')
    parser.add_argument('--clear', action='store_true', help='Clear all data from the collection')
    
    args = parser.parse_args()
    
    if args.data:
        inspect_chroma_data()
    elif args.clear:
        clear_chroma_data()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
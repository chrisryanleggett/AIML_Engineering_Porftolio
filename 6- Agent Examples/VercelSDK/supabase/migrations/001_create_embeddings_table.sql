CREATE EXTENSION IF NOT EXISTS vector;

-- Create the documents table to store vector embeddings for semantic search & RAG
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(1536)
);
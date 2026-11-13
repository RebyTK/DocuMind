BEGIN;

-- Enable pgvector.
CREATE EXTENSION IF NOT EXISTS vector;

-- Track applied database migrations.
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Store document-level information.
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,

    filename TEXT NOT NULL,

    file_type TEXT NOT NULL
        CHECK (file_type IN ('pdf', 'docx', 'txt')),

    file_hash TEXT NOT NULL UNIQUE,

    total_chunks INTEGER NOT NULL DEFAULT 0
        CHECK (total_chunks >= 0),

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Store individual document chunks and their embeddings.
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,

    document_id TEXT NOT NULL,

    chunk_index INTEGER NOT NULL
        CHECK (chunk_index >= 0),

    page_number INTEGER
        CHECK (page_number IS NULL OR page_number > 0),

    content TEXT NOT NULL
        CHECK (LENGTH(TRIM(content)) > 0),

    embedding VECTOR(384) NOT NULL,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_document
        FOREIGN KEY (document_id)
        REFERENCES documents(document_id)
        ON DELETE CASCADE,

    CONSTRAINT unique_document_chunk
        UNIQUE (document_id, chunk_index)
);

-- Speed up document-specific chunk queries.
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
ON document_chunks(document_id);

-- Approximate nearest-neighbor search using cosine distance.
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
ON document_chunks
USING hnsw (embedding vector_cosine_ops);

-- Record the migration.
INSERT INTO schema_migrations (version)
VALUES ('001_initial_schema')
ON CONFLICT (version) DO NOTHING;

COMMIT;

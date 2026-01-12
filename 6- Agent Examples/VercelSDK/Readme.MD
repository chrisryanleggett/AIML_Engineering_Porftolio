This portfolio example demonstrates a RAG (Retrieval Augmented Generation) system for document ingestion and embedding storage using OpenAI and Supabase.

## Getting Started

**Entry Point:** `index.js`

**Setup:**
```bash
npm install
```

**Environment Variables:**
Create a `.env` file with:
- `OPENAI_API_KEY` - Your OpenAI API key
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_SECRET_KEY` - Your Supabase service role key

**Run:**
```bash
npm start
```

This will ingest documents from `src/rag/docs/` directory, generate embeddings using OpenAI, and store them in a Supabase Postgres database.

## Architecture

![Agent Architecture](../ReadmeImages/AgentArchitecture.png)

**Current Implementation:**
- Document ingestion system that processes files from `src/rag/docs/`
- OpenAI embeddings generation (`text-embedding-3-small` model)
- Supabase/Postgres database storage for embeddings and document content
- Automatic table clearing and repopulation on each run

## Under Development / Next Steps

- Express server for API routes
- Web search functionality
- Function calls to enable LLM to call external tools
- Chat interface with streaming responses
- Multi-agent team with common chat/message bus setup
- Vercel SDK integration for unified provider abstraction







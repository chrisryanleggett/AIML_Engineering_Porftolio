This portfolio example demonstrates a **multi-agent RAG system** that uses structured outputs for query classification and routing to specialized agents. The system leverages the Vercel AI SDK for unified provider abstraction, OpenAI embeddings, and Supabase for vector storage.

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

The application will:
1. Ingest documents from `src/rag/docs/` (including subdirectories)
2. Generate embeddings using OpenAI's `text-embedding-3-small` model
3. Store embeddings in Supabase Postgres database
4. Run the RAG system with a sample query
5. Enter **interactive mode** for query routing testing

## Architecture

![Agent Architecture](../ReadmeImages/Architecture.png)

### Current Implementation

**Core Components:**
- **Document Ingestion**: Recursively processes `.txt` files from `src/rag/docs/` subdirectories
- **Embedding Generation**: Uses AI SDK's `embed()` function with OpenAI embeddings
- **Vector Storage**: Supabase/Postgres database with pgvector for similarity search
- **Query Classification**: Structured output using `generateObject()` to classify queries
- **Query Routing**: Routes queries to specialized agents based on classification
- **Interactive Mode**: Test query routing with real-time input

### Multi-Agent System

The system supports **7 specialized agents**, each with their own knowledge base:

1. **Customer Support** - Account management, billing, password resets, refunds
2. **Engineering** - Code standards, architecture, debugging, technical implementation
3. **DevOps** - Deployment, CI/CD, infrastructure, monitoring
4. **Frontend** - UI/UX guidelines, React patterns, component design
5. **Product** - Feature specifications, roadmap, user experience
6. **Marketing** - Content strategy, campaigns, SEO, analytics
7. **General** - Questions that don't fit a specific category

### Document Structure

Documents are organized by agent category:
```
src/rag/docs/
  ├── customer_support/
  │   ├── account_management.txt
  │   └── billing_refunds.txt
  ├── engineering/
  │   ├── coding_standards.txt
  │   └── architecture.txt
  ├── devops/
  │   ├── deployment.txt
  │   └── cicd.txt
  ├── frontend/
  │   ├── ui_guidelines.txt
  │   └── react_patterns.txt
  ├── product/
  │   ├── feature_specs.txt
  │   └── roadmap.txt
  └── marketing/
      ├── content_strategy.txt
      └── campaigns.txt
```

Each document chunk is stored with metadata including:
- `source`: Relative file path
- `filename`: Document filename
- `agent_category`: Agent category for routing

### Query Routing Flow

1. **User Query** → Entered via interactive prompt
2. **Query Classification** → `classifyQuery()` uses structured output to determine:
   - Primary agent to handle the query
   - Confidence score
   - Whether RAG context is needed
3. **Query Routing** → `routeQuery()` provides detailed routing:
   - Primary and secondary agents
   - Query type (factual, how_to, troubleshooting, etc.)
   - Escalation requirements
4. **Agent Selection** → Route to appropriate specialized agent (future: tool calling)

### Technologies Used

- **Vercel AI SDK** (`ai` package) - Unified interface for LLM operations
  - `generateText()` - Text generation
  - `generateObject()` - Structured output with Zod schemas
  - `embed()` - Vector embeddings
- **@ai-sdk/openai** - OpenAI provider for AI SDK
- **Zod** - Schema validation for structured outputs
- **Supabase** - PostgreSQL database with pgvector extension
- **OpenAI** - Embeddings and text generation models

## Interactive Mode

After running the initial RAG system, the application enters **interactive mode**:

```
Enter a query (or "exit" to quit): How do I reset my password?
```

The system will:
- Classify the query using structured output
- Route it to the appropriate agent
- Display routing decisions with confidence scores

**Example Queries to Test:**
- "How do I reset my password?" → `customer_support`
- "What's the deployment process?" → `devops`
- "How do I style React components?" → `frontend`
- "What's our product roadmap?" → `product`
- "How do I optimize our CI/CD?" → `devops`

## Under Development / Next Steps

- **Tool Calling**: Convert RAG system into reusable tools for agents
- **Agent Tools**: Specialized tools for each agent category
- **Multi-Agent Orchestration**: Agents working together with tool calling
- **Web Search Integration**: Web search tool for agents
- **Express Server**: API endpoints for production use
- **Chat Interface**: Streaming responses with chat UI
- **Agent Communication**: Message bus for multi-agent coordination







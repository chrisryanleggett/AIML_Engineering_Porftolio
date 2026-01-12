import 'dotenv/config';
import { SIMILARITY_MATCH_COUNT, EMBEDDING_MODEL_NAME, SIMILARITY_THRESHOLD } from "../constants.js";
import { supabase } from "../config.js";
import { openai } from '@ai-sdk/openai';
import { embed } from 'ai';

export async function retrieveSimilarDocs(query) {
    // Create vector embeddings based on user query using AI SDK
    const { embedding } = await embed({
        model: openai.textEmbeddingModel(EMBEDDING_MODEL_NAME),
        value: query,
    });

    // Retrieve similar docs from supabase based on the embeddings
    const { data: documents, error: matchError } = await supabase.rpc(
        'match_documents',
        {
            query_embedding: embedding,
            match_count: SIMILARITY_MATCH_COUNT,
            match_threshold: SIMILARITY_THRESHOLD, 
        }
    );

    if (matchError) {
        throw new Error(`Failed to fetch docs from supabase. Error: ${matchError}`);
    }

    return documents;
}
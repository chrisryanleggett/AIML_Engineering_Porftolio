import 'dotenv/config';
import { openai } from '../config.js';
import { createClient } from '@supabase/supabase-js';

// See OpenAI Embeddings API documentation: https://platform.openai.com/docs/api-reference/embeddings/create
// Public async function to create embeddings form input text using OpenAI library
export async function createEmbedding(content) {
    const response = await openai.embeddings.create({ // wait for async OpenAI embeddings API call
        model: "text-embedding-3-small",
        input: content,
    });
    // .data property from OpenAI API response holds embeddings array; [0] gets the first embedding
    return response.data[0].embedding;
}

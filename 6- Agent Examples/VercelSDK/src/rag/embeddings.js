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

// Supabase config: ensure environment variables are set, else throw descriptive errors
const supabasePrivateKey = process.env.SUPABASE_SECRET_KEY;
if (!supabasePrivateKey) throw new Error("SUPABASE_SECRET_KEY is missing or invalid");
const supabaseUrl = process.env.SUPABASE_URL;
if (!supabaseUrl) throw new Error("SUPABASE_URL is missing or invalid");
// Export supabase connection so other files can use the Supabase client, createClient is a Supabase function
export const supabase = createClient(supabaseUrl, supabasePrivateKey);
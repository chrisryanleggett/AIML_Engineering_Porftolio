// Application entry point
import 'dotenv/config';
import readline from 'readline';
import { ingestDocuments } from './src/rag/upsertDocuments.js';
import { retrieveSimilarDocs } from './src/rag/retrieveSimilarDocs.js';
import { getRagPrompt, combineDocuments } from './src/utils.js';
import { ANSWERING_MODEL } from './src/constants.js';
import { openai } from '@ai-sdk/openai';
import { generateText, embed } from 'ai';

// imports for generating structured outputs which will be important for agentic tool calling
import { generateObject } from 'ai';
import { z } from 'zod';

/*
 Build a basic retrieval system
*/

const EMBEDDING_MODEL_NAME = 'text-embedding-3-small';
const aiModel = openai(ANSWERING_MODEL);
const TEST_EMBEDDINGS = true; // Set to false to skip embedding test

const query = "How many houses were damaged during the great fire of london?";

async function main() {
    // ------------------------RAG System Implementation------------------

    await ingestDocuments(); // Ingest documents into Supabase PostgreSQL database
    const retrievedDocs = await retrieveSimilarDocs(query); // Retrieve similar documents

    console.log('\nRetrieved documents:');
    console.log(JSON.stringify(retrievedDocs, null, 2));

    const contextString = combineDocuments(retrievedDocs); // Combine retrieved documents into a context string
    const prompt = getRagPrompt(contextString, query); // create a prompt including context docs to send to the model

    // send prompt to model to generate response using AI SDK
    const { text } = await generateText({
        model: aiModel,
        prompt: prompt,
    });

    console.log('\nResponse:');
    console.log(text);

    console.log('\n' + '='.repeat(60));
    console.log('Application completed.');
    console.log('='.repeat(60));
}

/*
Query Router: Use structured output to classify queries and route to specialized agents
*/
async function classifyQuery(userQuery) {
    const result = await generateObject({
        model: aiModel,
        schemaName: 'query_classification',
        schemaDescription: 'Classify user queries to determine the appropriate agent to handle them.',
        // This is the definition of the schema for the classification object
        schema: z.object({
            reasoning: z
                .string()
                .describe('Brief explanation for why this classification was chosen'),
            primaryAgent: z
                .enum(['customer_support', 'engineering', 'devops', 'frontend', 'product', 'marketing', 'general'])
                .describe('Primary specialized agent that should handle this query'),
            confidence: z
                .number()
                .min(0)
                .max(1)
                .describe('Confidence score for this classification (0-1)'),
            requiresContext: z
                .boolean()
                .describe('Whether this query requires knowledge base context (RAG)'),
        }),
        prompt: `Classify this user query and determine which specialized agent should handle it:\n\n"${userQuery}"\n\n` +
                `Available agents:\n` +
                `- customer_support: Account issues, billing, password resets, order status, refunds\n` +
                `- engineering: Code questions, architecture, technical implementation, debugging\n` +
                `- devops: Infrastructure, deployment, CI/CD, monitoring, server configuration\n` +
                `- frontend: UI/UX questions, React, CSS, frontend frameworks, component design\n` +
                `- product: Feature requests, product strategy, user experience, roadmap\n` +
                `- marketing: Campaigns, content strategy, SEO, analytics, branding\n` +
                `- general: General questions that don't fit a specific category`,
    });

    console.log('\n=== Query Classification ===');
    console.log(JSON.stringify(result.object, null, 2));
    
    return result.object;
}

/*
The generateObject call in this function enables us to generate structured output 
using the Vercel AI SDK Generate Object interface. This allows the model to return 
JSON objects that match a pre-defined schema, including explicit data types and 
required properties. By using structured output, we can ensure type safety, 
consistency, and easy downstream processing when passing agent decisions or results 
between systems.
*/
async function routeQuery(userQuery) {
    const result = await generateObject({
        model: aiModel,
        schemaName: 'query_routing',
        schemaDescription: 'Route queries to specialized agents with detailed routing information.',
        schema: z.object({
            reasoning: z
                .string()
                .describe('Detailed reasoning for the routing decision'),
            primaryAgent: z
                .enum(['customer_support', 'engineering', 'devops', 'frontend', 'product', 'marketing', 'general'])
                .describe('Primary agent to handle this query'),
            secondaryAgents: z
                .array(z.string())
                .optional()
                .describe('Additional agents that might be helpful (if any)'),
            confidence: z
                .number()
                .min(0)
                .max(1)
                .describe('Confidence in routing decision'),
            requiresEscalation: z
                .boolean()
                .describe('Whether this query requires human review or escalation'),
            queryType: z
                .enum(['factual', 'how_to', 'troubleshooting', 'request', 'general'])
                .describe('Type of query being asked'),
        }),
        prompt: `Route this query to the appropriate specialized agent:\n\n"${userQuery}"\n\n` +
                `Consider:\n` +
                `- Which agent has the expertise to answer this?\n` +
                `- Does this need knowledge base context (RAG)?\n` +
                `- Are multiple agents needed?\n` +
                `- Does this require human escalation?`,
    });

    console.log('\n=== Query Routing Decision ===');
    console.log(JSON.stringify(result.object, null, 2));
    
    return result.object;
}

async function generateEmbeddings() {
    const { embedding } = await embed({
        model: openai.textEmbeddingModel(EMBEDDING_MODEL_NAME),
        value: 'This text will be converted to embedding',
    });

    console.log(`Embedding generated: ${embedding}`);
}

/**
 * Helper function to ask questions interactively using readline
 */
function askQuestion(query) {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    return new Promise(resolve => {
        rl.question(query, ans => {
            rl.close();
            resolve(ans);
        });
    });
}

async function run() {
    // RAG System
    await main();
    
    // Query Router for interactive mode - Primarily for testing.
    console.log('\n' + '='.repeat(60));
    console.log('QUERY ROUTER - Interactive Mode');
    console.log('='.repeat(60));
    console.log('\nEnter queries to classify and route to specialized agents.');
    console.log('Type "exit" or "quit" to stop, or press Ctrl+C to exit.\n');
    
    while (true) {
        const userQuery = await askQuestion('Enter a query (or "exit" to quit): ');
        
        const trimmedQuery = userQuery.trim();
        
        // Check for exit commands
        if (trimmedQuery.toLowerCase() === 'exit' || trimmedQuery.toLowerCase() === 'quit') {
            console.log('\nExiting interactive mode. Goodbye!');
            break;
        }
        
        // Skip empty queries
        if (!trimmedQuery) {
            console.log('Please enter a valid query.\n');
            continue;
        }
        
        // Process the query
        console.log(`\n--- Processing Query: "${trimmedQuery}" ---`);
        try {
            await classifyQuery(trimmedQuery);
            await routeQuery(trimmedQuery);
            console.log('\n' + '-'.repeat(60) + '\n');
        } catch (error) {
            console.error('\nError processing query:', error.message);
            console.log('\n' + '-'.repeat(60) + '\n');
        }
    }
    
    // Embedding test
    if (TEST_EMBEDDINGS) {
        console.log('\n' + '='.repeat(60));
        console.log('EMBEDDING TEST');
        console.log('='.repeat(60));
        await generateEmbeddings();
    }
}

run().catch(console.error);
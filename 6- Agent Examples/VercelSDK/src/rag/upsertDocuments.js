import 'dotenv/config';
import { supabase } from '../config.js';
import { EMBEDDING_MODEL_NAME, CHUNK_OVERLAP, CHUNK_SIZE } from '../constants.js';
import { simpleTextSplitter } from '../utils.js';
import { openai } from '@ai-sdk/openai';
import { embed } from 'ai';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

/*
  In ES modules, __dirname and __filename are not available by default. 
  Using fileURLToPath and import.meta.url creates these equivalents, 
  allowing our application to reliably work with file paths (e.g., for reading documents) in all environments.
*/
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


// Constants for local document folder, embedding model, target Postgres table for embeddings, and reset option.
// These are used for managing RAG embedding storage in Supabase/Postgres.
const SOURCE_DOCUMENTS_DIR = path.join(__dirname, 'docs');
const SUPABASE_TABLE_NAME = 'documents';
const CLEAR_SUPABASE_TABLE_CONTENTS = true;


// The async ingestDocuments() function enables the ingestion of documents in the/rag/docs folder
export async function ingestDocuments() {
    const docsDirPath = SOURCE_DOCUMENTS_DIR;
    console.log(`Ingesting documents from directory: ${docsDirPath}`);

    // Store documents to insert across files
    const allDocumentsToInsert = [];

    try {
        // Check if dir exists
        if (
            !fs.existsSync(docsDirPath) || !fs.lstatSync(docsDirPath).isDirectory()
        ) {
            throw new Error(
                `Source documents directory not found at ${docsDirPath}. Please create it and add files.`
            );
        }

        // Read files from directory
        const files = fs.readdirSync(docsDirPath);

        if (files.length === 0) {
            console.log(
                `No files found in ${docsDirPath}. Nothing to ingest.`
            );
            return;
        }
        console.log(`Found ${files.length} files to process.`);

        // If true, clear table so table always freshly repopulated on each run
        if (CLEAR_SUPABASE_TABLE_CONTENTS) {
            console.log(
                `Clearing existing documents from table '${SUPABASE_TABLE_NAME}'...`
            );
            const { error: deleteError } = await supabase
                .from(SUPABASE_TABLE_NAME)
                .delete()
                .neq('id', -1); //Deletes all rows in the table
            if (deleteError) {
                console.warn(
                    `Warnings: Could not clear existing documents: ${deleteError.message}`
                );
            } else {
                console.log('Existing documents cleared.');
            }
        }

        // Process each file
        let totalChunks = 0;

        for (const filename of files) {
            const filePath = path.join(docsDirPath, filename);
            console.log(`Processing file: ${filename}...`);

            // Read file contents
            const fileContent = fs.readFileSync(filePath, 'utf-8');
            console.log(` - Read ${fileContent.length} characters.`);

            // Split the large text into chunks
            const chunks = simpleTextSplitter(fileContent, CHUNK_SIZE, CHUNK_OVERLAP);

            if (chunks.length === 0) {
                console.log(` - No chunks generated for this file.`);
                continue; // Skip to next file
            }

            /**
             * Embed each chunk
             */
            console.log(`Embedding ${chunks.length} chunks of text`);
            let fileChunkCount = 0;
            for (const chunk of chunks) {
                fileChunkCount++;
                try {
                    // Create embedding using AI SDK
                    const { embedding } = await embed({
                        model: openai.textEmbeddingModel(EMBEDDING_MODEL_NAME),
                        value: chunk,
                    });
                    // Add metadata with source filename
                    allDocumentsToInsert.push({
                        content: chunk,
                        embedding: embedding,
                        metadata: { source: filename }, // Store filename here
                    });
                    console.log(`- Embedded chunk ${fileChunkCount} content from ${filename}`);
                } catch (embedError) {
                    console.error(
                        `   - Failed to embed content from ${filename}: ${embedError.message}. Skipping chunk.`
                    );
                }
            }
            totalChunks += chunks.length; // Track total chunks across all files
        }

        console.log(`Total chunks generated across all files: ${totalChunks}`);  

        if (allDocumentsToInsert.length === 0) {
            console.log(
                'No documents were successfully embedded across all files. Aborting upload.'
            );
            return;
        }

        console.log(
            `Total documents successfully prepared for insertion: ${allDocumentsToInsert.length}\n\n`
        );

        // Store all collected documents in Supabase
        console.log(
            `Uploading ${allDocumentsToInsert.length} documents to Supabase table '${SUPABASE_TABLE_NAME}' ... `
        );
        const { error: insertError } = await supabase
            .from(SUPABASE_TABLE_NAME)
            .insert(allDocumentsToInsert); //Insert all collected documents

        if (insertError) {
            console.error('Error inserting documents into Supabase:', insertError);
            throw new Error(`Supabase insert failed: ${insertError.message}`);
        } else {
            console.log(
                `Successfully inserted ${allDocumentsToInsert.length} documents into Supabase.`
            );
        }

        console.log('---Ingestion Complete! ---');
    } catch (error) {
        console.error('---Ingestion failed!--- ');
        console.error('Error during ingestion process:', error);
        process.exit(1); //exit with error code
    }
}

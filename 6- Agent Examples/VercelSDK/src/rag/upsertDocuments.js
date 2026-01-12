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

        // Recursively get all .txt files from directory and subdirectories
        function getAllTxtFiles(dir, fileList = []) {
            const files = fs.readdirSync(dir);
            
            files.forEach(file => {
                const filePath = path.join(dir, file);
                const stat = fs.statSync(filePath);
                
                if (stat.isDirectory()) {
                    getAllTxtFiles(filePath, fileList);
                } else if (file.endsWith('.txt')) {
                    fileList.push(filePath);
                }
            });
            
            return fileList;
        }
        
        const allFiles = getAllTxtFiles(docsDirPath);
        
        if (allFiles.length === 0) {
            console.log(
                `No .txt files found in ${docsDirPath} or subdirectories. Nothing to ingest.`
            );
            return;
        }
        console.log(`Found ${allFiles.length} .txt files to process.`);

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

        for (const filePath of allFiles) {
            // Get relative path from docs directory for better metadata
            const relativePath = path.relative(docsDirPath, filePath);
            const filename = path.basename(filePath);
            const directory = path.dirname(relativePath);
            
            // Extract agent category from directory path (e.g., "customer_support", "engineering")
            const agentCategory = directory === '.' ? 'general' : directory;
            
            console.log(`Processing file: ${relativePath}...`);

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
                    // Add metadata with source filename and agent category
                    allDocumentsToInsert.push({
                        content: chunk,
                        embedding: embedding,
                        metadata: { 
                            source: relativePath,
                            filename: filename,
                            agent_category: agentCategory
                        },
                    });
                    console.log(`- Embedded chunk ${fileChunkCount} from ${relativePath}`);
                } catch (embedError) {
                    console.error(
                        `   - Failed to embed content from ${relativePath}: ${embedError.message}. Skipping chunk.`
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

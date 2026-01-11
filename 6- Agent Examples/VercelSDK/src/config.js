import OpenAI from 'openai';
import dotenv from 'dotenv';

// Load the environment variables from the .env file
dotenv.config();

export const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

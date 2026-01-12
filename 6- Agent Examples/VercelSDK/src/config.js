import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

// Supabase client
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SECRET_KEY;
if (!supabaseUrl || !supabaseKey) {
    throw new Error("Missing SUPABASE_URL or SUPABASE_SECRET_KEY");
}
export const supabase = createClient(supabaseUrl, supabaseKey);
import fs from 'fs/promises';
import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from "@google/generative-ai";
import dotenv from 'dotenv';
import path from 'path';

dotenv.config();

class DictionaryProcessor {
    constructor() {
        const apiKey = process.env.API_KEY6;
        if (!apiKey) throw new Error("Gemini API_KEY not found");
        this.genAI = new GoogleGenerativeAI(apiKey);
        this.model = this.genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
    }

    async sleep(min, max) {
        const ms = Math.floor(Math.random() * (max - min + 1) + min);
        return new Promise(resolve => setTimeout(resolve, ms * 1000));
    }

    async queryGemini(words) {
        const prompt = `For each word in the following list, provide Chinese translation and details in JSON format:
${words.join(', ')}

Required format for each word:
{
    "word": "example",
    "phonetic": "/ɪɡˈzæmpəl/",
    "translation": "例子",
    "description": "详细介绍(100字以内)",
    "synonyms": ["similar1", "similar2"],
    "antonyms": ["opposite1", "opposite2"]
}

Special handling:
1. If word is in ALL CAPS: Try to find the common-case version
2. If word appears misspelled: Suggest the closest correct word
3. If word cannot be directly translated: Provide closest equivalent

Example special cases:
{
    "word": "RUNTIME",
    "suggested": "runtime",
    "phonetic": "/ˈrʌnˌtaɪm/",
    "translation": "运行时",
    "description": "程序运行期间的时间段，也指程序在运行时的环境",
    "synonyms": ["execution time", "running time"],
    "antonyms": ["compile time", "design time"]
}

{
    "word": "progam",
    "suggested": "program",
    "phonetic": "/ˈproʊˌɡræm/",
    "translation": "程序",
    "description": "发现可能的拼写错误，已更正为'program'并提供其翻译",
    "synonyms": ["application", "software"],
    "antonyms": ["hardware", "equipment"]
}

Return as a JSON array. For any field that cannot be determined, use null.`;

        const chat = this.model.startChat({
            generationConfig: {
                maxOutputTokens: 8192,
                temperature: 1,
            },
            safetySettings: [
                { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
                { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
                { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
            ]
        });

        try {
            let response = "";
            const result = await chat.sendMessageStream(prompt);
            for await (const chunk of result.stream) {
                response += chunk.text();
            }

            const jsonMatch = response.match(/\[[\s\S]*\]/);
            return jsonMatch ? JSON.parse(jsonMatch[0]) : [];
        } catch (error) {
            console.error("Error querying Gemini:", error);
            return [];
        }
    }

    async processWords() {
        try {
            // Read words file
            const content = await fs.readFile('./cached_words.txt', 'utf-8');
            const words = content.split('\n').filter(word => word.trim());

            // Split into chunks of 100
            const chunks = [];
            for (let i = 0; i < words.length; i += 100) {
                chunks.push(words.slice(i, i + 100));
            }

            const allResults = [];

            // Process each chunk
            for (let i = 0; i < chunks.length; i++) {
                console.log(`Processing chunk ${i + 1}/${chunks.length}`);
                const results = await this.queryGemini(chunks[i]);
                allResults.push(...results);

                // Random delay between requests
                await this.sleep(1, 3);
            }

            // Save results
            await fs.writeFile(
                'all_dic.json',
                JSON.stringify(allResults, null, 2),
                'utf-8'
            );

            console.log("Dictionary processing completed!");
            console.log(`Total words processed: ${allResults.length}`);

        } catch (error) {
            console.error("Error processing dictionary:", error);
        }
    }
}

// Run the processor
const processor = new DictionaryProcessor();
processor.processWords();
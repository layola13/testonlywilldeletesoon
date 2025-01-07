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
        const prompt = `For each word in the following list, provide Chinese translation, details and example sentences in JSON format:
${words.join(', ')}

Required format for each word:
{
    "word": "example",
    "phonetic": "/ɪɡˈzæmpəl/",
    "translation": "例子",
    "description": "详细介绍(100字以内)",
    "synonyms": ["similar1", "similar2"],
    "antonyms": ["opposite1", "opposite2"],
    "examples": [
        {
            "en": "This is a good example of modern architecture.",
            "zh": "这是现代建筑的一个好例子。"
        },
        {
            "en": "Let me give you an example.",
            "zh": "让我给你举个例子。"
        }
    ]
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
    "antonyms": ["compile time", "design time"],
    "examples": [
        {
            "en": "The program has a runtime error.",
            "zh": "这个程序有一个运行时错误。"
        },
        {
            "en": "The runtime environment must be configured correctly.",
            "zh": "运行时环境必须正确配置。"
        }
    ]
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
                const chunkText = chunk.text();
                process.stdout.write(chunkText);
                totalResponse += chunkText;
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
            // Ensure temp directory exists
            const tempDir = './dic_temp';
            await fs.mkdir(tempDir, { recursive: true });
    
            // Read words file
            const content = await fs.readFile('./cached_words.txt', 'utf-8');
            const words = content.split('\n').filter(word => word.trim());
    
            // Find last processed index
            const tempFiles = await fs.readdir(tempDir);
            const processedIndices = tempFiles
                .map(f => parseInt(f.split('.')[0]))
                .filter(n => !isNaN(n));
            const startIndex = processedIndices.length ? Math.max(...processedIndices) + 100 : 0;
    
            const size=50;
            // Process remaining chunks
            for (let i = startIndex; i < words.length; i += size) {
                const chunk = words.slice(i, i + size);
                console.log(`Processing chunk ${i/size + 1}, words ${i}-${i + chunk.length}`);
                
                try {
                    const results = await this.queryGemini(chunk);
                    
                    // Save chunk results to temp file
                    const tempFile = path.join(tempDir, `${i}.json`);
                    await fs.writeFile(
                        tempFile,
                        JSON.stringify(results, null, 2),
                        'utf-8'
                    );
    
                    // Random delay between requests
                    await this.sleep(1, 3);
                } catch (error) {
                    console.error(`Error processing chunk ${i/size + 1}:`, error);
                    continue; // Skip to next chunk on error
                }
            }
    
            // Combine all temp files into final result
            const allResults = [];
            const finalTempFiles = await fs.readdir(tempDir);
            for (const file of finalTempFiles.sort((a, b) => 
                parseInt(a.split('.')[0]) - parseInt(b.split('.')[0]))) {
                const content = await fs.readFile(path.join(tempDir, file), 'utf-8');
                allResults.push(...JSON.parse(content));
            }
    
            // Save combined results
            await fs.writeFile(
                'all_dic.json',
                JSON.stringify(allResults, null, 2),
                'utf-8'
            );
    
            console.log("Dictionary processing completed!");
            console.log(`Total words processed: ${allResults.length}`);
    
        } catch (error) {
            console.error("Error processing dictionary:", error);
            throw error;
        }
    }
}

// Run the processor
const processor = new DictionaryProcessor();
processor.processWords();
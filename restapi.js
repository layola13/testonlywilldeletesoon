import express from 'express';
import multer from 'multer';
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Mistral } from "@mistralai/mistralai";
import dotenv from "dotenv";
import sharp from 'sharp';
import rateLimit from 'express-rate-limit';
dotenv.config();

// Configure rate limiter
const limiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 10, // 1 request per window
    message: {
        status: 429,
        error: "Too many requests, please try again after 1 minute"
    },
    standardHeaders: true,
    legacyHeaders: false
});

const app = express();
const upload = multer({ storage: multer.memoryStorage() });
const port = 9081;

// Add after imports
let requestCounter = {
    analyze: 0,
    compareAnalyze: 0,
    total: 0
};
// Model type enum
const ModelType = {
    GEMINI: 'GEMINI',
    MIXTRAL: 'MIXTRAL',
    GEMINI_THINKING: 'GEMINI_THINKING'
};

class ImageAnalysisClient {
    constructor() {
        this.init();
    }

    init() {
        // Initialize Gemini
         // Initialize Gemini models
         const geminiApiKey = process.env.API_KEY6;
         const geminiThinkingApiKey = process.env.API_KEY5;
         if (!geminiApiKey) throw new Error("Gemini API_KEY not found");
         if (!geminiThinkingApiKey) throw new Error("Gemini Thinking API_KEY not found");
         
         this.genAI = new GoogleGenerativeAI(geminiApiKey);
         this.genAIThinking = new GoogleGenerativeAI(geminiThinkingApiKey);
 
         // Initialize Mixtral
         const mixtralApiKey = process.env.API_KEY_MIXTRAL12;
         if (!mixtralApiKey) throw new Error("Mixtral API_KEY not found");
         this.mistral = new Mistral({ apiKey: mixtralApiKey });
    }

    async analyzeImage(imageBuffer, modelType) {
        const processedImageBuffer = await sharp(imageBuffer)
            .grayscale()
            .jpeg({ quality: 100, progressive: true })
            .toBuffer();
        const base64Image = processedImageBuffer.toString('base64');

        const prompt = `Analyze the image for production date and expiration date. Return in JSON format.

    Rules:
    - Only extract dates that are explicitly labeled or clearly marked
    - If no clear production date or manufacturing date is found, set production_date to null
    - If no clear expiration date or 保质期 or 质期 is found, set expiration_date to null
    - Do not make assumptions or guess dates EXCEPT:
      * If only one date is found with no label:
        - If date is future (after ${new Date().toISOString().split('T')[0]}), set as expiration_date
        - If date is past, set as production_date
    - Date format must be YYYY.MM.DD when found
    - Production date and expiration date cannot be the same day
    
    Example responses:
    Case 1 - Labeled dates:
    {
        "production_date": "2024.08.20",    
        "expiration_date": "2026.08.20",    
        "production_id": null,
        "additional_info": null
    }

    Case 2 - Single unlabeled future date:
    {
        "production_date": null,
        "expiration_date": "2025.04.01",    // Future date assumed as expiration
        "production_id": null,
        "additional_info": "Single unlabeled date found"
    }

    Case 3 - Single unlabeled past date:
    {
        "production_date": "2023.04.01",    // Past date assumed as production
        "expiration_date": null,
        "production_id": null,
        "additional_info": "Single unlabeled date found"
    }
    
    Important: Return null for any field where the information is not explicitly visible in the image.`;

        try {
            if (modelType === ModelType.GEMINI) {
                return await this.analyzeWithGemini(base64Image, prompt);
            } else if (modelType === ModelType.GEMINI_THINKING) {
                return await this.analyzeWithGeminiThinking(base64Image, prompt);
            } else {
                return await this.analyzeWithMixtral(base64Image, prompt);
            }
        } catch (error) {
            console.error(`Error analyzing with ${modelType}:`, error);
            throw error;
        }
    }

    async analyzeWithGemini(base64Image, prompt) {
        const model = this.genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
        const result = await model.generateContent([
            { text: prompt },
            {
                inlineData: {
                    data: base64Image,
                    mimeType: "image/jpeg"
                }
            }
        ]);

        const text = result.response.text();
        const jsonMatch = text.match(/```json\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
            return JSON.parse(jsonMatch[1]);
        }
        throw new Error("No JSON content found in Gemini response");
    }

    async analyzeWithMixtral(base64Image, prompt) {
        try {
            const result = await this.mistral.chat.stream({
                model: "pixtral-large-latest",
                messages: [
                    {
                        role: "user",
                        content: [
                            { type: "text", text: prompt },
                            {
                                type: "image_url",
                                imageUrl: `data:image/jpeg;base64,${base64Image}`,
                            },
                        ]
                    }
                ],
                max_tokens: 1024,
                temperature: 0.8,
            });
    
            let response = "";
            for await (const chunk of result) {
                response += chunk.data.choices[0].delta.content;
            }
    
            const jsonMatch = response.match(/```json\s*([\s\S]*?)\s*```/);
            if (jsonMatch) {
                return JSON.parse(jsonMatch[1]);
            }
            return {
                production_date: null,
                expiration_date: null,
                production_id: null,
                additional_info: "Error: No valid JSON found in Mixtral response"
            };
        } catch (error) {
            console.error("Mixtral API error:", error);
            return {
                production_date: null,
                expiration_date: null,
                production_id: null,
                additional_info: `Mixtral Error: ${error.message}`
            };
        }
    }

    async analyzeWithGeminiThinking(base64Image, prompt) {
        const model = this.genAIThinking.getGenerativeModel({ model: "gemini-2.0-flash-thinking-exp-1219" });
        const result = await model.generateContent([
            { text: prompt },
            {
                inlineData: {
                    data: base64Image,
                    mimeType: "image/jpeg"
                }
            }
        ]);

        const text = result.response.text();
        const jsonMatch = text.match(/```json\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
            return JSON.parse(jsonMatch[1]);
        }
        throw new Error("No JSON content found in Gemini Thinking response");
    }

    async ask(prompt, modelType) {
        try {
            if (modelType === ModelType.GEMINI || modelType === ModelType.GEMINI_THINKING) {
                const genAI = modelType === ModelType.GEMINI ? this.genAI : this.genAIThinking;
                const modelName = modelType === ModelType.GEMINI ? "gemini-1.5-flash-002" : "gemini-2.0-flash-thinking-exp-1219";
                const model = genAI.getGenerativeModel({ model: modelName });
                
                const chat = model.startChat({
                    generationConfig: {
                        maxOutputTokens: 1024*1024,
                        temperature: 1,
                    },
                    safetySettings: [
                        { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
                        { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
                        { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                        { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                    ]
                });

                let totalResponse = "";
                const result = await chat.sendMessageStream(prompt);
                for await (const chunk of result.stream) {
                    const chunkText = chunk.text();
                    totalResponse += chunkText;
                }
                return { response: totalResponse };
            } else {
                // Mixtral handling
                const result = await this.mistral.chat.stream({
                    model: "mistral-large-latest",
                    messages: [{ role: "user", content: prompt }],
                    max_tokens: 1024*128,
                    temperature: 0.8,
                });
                
                let response = "";
                for await (const chunk of result) {
                    response += chunk.data.choices[0].delta.content;
                }
                return { response };
            }
        } catch (error) {
            if (error.toString().includes("Too Many Requests") || 
                error.toString().includes("Please try again later")) {
                throw new Error("Rate limit exceeded, please try again later");
            }
            console.error(`Error in ${modelType} ask:`, error);
            throw error;
        }
    }
}

const client = new ImageAnalysisClient();

app.post('/analyze', limiter,upload.single('image'), async (req, res) => {
    requestCounter.analyze++;
    requestCounter.total++;
    try {
        if (!req.file) {
            return res.status(400).json({ status: 400, error: "No image file provided" });
        }

        const modelType = req.body.model?.toUpperCase();
        if (!ModelType[modelType]) {
            return res.status(400).json({ status: 400, error: "Invalid model type. Use GEMINI or MIXTRAL" });
        }

        const result = await client.analyzeImage(req.file.buffer, modelType);
        res.json({ status: 200, data: result });
    } catch (error) {
        console.error("Analysis error:", error);
        res.status(500).json({ status: 500, error: error.message });
    }
});


// Update HTML form
app.get('/check', limiter, (req, res) => {
    res.send(`
        <html>
            <body>
                <h1>AI API Service</h1>
                <h2>API Endpoints:</h2>
                <ul>
                    <li>POST /analyze - Upload image for analysis</li>
                    <li>POST /ask - Ask AI a question</li>
                    <li>GET /status - Check API status</li>
                </ul>
                
                <h2>Image Analysis Form:</h2>
                <form action="/analyze" method="post" enctype="multipart/form-data">
                    <p>Select image file: <input type="file" name="image" accept="image/*" required></p>
                    <p>Select model: 
                        <select name="model" required>
                            <option value="GEMINI">GEMINI</option>
                            <option value="MIXTRAL">MIXTRAL</option>
                            <option value="GEMINI_THINKING">GEMINI THINKING</option>
                        </select>
                    </p>
                    <input type="submit" value="Analyze">
                </form>

                <h2>Ask AI Form:</h2>
                <form id="askForm">
                    <p>Question: <input type="text" id="prompt" required style="width:300px"></p>
                    <p>Select model: 
                        <select id="model" required>
                            <option value="GEMINI">GEMINI</option>
                            <option value="MIXTRAL">MIXTRAL</option>
                            <option value="GEMINI_THINKING">GEMINI THINKING</option>
                        </select>
                    </p>
                    <button type="submit">Ask</button>
                    <pre id="result"></pre>
                </form>

                <script>
                document.getElementById('askForm').onsubmit = async (e) => {
                    e.preventDefault();
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            prompt: document.getElementById('prompt').value,
                            model: document.getElementById('model').value
                        })
                    });
                    const data = await response.json();
                    document.getElementById('result').textContent = 
                        JSON.stringify(data, null, 2);
                };
                </script>
            </body>
        </html>
    `);
});

app.post('/compareAnalyze',limiter, upload.single('image'), async (req, res) => {
    requestCounter.compareAnalyze++;
    requestCounter.total++;
    try {
        if (!req.file) {
            return res.status(400).json({ status: 400, error: "No image file provided" });
        }

        const [geminiResult, mixtralResult, geminiThinkingResult] = await Promise.all([
            client.analyzeImage(req.file.buffer, ModelType.GEMINI)
                .catch(error => ({
                    production_date: null,
                    expiration_date: null,
                    production_id: null,
                    additional_info: null
                })),
            client.analyzeImage(req.file.buffer, ModelType.MIXTRAL)
                .catch(error => ({
                    production_date: null,
                    expiration_date: null,
                    production_id: null,
                    additional_info: null
                })),
            client.analyzeImage(req.file.buffer, ModelType.GEMINI_THINKING)
                .catch(error => ({
                    production_date: null,
                    expiration_date: null,
                    production_id: null,
                    additional_info: null
                }))
        ]);

        res.json({
            status: 200,
            datas: [geminiResult, mixtralResult, geminiThinkingResult]
        });
    } catch (error) {
        console.error("Comparison analysis error:", error);
        res.status(500).json({
            status: 500,
            error: 'Unknown error, please contact the administrator'
        });
    }
});
// Add status endpoint
// Update status endpoint
app.get('/status', (req, res) => {
    res.json({
        status: "running",
        models: "model",
        version: "1.0.0",
        copyright: "sonygod",
        requests: {
            f1: requestCounter.analyze,
            f2: requestCounter.compareAnalyze,
            total: requestCounter.total
        }
    });
});

app.post('/ask', limiter, express.json(), async (req, res) => {
    requestCounter.total++;
    try {
        const { prompt, model } = req.body;
        
        if (!prompt) {
            return res.status(400).json({ 
                status: 400, 
                error: "No prompt provided" 
            });
        }

        const modelType = model?.toUpperCase();
        if (!ModelType[modelType]) {
            return res.status(400).json({ 
                status: 400, 
                error: "Invalid model type. Use GEMINI, MIXTRAL, or GEMINI_THINKING" 
            });
        }

        const result = await client.ask(prompt, modelType);
        res.json({ status: 200, data: result });
    } catch (error) {
        console.error("Ask error:", error);
        res.status(500).json({ status: 500, error: error.message });
    }
});

// Change server binding
app.listen(port, '0.0.0.0', () => {
    console.log(`Server running on port ${port} (0.0.0.0)`);
});
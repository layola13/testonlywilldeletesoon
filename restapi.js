import express from 'express';
import multer from 'multer';
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Mistral } from "@mistralai/mistralai";
import dotenv from "dotenv";
import sharp from 'sharp';

dotenv.config();

const app = express();
const upload = multer({ storage: multer.memoryStorage() });
const port = 9081;

// Model type enum
const ModelType = {
    GEMINI: 'GEMINI',
    MIXTRAL: 'MIXTRAL'
};

class ImageAnalysisClient {
    constructor() {
        this.init();
    }

    init() {
        // Initialize Gemini
        const geminiApiKey = process.env.API_KEY6;
        if (!geminiApiKey) throw new Error("Gemini API_KEY not found");
        this.genAI = new GoogleGenerativeAI(geminiApiKey);

        // Initialize Mixtral
        const mixtralApiKey = process.env.API_KEY_MIXTRAL12;
        if (!mixtralApiKey) throw new Error("Mixtral API_KEY not found");
        this.mistral = new Mistral({ apiKey: mixtralApiKey });
    }

    async analyzeImage(imageBuffer, modelType) {
        const processedImageBuffer = await sharp(imageBuffer)
            .grayscale()
            .jpeg({ quality: 30, progressive: true })
            .toBuffer();
        const base64Image = processedImageBuffer.toString('base64');

        const prompt = `Recognize the production date and expiration date of items in the image, return in JSON format:
        {
            "production_date": "2024.08.20",
            "expiration_date": "2026.08.20",
            "production_id": "L233EEV",
            "additional_info": "CDNK25012111170001"
        }
        Date format should be YYYY.MM.DD. Production date and expiration date cannot be the same day.`;

        try {
            if (modelType === ModelType.GEMINI) {
                return await this.analyzeWithGemini(base64Image, prompt);
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
        throw new Error("No JSON content found in Mixtral response");
    }
}

const client = new ImageAnalysisClient();

app.post('/analyze', upload.single('image'), async (req, res) => {
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


// Add static HTML content
app.get('/check', (req, res) => {
    res.send(`
        <html>
            <body>
                <h1>Image Analysis API</h1>
                <h2>API Endpoints:</h2>
                <ul>
                    <li>POST /analyze - Upload image for analysis</li>
                    <li>GET /status - Check API status</li>
                </ul>
                
                <h2>Test Form:</h2>
                <form action="/analyze" method="post" enctype="multipart/form-data">
                    <p>Select image file: <input type="file" name="image" accept="image/*" required></p>
                    <p>Select model: 
                        <select name="model" required>
                            <option value="GEMINI">GEMINI</option>
                            <option value="MIXTRAL">MIXTRAL</option>
                        </select>
                    </p>
                    <input type="submit" value="Analyze">
                </form>
            </body>
        </html>
    `);
});

// Add status endpoint
app.get('/status', (req, res) => {
    res.json({
        status: "running",
        models: "model",
        version: "1.0.0"
    });
});

// Change server binding
app.listen(port, '0.0.0.0', () => {
    console.log(`Server running on port ${port} (0.0.0.0)`);
});
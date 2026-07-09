import express, { Request, Response } from "express";
import cors from "cors";
import path from "path";
import dotenv from "dotenv";

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;
const MODEL_SERVICE_URL = process.env.MODEL_SERVICE_URL || "http://127.0.0.1:5001/predict";

// Middlewares
app.use(cors());
app.use(express.json());

// API: Predict Review Sentiment & Category
app.post("/predict", async (req: Request, res: Response): Promise<void> => {
  try {
    const { text } = req.body;

    if (!text || typeof text !== "string" || !text.trim()) {
      res.status(400).json({ error: "Review text cannot be empty." });
      return;
    }

    console.log(`[Proxy] Forwarding review for prediction to: ${MODEL_SERVICE_URL}`);

    const response = await fetch(MODEL_SERVICE_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: text.trim() }),
    });

    const data = await response.json();

    if (!response.ok) {
      res.status(response.status).json(data);
      return;
    }

    res.json(data);
  } catch (error: any) {
    console.error("[Error] Connection to model service failed:", error.message);
    res.status(500).json({
      error: `Failed to connect to machine learning model service: ${error.message}`,
    });
  }
});

// Serve React Frontend Static Files (in Production)
const clientDistPath = path.join(__dirname, "../../client/dist");
app.use(express.static(clientDistPath));

// Fallback to React SPA router for all other GET routes
app.get("*", (req: Request, res: Response) => {
  res.sendFile(path.join(clientDistPath, "index.html"));
});

// Start Express Server
app.listen(PORT, () => {
  console.log("=".repeat(60));
  console.log(`  [OK] Node.js Express server running at http://localhost:${PORT}`);
  console.log(`  [OK] Proxied model service target: ${MODEL_SERVICE_URL}`);
  console.log("=".repeat(60));
});

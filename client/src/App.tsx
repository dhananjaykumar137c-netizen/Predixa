import React, { useState, useRef } from 'react';

interface PredictionResult {
  category: string;
  label: number;
  confidence: number;
  rating: number;
  durationMs: number;
  wordCount: number;
}

const DEMO_PILLS = [
  {
    id: 'pill-1',
    label: '✨ Moisturizer Glow (Beauty)',
    text: 'This moisturizer is absolutely amazing for dry skin. It leaves a beautiful glow!',
  },
  {
    id: 'pill-2',
    label: '🔌 Washing Machine Leak (Appliances)',
    text: 'The washing machine stopped working after just 2 months. Very disappointed with the build quality.',
  },
  {
    id: 'pill-3',
    label: '🌸 Elegant Perfume (Beauty)',
    text: 'Smells really fresh and clean. The bottle design is super elegant.',
  },
  {
    id: 'pill-4',
    label: '🔥 Slow Heater (Appliances)',
    text: 'The power cord is too short, and the heating element takes too long to warm up.',
  },
];

export default function App() {
  const [reviewText, setReviewText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  const charLimit = 1000;

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setReviewText(e.target.value.slice(0, charLimit));
  };

  const handleClear = () => {
    setReviewText('');
    setResult(null);
    setError(null);
  };

  const handlePillClick = (text: string) => {
    setReviewText(text);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedText = reviewText.trim();

    if (!trimmedText) {
      setError('Please enter some review text to analyze.');
      return;
    }

    setError(null);
    setResult(null);
    setIsLoading(true);

    const startTime = performance.now();

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: trimmedText }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Server error occurred during prediction.');
      }

      const durationMs = Math.round(performance.now() - startTime);

      setResult({
        category: data.category,
        label: data.label,
        confidence: data.confidence,
        rating: data.rating,
        durationMs,
        wordCount: trimmedText.split(/\s+/).filter((w) => w.length > 0).length,
      });

      // Smooth scroll to results
      setTimeout(() => {
        resultRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  // SVG Circular progress definitions
  const CIRCLE_CIRCUMFERENCE = 339.29; // 2 * PI * 54
  const strokeDashoffset = result
    ? CIRCLE_CIRCUMFERENCE - (result.confidence / 100) * CIRCLE_CIRCUMFERENCE
    : CIRCLE_CIRCUMFERENCE;

  return (
    <>
      {/* SVG Gradient Stroke Definition */}
      <svg style={{ width: 0, height: 0, position: 'absolute' }} aria-hidden="true">
        <defs>
          <linearGradient id="gradient-stroke" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#818cf8" />
            <stop offset="100%" stopColor="#14b8a6" />
          </linearGradient>
        </defs>
      </svg>

      <header>
        <div className="brand-badge" id="brand-badge-id">ADIP Model Dashboard</div>
        <h1 id="main-heading-id">Multi-Task Review Analyzer</h1>
        <p className="subtitle" id="subtitle-id">
          Fine-tuned DistilBERT engine classifying reviews into Product Categories and predicting Star Ratings simultaneously.
        </p>
      </header>

      <main>
        <section className="glass-card" aria-label="Review analyzer controls">
          {error && (
            <div className="error-alert" id="error-box">
              <svg width="20" height="20" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                  clipRule="evenodd"
                ></path>
              </svg>
              <span id="error-message">{error}</span>
            </div>
          )}

          {/* Review Input Section */}
          <div className="input-group">
            <div className="input-header">
              <label htmlFor="review-input" className="input-label">Product Review Text</label>
              <span className="char-counter" id="char-counter">
                {reviewText.length} / {charLimit} characters
              </span>
            </div>
            <textarea
              id="review-input"
              placeholder="Type or paste a product review here..."
              maxLength={charLimit}
              value={reviewText}
              onChange={handleInputChange}
              disabled={isLoading}
            ></textarea>
            {reviewText.length > 0 && !isLoading && (
              <button type="button" className="clear-btn" id="clear-btn" onClick={handleClear}>
                Clear
              </button>
            )}
          </div>

          {/* Quick-fill Pills */}
          <div className="demo-section">
            <h2 className="demo-title" id="demo-title-id">Select a quick example:</h2>
            <div className="demo-pills" id="demo-pills-container">
              {DEMO_PILLS.map((pill) => (
                <button
                  key={pill.id}
                  className="pill"
                  id={pill.id}
                  onClick={() => handlePillClick(pill.text)}
                  disabled={isLoading}
                >
                  {pill.label}
                </button>
              ))}
            </div>
          </div>

          {/* Submit Action */}
          <div className="action-container">
            <button
              type="submit"
              className="submit-btn"
              id="submit-btn"
              onClick={handleSubmit}
              disabled={isLoading}
            >
              {isLoading && <span className="spinner" id="btn-spinner"></span>}
              <span id="btn-text">{isLoading ? 'Analyzing...' : 'Analyze Review'}</span>
            </button>
          </div>

          {/* Loading Skeleton */}
          {isLoading && (
            <div className="loading-skeleton" id="skeleton-loader" style={{ marginTop: '2rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                <div className="res-card">
                  <div className="skeleton-block" style={{ width: '50%', height: '16px', marginBottom: '1.5rem' }}></div>
                  <div className="skeleton-block" style={{ width: '100px', height: '100px', borderRadius: '50%', marginBottom: '1.5rem' }}></div>
                  <div className="skeleton-block" style={{ width: '70%', height: '24px' }}></div>
                </div>
                <div className="res-card">
                  <div className="skeleton-block" style={{ width: '50%', height: '16px', marginBottom: '1.5rem' }}></div>
                  <div className="skeleton-block" style={{ width: '110px', height: '50px', marginBottom: '1.5rem' }}></div>
                  <div className="skeleton-block" style={{ width: '80%', height: '28px' }}></div>
                </div>
              </div>
            </div>
          )}

          {/* Prediction Outputs */}
          {result && !isLoading && (
            <div ref={resultRef} className="result-panel active" id="result-container">
              <div className="results-grid">
                {/* Category Card */}
                <article className="res-card" aria-label="Category prediction">
                  <h3 className="res-label">Predicted Category</h3>
                  <div className="confidence-wrapper">
                    <svg className="confidence-circle" viewBox="0 0 120 120">
                      <circle className="circle-bg" cx="60" cy="60" r="54"></circle>
                      <circle
                        className="circle-progress"
                        id="confidence-circle-progress"
                        cx="60"
                        cy="60"
                        r="54"
                        style={{ strokeDashoffset }}
                      ></circle>
                    </svg>
                    <div className="confidence-text" id="result-confidence">
                      {Math.round(result.confidence)}%
                    </div>
                  </div>
                  <div className="category-name" id="result-category">
                    {result.category}
                  </div>
                </article>

                {/* Rating Card */}
                <article className="res-card" aria-label="Star rating prediction">
                  <h3 className="res-label">Predicted Rating</h3>
                  <div className="stars-container">
                    <div className="stars-value" id="result-rating">
                      {result.rating.toFixed(1)} <span>/ 5.0</span>
                    </div>
                    <div className="stars-outer">
                      ★★★★★
                      <div
                        className="stars-inner"
                        id="result-stars-fill"
                        style={{ width: `${(result.rating / 5.0) * 100}%` }}
                      >
                        ★★★★★
                      </div>
                    </div>
                  </div>
                </article>
              </div>

              <div className="meta-banner">
                <span id="word-count-info">
                  Length: {result.wordCount} word{result.wordCount === 1 ? '' : 's'}
                </span>
                <span id="result-time">Inference: {result.durationMs} ms</span>
              </div>
            </div>
          )}
        </section>
      </main>

      <footer>
        <p>&copy; 2026 ADIP Project. Powered by React, Node.js & PyTorch.</p>
      </footer>
    </>
  );
}

import { useState } from 'react'
import { motion } from 'framer-motion'
import { predictSentiment } from './api'
import InputBox from './components/InputBox'
import ResultCard from './components/ResultCard'
import ImageAnalyzer from './components/ImageAnalyzer'

export default function App() {
  const [activeTab, setActiveTab] = useState('text') // 'text' or 'image'
  const [review, setReview] = useState('')
  const [loading, setLoading] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)

  const onAnalyze = async () => {
    setError(null)
    setPrediction(null)
    if (!review.trim()) {
      setError('Please enter a movie review.')
      return
    }
    try {
      setLoading(true)
      const data = await predictSentiment(review)
      setPrediction(data.prediction)
    } catch (e) {
      setError('Failed to analyze. Is the backend running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="glass rounded-2xl w-full md:w-[600px] p-6"
      >
        <div className="text-center mb-6">
          <h1 className="text-2xl md:text-3xl font-semibold">
            🎬 MovieMood — Sentiment Analyzer
          </h1>
          <p className="text-slate-600 mt-1">Analyze text reviews or images</p>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-2 mb-6 border-b border-slate-200">
          <button
            onClick={() => {
              setActiveTab('text')
              setError(null)
              setPrediction(null)
            }}
            className={`flex-1 py-2 px-4 text-sm font-medium transition-colors ${
              activeTab === 'text'
                ? 'border-b-2 border-slate-900 text-slate-900'
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            📝 Text Review
          </button>
          <button
            onClick={() => {
              setActiveTab('image')
              setError(null)
              setPrediction(null)
            }}
            className={`flex-1 py-2 px-4 text-sm font-medium transition-colors ${
              activeTab === 'image'
                ? 'border-b-2 border-slate-900 text-slate-900'
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            🖼️ Image Emotion
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === 'text' ? (
          <>
            <InputBox value={review} onChange={setReview} />

            <div className="mt-4 flex justify-center">
              <motion.button
                whileTap={{ scale: 0.98 }}
                whileHover={{ scale: 1.02 }}
                onClick={onAnalyze}
                disabled={loading}
                className="px-5 py-3 rounded-lg bg-slate-900 text-white disabled:opacity-50"
              >
                {loading ? 'Analyzing...' : 'Analyze Sentiment'}
              </motion.button>
            </div>

            {error && (
              <div className="mt-4 text-center text-red-600 text-sm">{error}</div>
            )}

            {prediction && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="mt-6"
              >
                <ResultCard prediction={prediction} />
              </motion.div>
            )}
          </>
        ) : (
          <ImageAnalyzer />
        )}
      </motion.div>
    </div>
  )
}


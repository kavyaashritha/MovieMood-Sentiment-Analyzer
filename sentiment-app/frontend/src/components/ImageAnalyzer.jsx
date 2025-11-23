import { useState } from 'react'
import { motion } from 'framer-motion'
import { predictImageSentiment } from '../api'

export default function ImageAnalyzer() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (!file) return

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file (JPG, PNG, etc.)')
      setSelectedFile(null)
      setPreview(null)
      return
    }

    setError(null)
    setSelectedFile(file)
    setPrediction(null)

    // Create preview
    const reader = new FileReader()
    reader.onloadend = () => {
      setPreview(reader.result)
    }
    reader.readAsDataURL(file)
  }

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first.')
      return
    }

    setError(null)
    setPrediction(null)

    try {
      setLoading(true)
      const data = await predictImageSentiment(selectedFile)
      setPrediction(data.prediction)
    } catch (e) {
      setError('Failed to analyze image. Is the backend running on port 8000?')
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setSelectedFile(null)
    setPreview(null)
    setPrediction(null)
    setError(null)
  }

  const getEmotionColor = (emotion) => {
    const emotionLower = emotion?.toLowerCase() || ''
    if (emotionLower.includes('happy') || emotionLower.includes('joy')) {
      return 'bg-yellow-100 border-yellow-300 text-yellow-800'
    } else if (emotionLower.includes('sad')) {
      return 'bg-blue-100 border-blue-300 text-blue-800'
    } else if (emotionLower.includes('angry')) {
      return 'bg-red-100 border-red-300 text-red-800'
    } else if (emotionLower.includes('neutral')) {
      return 'bg-gray-100 border-gray-300 text-gray-800'
    }
    return 'bg-purple-100 border-purple-300 text-purple-800'
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-xl md:text-2xl font-semibold">
          🖼️ Image Emotion Analyzer
        </h2>
        <p className="text-slate-600 mt-1">Upload an image to detect emotions</p>
      </div>

      {/* File Upload */}
      <div className="space-y-4">
        <div className="flex items-center justify-center">
          <label className="cursor-pointer">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="px-4 py-3 rounded-lg bg-slate-100 hover:bg-slate-200 transition-colors border-2 border-dashed border-slate-300 text-center">
              {selectedFile ? selectedFile.name : '📁 Choose Image File'}
            </div>
          </label>
        </div>

        {/* Preview */}
        {preview && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex justify-center"
          >
            <div className="relative">
              <img
                src={preview}
                alt="Preview"
                className="max-w-full max-h-64 rounded-lg border-2 border-slate-200 shadow-md"
              />
              {selectedFile && (
                <button
                  onClick={handleClear}
                  className="absolute top-2 right-2 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs hover:bg-red-600"
                >
                  ×
                </button>
              )}
            </div>
          </motion.div>
        )}

        {/* Analyze Button */}
        <div className="flex justify-center">
          <motion.button
            whileTap={{ scale: 0.98 }}
            whileHover={{ scale: 1.02 }}
            onClick={handleAnalyze}
            disabled={loading || !selectedFile}
            className="px-5 py-3 rounded-lg bg-slate-900 text-white disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Analyzing...' : 'Analyze Emotion'}
          </motion.button>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="text-center text-red-600 text-sm bg-red-50 p-3 rounded-lg border border-red-200">
          {error}
        </div>
      )}

      {/* Prediction Result */}
      {prediction && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="mt-6"
        >
          <div className={`p-4 rounded-lg border-2 ${getEmotionColor(prediction)}`}>
            <div className="text-center">
              <div className="text-sm font-medium mb-1">Predicted Emotion</div>
              <div className="text-2xl font-bold capitalize">{prediction}</div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}


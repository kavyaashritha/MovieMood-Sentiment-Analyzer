import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export async function predictSentiment(review) {
  const res = await axios.post(`${API_BASE}/predict`, { review })
  return res.data
}

export async function predictImageSentiment(imageFile) {
  const formData = new FormData()
  formData.append('file', imageFile)
  const res = await axios.post(`${API_BASE}/predict-image`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return res.data
}


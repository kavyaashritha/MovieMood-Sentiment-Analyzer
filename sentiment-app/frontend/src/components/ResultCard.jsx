export default function ResultCard({ prediction }) {
  const isPositive = prediction?.toLowerCase() === 'positive'
  return (
    <div
      className={`rounded-xl p-4 text-white text-center ${
        isPositive ? 'bg-positive' : 'bg-negative'
      }`}
    >
      <div className="text-sm opacity-90">Prediction</div>
      <div className="text-xl font-semibold mt-1">
        {isPositive ? 'Positive' : 'Negative'}
      </div>
    </div>
  )
}


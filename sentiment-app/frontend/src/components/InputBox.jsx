export default function InputBox({ value, onChange }) {
  return (
    <div className="flex flex-col gap-2">
      <label className="text-sm text-slate-700">Movie Review</label>
      <textarea
        className="w-full min-h-[140px] rounded-xl p-4 border border-slate-200 focus:outline-none focus:ring-2 focus:ring-slate-300 bg-white/70"
        placeholder="Type or paste your movie review here..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  )
}


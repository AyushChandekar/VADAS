import { useState, useEffect, useRef } from 'react'

export default function LiveView() {
  const imgRef = useRef(null)
  const [timestamp, setTimestamp] = useState(Date.now())

  useEffect(() => {
    const timer = setInterval(() => {
      setTimestamp(Date.now())
    }, 5000) // refresh every 5 seconds
    return () => clearInterval(timer)
  }, [])

  return (
    <div className="relative bg-black rounded-lg overflow-hidden">
      <img
        ref={imgRef}
        src={`/api/frame?t=${timestamp}`}
        alt="Live annotated view"
        className="w-full h-auto"
        onError={(e) => {
          e.target.style.opacity = 0.3
        }}
        onLoad={(e) => {
          e.target.style.opacity = 1
        }}
      />
      <div className="absolute top-2 right-2 bg-black/60 px-2 py-1 rounded text-xs text-green-400">
        LIVE
      </div>
    </div>
  )
}

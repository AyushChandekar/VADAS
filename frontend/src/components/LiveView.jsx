import { useState, useEffect, useRef } from 'react'

export default function LiveView() {
  const [devices, setDevices] = useState([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const [timestamp, setTimestamp] = useState(Date.now())

  // Get available camera devices
  useEffect(() => {
    async function getDevices() {
      try {
        // Request permission first to get labels
        await navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
          stream.getTracks().forEach(t => t.stop())
        })
        
        const devs = await navigator.mediaDevices.enumerateDevices()
        const videoDevs = devs.filter(d => d.kind === 'videoinput')
        setDevices(videoDevs)
        if (videoDevs.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoDevs[0].deviceId)
        }
      } catch (err) {
        console.error("Error enumerating devices:", err)
        setError("Camera access denied or not available")
      }
    }
    getDevices()
    
    const handleDeviceChange = () => getDevices()
    navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange)
    return () => navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange)
  }, [])

  // Handle camera stream and WebSocket
  useEffect(() => {
    if (!selectedDeviceId || !isStreaming) {
      if (wsRef.current) wsRef.current.close()
      return
    }

    let stream = null
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    // For HF Spaces, we might need to handle the path correctly
    const host = window.location.host
    const wsUrl = `${protocol}//${host}/api/ws/stream`
    
    console.log(`Attempting WebSocket connection: ${wsUrl}`)
    try {
      wsRef.current = new WebSocket(wsUrl)
      
      wsRef.current.onopen = () => {
        console.log("✅ WebSocket Connected")
        setError(null)
      }
      wsRef.current.onclose = (e) => {
        console.log("❌ WebSocket Closed:", e.code, e.reason)
      }
      wsRef.current.onerror = (err) => {
        console.error("⚠️ WebSocket Error:", err)
        setError("Connection to processing server failed. Please check if the backend is running.")
      }
    } catch (e) {
      console.error("Failed to create WebSocket:", e)
    }

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            deviceId: { exact: selectedDeviceId },
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
        setError(null)
      } catch (err) {
        console.error("Error accessing camera:", err)
        setError(`Failed to start camera: ${err.message}`)
        setIsStreaming(false)
      }
    }

    startCamera()

    const interval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN && videoRef.current && canvasRef.current) {
        const canvas = canvasRef.current
        const video = videoRef.current
        
        if (video.videoWidth > 0) {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
          const ctx = canvas.getContext('2d')
          ctx.drawImage(video, 0, 0)
          
          canvas.toBlob((blob) => {
            if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(blob)
            }
          }, 'image/jpeg', 0.6) // Compressed JPEG
        } else {
          console.warn("Video width is 0, skipping frame")
        }
      } else if (isStreaming) {
        if (!wsRef.current) console.log("WS not initialized")
        else if (wsRef.current.readyState !== WebSocket.OPEN) console.log("WS readyState:", wsRef.current.readyState)
      }
      setTimestamp(Date.now())
    }, 150) // ~6-7 FPS for stability

    return () => {
      clearInterval(interval)
      if (stream) {
        stream.getTracks().forEach(t => t.stop())
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [selectedDeviceId, isStreaming])

  return (
    <div className="space-y-4">
      {/* Debug Info (Only visible if there's an issue) */}
      <div className="text-[10px] text-gray-500 flex gap-4 justify-center">
        <span>WS: {wsRef.current?.readyState === 1 ? '✅' : '❌'}</span>
        <span>Devices: {devices.length}</span>
        <span>Streaming: {isStreaming ? 'ON' : 'OFF'}</span>
      </div>

      {/* Camera Selection & Controls */}
      <div className="flex flex-col sm:flex-row items-center gap-4 bg-gray-800 p-4 rounded-lg shadow-lg">
        <div className="flex-1 w-full">
          <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wider font-semibold">
            Select Input Source
          </label>
          <select
            value={selectedDeviceId}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
            disabled={isStreaming}
            className="w-full bg-gray-700 text-white px-3 py-2 rounded border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 transition-all"
          >
            {devices.length === 0 && <option>No cameras found</option>}
            {devices.map(device => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label || `Camera ${device.deviceId.slice(0, 5)}...`}
              </option>
            ))}
          </select>
        </div>
        
        <button
          onClick={() => setIsStreaming(!isStreaming)}
          disabled={devices.length === 0}
          className={`w-full sm:w-auto px-8 py-2 rounded-md font-bold text-white shadow-md transition-all active:scale-95 ${
            isStreaming 
              ? 'bg-red-500 hover:bg-red-600' 
              : 'bg-blue-600 hover:bg-blue-700'
          } disabled:bg-gray-600 disabled:cursor-not-allowed`}
        >
          {isStreaming ? 'Stop System' : 'Start System'}
        </button>
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-500 text-red-200 px-4 py-2 rounded-lg text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* Viewport */}
      <div className="relative bg-black rounded-xl overflow-hidden shadow-2xl aspect-video border-2 border-gray-800">
        {/* Hidden processing elements */}
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted 
          style={{ position: 'absolute', width: '1px', height: '1px', opacity: 0.01 }} 
        />
        <canvas ref={canvasRef} className="hidden" />

        {/* Live Output */}
        <img
          src={`/api/frame?t=${timestamp}`}
          alt="Processed Stream"
          className="w-full h-full object-contain"
          onError={(e) => {
            e.target.style.opacity = 0.2
          }}
          onLoad={(e) => {
            e.target.style.opacity = 1
          }}
        />

        {/* Overlay Info */}
        <div className="absolute top-4 left-4 flex flex-col gap-2">
          {isStreaming ? (
            <div className="flex items-center gap-2 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full border border-green-500/50">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-xs font-bold text-green-400 tracking-wide uppercase">Processing</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full border border-gray-500/50">
              <div className="w-2 h-2 bg-gray-500 rounded-full" />
              <span className="text-xs font-bold text-gray-400 tracking-wide uppercase">Standby</span>
            </div>
          )}
        </div>

        {!isStreaming && !error && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm">
            <p className="text-gray-300 font-medium">Click "Start System" to begin live analysis</p>
          </div>
        )}
      </div>
      
      <p className="text-[10px] text-gray-500 text-center uppercase tracking-[0.2em]">
        Neural Network Inference Pipeline • Real-time Indian Road Scene Analysis
      </p>
    </div>
  )
}

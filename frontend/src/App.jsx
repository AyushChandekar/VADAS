import { useState, useEffect, useRef } from 'react'
import VideoUploader from './components/VideoUploader'
import StatusBar from './components/StatusBar'
import { usePolling } from './hooks/usePolling'

export default function App() {
  const [appState, setAppState] = useState('IDLE') // IDLE, UPLOADED, PROCESSING, COMPLETED
  const { data: procStatus } = usePolling('/api/process/status', appState === 'PROCESSING' ? 1000 : 5000)
  
  const videoRef1 = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isStreaming, setIsStreaming] = useState(false)

  const handleUploadSuccess = () => {
    setAppState('UPLOADED')
  }

  const startProcessing = async () => {
    setAppState('PROCESSING')
    setIsStreaming(true)
    videoRef1.current?.play()
  }

  // Dual Player Sync Logic
  const togglePlay = () => {
    const nextPlay = !isPlaying
    setIsPlaying(nextPlay)
    if (nextPlay) {
      videoRef1.current?.play()
      setIsStreaming(true)
    } else {
      videoRef1.current?.pause()
      setIsStreaming(false)
    }
  }

  const handleTimeUpdate = () => {
    if (videoRef1.current) {
      setCurrentTime(videoRef1.current.currentTime)
    }
  }

  const handleSeek = (e) => {
    const time = parseFloat(e.target.value)
    setCurrentTime(time)
    if (videoRef1.current) videoRef1.current.currentTime = time
    // Reset stream on seek
    setIsStreaming(false)
    setTimeout(() => setIsStreaming(true), 100)
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6 font-sans selection:bg-indigo-500/30">
      <header className="mb-10 text-center animate-in fade-in slide-in-from-top duration-700">
        <h1 className="text-5xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-blue-500">
          VADAS-INDIA
        </h1>
        <p className="text-gray-500 mt-2 font-medium tracking-wide">Vehicle Autonomous Driving Assistance System</p>
      </header>

      <div className="max-w-7xl mx-auto space-y-8">
        <StatusBar />

        {appState === 'IDLE' && (
          <div className="bg-gray-900/50 p-10 rounded-3xl shadow-2xl border border-gray-800/50 max-w-xl mx-auto backdrop-blur-sm animate-in zoom-in duration-500">
            <div className="flex items-center gap-4 mb-8">
              <div className="w-10 h-10 rounded-full bg-indigo-500/10 flex items-center justify-center text-indigo-400 font-bold border border-indigo-500/20">1</div>
              <h2 className="text-2xl font-bold">Upload Source Video</h2>
            </div>
            <VideoUploader onUploadSuccess={handleUploadSuccess} />
          </div>
        )}

        {appState === 'UPLOADED' && (
          <div className="bg-gray-900/50 p-10 rounded-3xl shadow-2xl border border-gray-800/50 text-center max-w-xl mx-auto backdrop-blur-sm animate-in zoom-in duration-500">
             <div className="w-20 h-20 bg-green-500/10 rounded-full flex items-center justify-center mx-auto mb-6 border border-green-500/20">
              <svg className="w-10 h-10 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h2 className="text-2xl font-bold mb-3 text-green-400">Video Uploaded!</h2>
            <p className="text-gray-400 mb-10 text-sm leading-relaxed">
              Ready for real-time AI analysis.
            </p>
            <button
              onClick={startProcessing}
              className="w-full py-5 bg-indigo-600 hover:bg-indigo-500 rounded-2xl font-black text-xl transition-all transform hover:scale-[1.02] active:scale-95 shadow-xl shadow-indigo-600/20 flex items-center justify-center gap-3"
            >
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
              Start Real-Time Analysis
            </button>
          </div>
        )}

        {(appState === 'PROCESSING' || appState === 'COMPLETED') && (
          <div className="space-y-8 animate-in fade-in duration-1000">
            <div className="grid lg:grid-cols-2 gap-6">
              <div className="group space-y-3">
                <div className="flex items-center justify-between px-2">
                  <span className="text-xs font-black text-gray-500 uppercase tracking-[0.2em]">Original Stream</span>
                </div>
                <div className="aspect-video bg-black rounded-3xl overflow-hidden border border-gray-800 shadow-2xl group-hover:border-gray-700 transition-colors">
                  <video 
                    ref={videoRef1}
                    src="/api/video/original" 
                    className="w-full h-full object-contain"
                    onTimeUpdate={handleTimeUpdate}
                    onLoadedMetadata={(e) => setDuration(e.target.duration)}
                    onPlay={() => setIsPlaying(true)}
                    onPause={() => setIsPlaying(false)}
                  />
                </div>
              </div>
              
              <div className="group space-y-3">
                <div className="flex items-center justify-between px-2">
                  <span className="text-xs font-black text-indigo-500 uppercase tracking-[0.2em]">AI Real-Time Analysis</span>
                </div>
                <div className="aspect-video bg-black rounded-3xl overflow-hidden border border-indigo-500/20 shadow-2xl shadow-indigo-500/5 group-hover:border-indigo-500/40 transition-colors flex items-center justify-center">
                  {isStreaming ? (
                    <img 
                      src={`/api/stream?t=${currentTime}`} 
                      className="w-full h-full object-contain"
                      alt="AI Analysis"
                    />
                  ) : (
                    <div className="text-gray-600 animate-pulse font-mono uppercase tracking-widest text-xs">Waiting for stream...</div>
                  )}
                </div>
              </div>
            </div>

            {/* Custom Unified Controller */}
            <div className="bg-gray-900/80 backdrop-blur-md p-8 rounded-[2.5rem] border border-gray-800 shadow-3xl max-w-4xl mx-auto">
              <div className="flex items-center gap-8">
                <button 
                  onClick={togglePlay}
                  className="w-16 h-16 flex items-center justify-center bg-indigo-600 hover:bg-indigo-500 text-white rounded-full transition-all transform hover:scale-105 active:scale-95 shadow-xl shadow-indigo-600/30"
                >
                  {isPlaying ? (
                    <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
                  ) : (
                    <svg className="w-8 h-8 ml-1" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                  )}
                </button>

                <div className="flex-1 space-y-3">
                  <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div 
                      className="absolute top-0 left-0 h-full bg-indigo-500 transition-all duration-100" 
                      style={{ width: `${(currentTime / duration) * 100}%` }}
                    />
                    <input 
                      type="range"
                      min="0"
                      max={duration || 0}
                      step="0.01"
                      value={currentTime}
                      onChange={handleSeek}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                    />
                  </div>
                    <div className="flex justify-between items-center text-[11px] font-mono font-bold text-gray-500 tracking-tighter">
                    <span className="text-gray-300">{formatTime(currentTime)}</span>
                    <span className="bg-gray-800 px-2 py-0.5 rounded text-gray-400">AI_STREAM_ACTIVE</span>
                    <span>{formatTime(duration)}</span>
                  </div>
                </div>
                
                <div className="h-10 w-px bg-gray-800 mx-2" />

                <button 
                  onClick={() => {setAppState('IDLE'); setIsStreaming(false);}}
                  className="px-6 py-3 text-xs font-black text-gray-500 hover:text-white hover:bg-white/5 rounded-2xl transition-all uppercase tracking-widest"
                >
                  Reset
                </button>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4 max-w-4xl mx-auto opacity-50 grayscale hover:grayscale-0 transition-all duration-500">
               <LegendItem color="bg-green-500" label="Drivable Zone" />
               <LegendItem color="bg-red-500" label="GRU Path" />
               <LegendItem color="bg-blue-500" label="Obstacles" />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function LegendItem({ color, label }) {
  return (
    <div className="flex items-center gap-2 justify-center py-2 px-4 rounded-xl border border-gray-800/50 bg-gray-900/30">
      <div className={`w-2 h-2 rounded-full ${color} shadow-[0_0_10px_rgba(0,0,0,0.5)]`} />
      <span className="text-[10px] font-black uppercase tracking-widest text-gray-400">{label}</span>
    </div>
  )
}

function formatTime(seconds) {
  if (!seconds || isNaN(seconds)) return '00:00'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
}

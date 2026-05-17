import { usePolling } from '../hooks/usePolling'

const ACTION_STYLES = {
  'DRIVE FORWARD': { bg: 'bg-green-600', text: 'text-white' },
  'SLOW DOWN':     { bg: 'bg-yellow-500', text: 'text-black' },
  'STOP':          { bg: 'bg-red-600', text: 'text-white' },
  'TURN LEFT':     { bg: 'bg-blue-500', text: 'text-white' },
  'TURN RIGHT':    { bg: 'bg-blue-500', text: 'text-white' },
  'HORN':          { bg: 'bg-yellow-400', text: 'text-black' },
  'WIPER':         { bg: 'bg-gray-500', text: 'text-white' },
  'OFFLINE':       { bg: 'bg-gray-700', text: 'text-gray-400' },
  'INITIALIZING':  { bg: 'bg-gray-700', text: 'text-gray-400' },
}

export default function ActionDisplay() {
  const { data } = usePolling('/api/status', 2000)

  const action = data?.action || 'OFFLINE'
  const reason = data?.reason || 'Waiting for connection...'
  const confidence = data?.confidence || 0
  const fps = data?.fps || 0
  const style = ACTION_STYLES[action] || ACTION_STYLES['OFFLINE']

  return (
    <div className={`rounded-lg p-4 ${style.bg} ${style.text}`}>
      <div className="text-3xl font-bold tracking-wide">{action}</div>
      <div className="text-sm mt-1 opacity-80">{reason}</div>
      <div className="flex gap-4 mt-3 text-xs opacity-70">
        <span>Confidence: {(confidence * 100).toFixed(0)}%</span>
        <span>FPS: {fps}</span>
      </div>
    </div>
  )
}

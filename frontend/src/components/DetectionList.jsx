import { usePolling } from '../hooks/usePolling'

const PROXIMITY_COLORS = {
  NEAR: 'text-red-400',
  MEDIUM: 'text-yellow-400',
  FAR: 'text-green-400',
}

export default function DetectionList() {
  const { data } = usePolling('/api/detections', 3000)

  const detections = data || []

  return (
    <div className="bg-gray-800 rounded-lg p-4 h-full">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Detected Objects ({detections.length})
      </h2>

      {detections.length === 0 ? (
        <p className="text-gray-500 text-sm">No detections</p>
      ) : (
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {detections.map((det, i) => (
            <div key={i} className="flex items-center justify-between text-sm bg-gray-700/50 rounded px-3 py-2">
              <div>
                <span className="font-medium">{det.class_name}</span>
                <span className="text-gray-400 ml-2">
                  {(det.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <span className={`text-xs font-bold ${PROXIMITY_COLORS[det.proximity] || 'text-gray-400'}`}>
                {det.proximity}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

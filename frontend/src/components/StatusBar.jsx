import { usePolling } from '../hooks/usePolling'

export default function StatusBar() {
  const { data, error } = usePolling('/api/health', 5000)

  const pipelineLoaded = data?.pipeline_loaded || false
  const cameraConnected = data?.camera_connected || false
  const gpuAvailable = data?.gpu_available || false
  const gpuName = data?.gpu_name || 'N/A'

  return (
    <div className="bg-gray-800 rounded-lg px-4 py-3 flex flex-wrap gap-4 text-xs">
      <StatusDot label="Pipeline" ok={pipelineLoaded} />
      <StatusDot label="Camera" ok={cameraConnected} />
      <StatusDot label="GPU" ok={gpuAvailable} detail={gpuName} />
      {error && (
        <span className="text-red-400">Backend offline</span>
      )}
      <span className="ml-auto text-gray-500">
        {new Date().toLocaleTimeString()}
      </span>
    </div>
  )
}

function StatusDot({ label, ok, detail }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className={`w-2 h-2 rounded-full ${ok ? 'bg-green-400' : 'bg-red-400'}`} />
      <span className="text-gray-300">{label}</span>
      {detail && ok && (
        <span className="text-gray-500">({detail})</span>
      )}
    </div>
  )
}

import LiveView from './components/LiveView'
import ActionDisplay from './components/ActionDisplay'
import DetectionList from './components/DetectionList'
import StatusBar from './components/StatusBar'
import VideoUploader from './components/VideoUploader'

export default function App() {
  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-4">
        <div>
          <h1 className="text-2xl font-bold">VADAS-India</h1>
          <p className="text-sm text-gray-400">
            Vehicle Autonomous Driving Assistance System
          </p>
        </div>
      </header>

      <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <div className="space-y-4">
          <VideoUploader />
          <StatusBar />
          <LiveView />
          <ActionDisplay />
        </div>

        <div className="space-y-4">
          <DetectionList />
        </div>
      </div>
    </div>
  )
}

import LiveView from './components/LiveView'
import ActionDisplay from './components/ActionDisplay'
import DetectionList from './components/DetectionList'
import StatusBar from './components/StatusBar'

export default function App() {
  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      {/* Header */}
      <header className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold">VADAS-India</h1>
          <p className="text-sm text-gray-400">
            Vehicle Autonomous Driving Assistance System
          </p>
        </div>
      </header>

      {/* Status Bar */}
      <StatusBar />

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-4">
        {/* Live View — takes 2 columns */}
        <div className="lg:col-span-2 space-y-4">
          <LiveView />
          <ActionDisplay />
        </div>

        {/* Right Sidebar */}
        <div className="space-y-4">
          <DetectionList />
        </div>
      </div>
    </div>
  )
}

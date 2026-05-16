import { useState } from 'react'

export default function VideoUploader({ onUploadSuccess }) {
  const [file, setFile] = useState(null)
  const [message, setMessage] = useState('Select a video to begin analysis.')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(event) {
    event.preventDefault()
    if (!file) {
      setMessage('Please choose a video file first.')
      return
    }

    if (file.size > 40 * 1024 * 1024) {
      setMessage('File too large. Maximum size is 40MB.')
      return
    }

    const formData = new FormData()
    formData.append('file', file)

    setLoading(true)
    setMessage('Uploading to AI server...')

    try {
      const res = await fetch('/api/upload_video', {
        method: 'POST',
        body: formData,
      })

      let data
      const contentType = res.headers.get('content-type')
      if (contentType && contentType.includes('application/json')) {
        data = await res.json()
      } else {
        const text = await res.text()
        console.error('Non-JSON response:', text)
        throw new Error(`Server returned ${res.status}: ${text.slice(0, 100)}...`)
      }

      if (!res.ok) {
        throw new Error(data.detail || 'Upload failed')
      }

      setMessage('Upload complete!')
      setFile(null)
      if (onUploadSuccess) onUploadSuccess()
    } catch (error) {
      setMessage(`Upload failed: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="group relative border-2 border-dashed border-gray-700 hover:border-indigo-500/50 rounded-2xl p-8 transition-all bg-gray-900/50">
          <input
            type="file"
            accept="video/*"
            onChange={(event) => {
              const selectedFile = event.target.files?.[0] ?? null
              setFile(selectedFile)
              if (selectedFile) setMessage(`Selected: ${selectedFile.name}`)
            }}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
          />
          <div className="text-center space-y-2">
            <div className="w-12 h-12 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
              <svg className="w-6 h-6 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <p className="text-sm font-medium text-gray-300">Click to browse or drag & drop</p>
            <p className="text-xs text-gray-500">MP4, MOV up to 40MB</p>
          </div>
        </div>

        {file && (
          <button
            type="submit"
            disabled={loading}
            className="w-full py-4 bg-indigo-600 hover:bg-indigo-500 rounded-xl font-bold text-sm transition-all shadow-lg shadow-indigo-500/20 disabled:opacity-50"
          >
            {loading ? 'Uploading…' : 'Upload and Continue'}
          </button>
        )}
      </form>
      <p className="text-center text-xs text-gray-500">{message}</p>
    </div>
  )
}

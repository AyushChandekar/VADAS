import { useState } from 'react'

export default function VideoUploader() {
  const [file, setFile] = useState(null)
  const [message, setMessage] = useState('Upload a video to start inference.')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(event) {
    event.preventDefault()
    if (!file) {
      setMessage('Please choose a video file first.')
      return
    }

    const formData = new FormData()
    formData.append('file', file)

    setLoading(true)
    setMessage('Uploading video...')

    try {
      const res = await fetch('/api/upload_video', {
        method: 'POST',
        body: formData,
      })

      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.detail || 'Upload failed')
      }

      setMessage('Upload complete. Inference will start shortly.')
      setFile(null)
    } catch (error) {
      setMessage(`Upload failed: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="mb-3 text-sm text-gray-300 font-semibold">Video upload</div>
      <form onSubmit={handleSubmit} className="space-y-3">
        <input
          type="file"
          accept="video/*"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          className="block w-full text-sm text-gray-200 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-gray-700 file:text-white"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
        >
          {loading ? 'Uploading…' : 'Upload video'}
        </button>
      </form>
      <p className="mt-3 text-xs text-gray-400">{message}</p>
    </div>
  )
}

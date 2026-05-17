import { useState, useEffect, useRef } from 'react'

export function usePolling(url, intervalMs = 5000) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const timer = useRef(null)

  useEffect(() => {
    let active = true

    async function fetchData() {
      try {
        const res = await fetch(url)
        if (res.ok) {
          const json = await res.json()
          if (active) {
            setData(json)
            setError(null)
          }
        }
      } catch (err) {
        if (active) setError(err.message)
      }
    }

    fetchData()
    timer.current = setInterval(fetchData, intervalMs)

    return () => {
      active = false
      clearInterval(timer.current)
    }
  }, [url, intervalMs])

  return { data, error }
}

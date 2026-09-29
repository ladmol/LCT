import { useEffect, useState } from 'react'
import type { ChangeEvent } from 'react'
import './App.css'

type Health = {
  status: string
  model_version: string
  embedding_dim: number
  weights_available: boolean
}

type Result = {
  model_version: string
  embedding_dim: number
  inference_ms: number
  detections: Array<{ embedding_norm: number; embedding: number[] }>
}

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [result, setResult] = useState<Result | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch('/api/health')
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  function chooseImage(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null
    if (preview) URL.revokeObjectURL(preview)
    setFile(selected)
    setResult(null)
    setError('')
    if (selected) setPreview(URL.createObjectURL(selected))
  }

  async function createEmbedding() {
    if (!file || !size.width || !size.height) return
    setLoading(true)
    setError('')
    setResult(null)
    const form = new FormData()
    form.append('image', file)
    form.append(
      'meta',
      JSON.stringify({
        event_id: `demo-${Date.now()}`,
        camera_id: 'demo-ui',
        captured_at: new Date().toISOString(),
        detections: [
          { detection_id: 'vehicle-1', bbox_xywh: [0, 0, size.width, size.height] },
        ],
      }),
    )
    try {
      const response = await fetch('/api/extract', { method: 'POST', body: form })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail?.message ?? 'Ошибка API')
      setResult(payload)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Не удалось получить признак')
    } finally {
      setLoading(false)
    }
  }

  const vector = result?.detections[0]?.embedding ?? []

  return (
    <main>
      <header>
        <div className="brand">FALCON <span>VISION</span></div>
        <div className={`status ${health?.weights_available ? 'online' : ''}`}>
          <i /> {health?.weights_available ? 'Модель готова' : 'API недоступен'}
        </div>
      </header>

      <section className="intro">
        <p className="eyebrow">VEHICLE RE-IDENTIFICATION</p>
        <h1>Цифровой признак<br />автомобиля</h1>
        <p className="lead">Сравниваем автомобили по внешнему виду без использования государственных номеров.</p>
      </section>

      <section className="workspace">
        <div className="upload">
          {preview ? (
            <img src={preview} alt="Загруженный автомобиль" onLoad={(event) => setSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} />
          ) : (
            <div className="placeholder"><b>Добавьте фотографию</b><span>JPEG или PNG, до 20 МБ</span></div>
          )}
          <label className="file-button">{file ? 'Заменить фото' : 'Выбрать фото'}<input type="file" accept="image/jpeg,image/png" onChange={chooseImage} /></label>
        </div>

        <div className="panel">
          <div className="panel-title"><span>01</span><div><b>Извлечение признака</b><small>Ансамбль ConvNeXt + ResNet50</small></div></div>
          <div className="facts">
            <div><span>Размерность</span><b>{health?.embedding_dim ?? 2816}</b></div>
            <div><span>Версия</span><b>{health?.model_version ?? 'ожидание API'}</b></div>
            <div><span>Область</span><b>{size.width ? `${size.width} × ${size.height}` : '—'}</b></div>
          </div>
          <button className="run" disabled={!file || loading || !health?.weights_available} onClick={createEmbedding}>
            {loading ? 'Модель обрабатывает…' : 'Создать цифровой признак'}
          </button>
          {error && <p className="error">{error}</p>}
          {result && (
            <div className="result">
              <div className="success">Готово за {result.inference_ms.toFixed(0)} мс</div>
              <div className="result-row"><span>Норма вектора</span><b>{result.detections[0].embedding_norm.toFixed(6)}</b></div>
              <div className="result-row"><span>Первые значения</span><code>{vector.slice(0, 6).map((value) => value.toFixed(4)).join(' · ')}</code></div>
            </div>
          )}
        </div>
      </section>
    </main>
  )
}

export default App

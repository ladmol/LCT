import { useEffect, useState } from 'react'
import type { ChangeEvent, DragEvent } from 'react'
import './App.css'

const REFUSAL_THRESHOLD = 0.5051871538
const MAX_IMAGE_BYTES = 20 * 1024 * 1024

type Health = {
  status: string
  model_version: string
  embedding_dim: number
  weights_available: boolean
}

type ExtractResult = {
  model_version: string
  embedding_dim: number
  inference_ms: number
  detections: Array<{ embedding_norm: number; embedding: number[] }>
}

type SelectedImage = {
  file: File
  preview: string
  width: number
  height: number
}

type Comparison = {
  similarity: number
  score: number
  sameVehicle: boolean
  inferenceMs: number
  modelVersion: string
  embeddingDim: number
}

type UploadCardProps = {
  number: string
  title: string
  image: SelectedImage | null
  onFile: (file: File) => void
  onSize: (width: number, height: number) => void
}

function UploadCard({ number, title, image, onFile, onSize }: UploadCardProps) {
  const [dragging, setDragging] = useState(false)

  function chooseImage(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (file) onFile(file)
    event.target.value = ''
  }

  function dropImage(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setDragging(false)
    const file = event.dataTransfer.files?.[0]
    if (file) onFile(file)
  }

  return (
    <article className="upload-card">
      <div className="upload-title"><span>{number}</span><b>{title}</b></div>
      <div
        className={`upload ${dragging ? 'dragging' : ''}`}
        onDragEnter={(event) => { event.preventDefault(); setDragging(true) }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragging(false)
        }}
        onDrop={dropImage}
      >
        {image ? (
          <img
            src={image.preview}
            alt={title}
            onLoad={(event) => onSize(event.currentTarget.naturalWidth, event.currentTarget.naturalHeight)}
          />
        ) : (
          <div className="placeholder">
            <b>Перетащите фотографию сюда</b>
            <span>или выберите JPEG/PNG до 20 МБ</span>
          </div>
        )}
        <label className="file-button">
          {image ? 'Заменить фото' : 'Выбрать фото'}
          <input type="file" accept="image/jpeg,image/png" onChange={chooseImage} />
        </label>
      </div>
      <div className="file-meta">
        <span>{image?.file.name ?? 'Файл не выбран'}</span>
        <b>{image?.width ? `${image.width} × ${image.height}` : '—'}</b>
      </div>
    </article>
  )
}

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [first, setFirst] = useState<SelectedImage | null>(null)
  const [second, setSecond] = useState<SelectedImage | null>(null)
  const [comparison, setComparison] = useState<Comparison | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch('/api/health')
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  useEffect(() => () => {
    if (first?.preview) URL.revokeObjectURL(first.preview)
  }, [first?.preview])

  useEffect(() => () => {
    if (second?.preview) URL.revokeObjectURL(second.preview)
  }, [second?.preview])

  function selectFile(file: File, setter: (value: SelectedImage) => void) {
    setComparison(null)
    setError('')
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
      setError('Поддерживаются только JPEG и PNG.')
      return
    }
    if (file.size > MAX_IMAGE_BYTES) {
      setError('Размер изображения не должен превышать 20 МБ.')
      return
    }
    setter({ file, preview: URL.createObjectURL(file), width: 0, height: 0 })
  }

  async function extract(image: SelectedImage, slot: string): Promise<ExtractResult> {
    const form = new FormData()
    form.append('image', image.file)
    form.append('meta', JSON.stringify({
      event_id: `comparison-${slot}-${Date.now()}`,
      camera_id: `demo-${slot}`,
      captured_at: new Date().toISOString(),
      detections: [{
        detection_id: `vehicle-${slot}`,
        bbox_xywh: [0, 0, image.width, image.height],
      }],
    }))
    const response = await fetch('/api/extract', { method: 'POST', body: form })
    const payload = await response.json()
    if (!response.ok) throw new Error(payload.detail?.message ?? `Не удалось обработать фото ${slot}`)
    return payload as ExtractResult
  }

  async function compareImages() {
    if (!first || !second || !first.width || !second.width) return
    setLoading(true)
    setError('')
    setComparison(null)
    try {
      const [left, right] = await Promise.all([
        extract(first, 'A'),
        extract(second, 'B'),
      ])
      if (left.model_version !== right.model_version || left.embedding_dim !== right.embedding_dim) {
        throw new Error('Признаки получены разными версиями модели и не могут сравниваться.')
      }
      const leftVector = left.detections[0]?.embedding
      const rightVector = right.detections[0]?.embedding
      if (!leftVector || !rightVector || leftVector.length !== rightVector.length) {
        throw new Error('Модель вернула несовместимые признаки.')
      }
      const similarity = leftVector.reduce((sum, value, index) => sum + value * rightVector[index], 0)
      setComparison({
        similarity,
        score: Math.max(0, Math.min(1, (similarity + 1) / 2)),
        sameVehicle: similarity >= REFUSAL_THRESHOLD,
        inferenceMs: left.inference_ms + right.inference_ms,
        modelVersion: left.model_version,
        embeddingDim: left.embedding_dim,
      })
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Не удалось сравнить изображения')
    } finally {
      setLoading(false)
    }
  }

  const ready = Boolean(first?.width && second?.width && health?.weights_available)

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
        <h1>Один автомобиль<br />или два?</h1>
        <p className="lead">Загрузите два снимка. Модель сравнит автомобили по внешнему виду без использования государственных номеров.</p>
      </section>

      <section className="compare-grid">
        <UploadCard
          number="01"
          title="Первый снимок"
          image={first}
          onFile={(file) => selectFile(file, setFirst)}
          onSize={(width, height) => setFirst((value) => value ? { ...value, width, height } : value)}
        />
        <UploadCard
          number="02"
          title="Второй снимок"
          image={second}
          onFile={(file) => selectFile(file, setSecond)}
          onSize={(width, height) => setSecond((value) => value ? { ...value, width, height } : value)}
        />
      </section>

      <section className="panel comparison-panel">
        <div className="panel-title">
          <span>03</span>
          <div><b>Сопоставление цифровых признаков</b><small>ConvNeXt Tiny + ResNet50 · cosine similarity</small></div>
        </div>
        <div className="comparison-layout">
          <div>
            <div className="facts">
              <div><span>Размерность</span><b>{health?.embedding_dim ?? 2816}</b></div>
              <div><span>Версия модели</span><b>{health?.model_version ?? 'ожидание API'}</b></div>
              <div><span>Порог решения</span><b>{REFUSAL_THRESHOLD.toFixed(3)}</b></div>
            </div>
            <button className="run" disabled={!ready || loading} onClick={compareImages}>
              {loading ? 'Модель сравнивает…' : 'Сравнить автомобили'}
            </button>
            {error && <p className="error">{error}</p>}
          </div>

          <div className={`verdict ${comparison ? (comparison.sameVehicle ? 'match' : 'different') : ''}`}>
            {comparison ? (
              <>
                <span>Результат сравнения</span>
                <strong>{comparison.sameVehicle ? 'Вероятно, один автомобиль' : 'Вероятно, разные автомобили'}</strong>
                <div className="similarity">{comparison.similarity.toFixed(4)}</div>
                <small>Косинусная близость · условный score {(comparison.score * 100).toFixed(1)}%, не вероятность</small>
                <p>{comparison.embeddingDim} чисел · {comparison.inferenceMs.toFixed(0)} мс суммарно</p>
              </>
            ) : (
              <div className="empty-verdict"><b>Результат появится здесь</b><span>Добавьте два изображения и запустите сравнение</span></div>
            )}
          </div>
        </div>
      </section>
    </main>
  )
}

export default App

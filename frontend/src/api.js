import axios from 'axios'

const http = axios.create({ baseURL: '/api/v1' })

export const api = {
  health: () => http.get('/health').then(r => r.data),

  chat: (payload) => http.post('/chat', payload).then(r => r.data),

  search: (query, k = 4, withScore = false) =>
    http.post('/search', { query, k, with_score: withScore }).then(r => r.data),

  ingestTexts: (texts, metadatas) =>
    http.post('/ingest', { texts, metadatas }).then(r => r.data),

  ingestFile: (file, onProgress) => {
    const form = new FormData()
    form.append('file', file)
    return http.post('/ingest/file', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: e => onProgress && onProgress(Math.round(e.loaded * 100 / e.total)),
    }).then(r => r.data)
  },

  deleteDocuments: (ids) =>
    http.delete('/documents', { data: { ids } }).then(r => r.data),
}

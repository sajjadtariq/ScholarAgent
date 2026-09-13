'use client';

import {
  type ChangeEvent,
  type FormEvent,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from 'react';

import {
  askQuestion,
  getDocuments,
  pdfUrl,
  uploadPdf,
  type ChatResult,
  type Citation,
  type PdfDocument,
} from '@/lib/api';

type Exchange = { question: string; result: ChatResult };

export default function Home() {
  const fileInput = useRef<HTMLInputElement>(null);
  const [documents, setDocuments] = useState<PdfDocument[]>([]);
  const [activeDocumentId, setActiveDocumentId] = useState('');
  const [activePage, setActivePage] = useState(1);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState<'upload' | 'chat' | null>(null);
  const [uploadName, setUploadName] = useState('');
  const [uploadProgress, setUploadProgress] = useState({
    current: 0,
    total: 0,
  });
  const [error, setError] = useState('');

  useEffect(() => {
    void getDocuments()
      .then((items) => {
        setDocuments(items);
        setActiveDocumentId(items[0]?.id ?? '');
      })
      .catch((reason: unknown) => setError(messageOf(reason)));
  }, []);

  async function handleFiles(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = '';
    if (!files.length) return;
    setBusy('upload');
    setUploadProgress({ current: 0, total: files.length });
    setError('');
    try {
      for (const [index, file] of files.entries()) {
        setUploadName(file.name);
        setUploadProgress({ current: index + 1, total: files.length });
        const uploaded = await uploadPdf(file);
        setDocuments((current) => [...current, uploaded]);
        setActiveDocumentId((current) => current || uploaded.id);
        setActivePage(1);
      }
    } catch (reason) {
      setError(messageOf(reason));
      const currentDocuments = await getDocuments().catch(() => null);
      if (currentDocuments) {
        setDocuments(currentDocuments);
        setActiveDocumentId(activeDocumentId || currentDocuments[0]?.id || '');
      }
    } finally {
      setBusy(null);
      setUploadName('');
      setUploadProgress({ current: 0, total: 0 });
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = question.trim();
    if (!value || !documents.length || busy) return;
    setQuestion('');
    setBusy('chat');
    setError('');
    try {
      const result = await askQuestion(value);
      setExchanges((current) => [...current, { question: value, result }]);
    } catch (reason) {
      setError(messageOf(reason));
      setQuestion(value);
    } finally {
      setBusy(null);
    }
  }

  function openCitation(citation: Citation) {
    setActiveDocumentId(citation.document_id);
    setActivePage(citation.page_number);
  }

  const activeDocument = documents.find(
    (document) => document.id === activeDocumentId,
  );

  return (
    <main className="app-shell">
      <header className="topbar">
        <strong className="brand">PDF Research</strong>
        <button
          className="button secondary"
          disabled={busy === 'upload'}
          onClick={() => fileInput.current?.click()}
          type="button"
        >
          Upload PDF
        </button>
        <input
          accept="application/pdf,.pdf"
          className="sr-only"
          multiple
          onChange={handleFiles}
          ref={fileInput}
          type="file"
        />
      </header>

      <div className="workspace">
        <section className="viewer-pane" aria-label="PDF viewer">
          <div className="viewer-toolbar">
            <div className="document-tabs">
              {documents.map((document) => (
                <button
                  className={
                    document.id === activeDocumentId
                      ? 'document-tab active'
                      : 'document-tab'
                  }
                  key={document.id}
                  onClick={() => {
                    setActiveDocumentId(document.id);
                    setActivePage(1);
                  }}
                  type="button"
                >
                  <span>{document.name}</span>
                  <small>{document.page_count}p</small>
                </button>
              ))}
            </div>
            {activeDocument && (
              <span className="page-label">
                Page {activePage} of {activeDocument.page_count}
              </span>
            )}
          </div>
          {activeDocument ? (
            <iframe
              className="pdf-frame"
              key={`${activeDocument.id}-${activePage}`}
              src={pdfUrl(activeDocument.id, activePage)}
              title={`${activeDocument.name}, page ${activePage}`}
            />
          ) : (
            <div className="viewer-empty">
              <h1>No PDF open</h1>
              <p>Upload a searchable PDF to begin.</p>
              <button
                className="button primary"
                onClick={() => fileInput.current?.click()}
                type="button"
              >
                Choose PDF
              </button>
            </div>
          )}
        </section>

        <section className="chat-pane" aria-label="Research chat">
          <div className="chat-heading">
            <h2>Chat</h2>
            <span>
              {documents.length}{' '}
              {documents.length === 1 ? 'document' : 'documents'}
            </span>
          </div>
          <div className="messages" aria-live="polite">
            {!exchanges.length && (
              <div className="chat-empty">
                <p>Ask a question. Citations open the source page.</p>
              </div>
            )}
            {exchanges.map((exchange, index) => (
              <div className="exchange" key={`${exchange.question}-${index}`}>
                <div className="user-message">{exchange.question}</div>
                <article className="assistant-message">
                  {renderAnswer(exchange.result, openCitation)}
                </article>
              </div>
            ))}
            {busy === 'chat' && (
              <div className="thinking">
                <span className="spinner small" aria-hidden="true" /> Searching
                documents…
              </div>
            )}
          </div>
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          <form className="composer" onSubmit={submit}>
            <textarea
              aria-label="Ask a question about the uploaded PDFs"
              disabled={!documents.length || busy !== null}
              maxLength={2000}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder={
                documents.length
                  ? 'Ask a question about your PDFs…'
                  : 'Upload a PDF first…'
              }
              rows={3}
              value={question}
            />
            <button
              aria-label="Send question"
              className="send"
              disabled={!question.trim() || !documents.length || busy !== null}
              type="submit"
            >
              Send
            </button>
          </form>
        </section>
      </div>
      {busy === 'upload' && (
        <div className="upload-overlay" role="status" aria-live="polite">
          <div className="upload-status">
            <span className="spinner" aria-hidden="true" />
            <div>
              <strong>Indexing PDF</strong>
              <p>{uploadName}</p>
              {uploadProgress.total > 1 && (
                <small>
                  File {uploadProgress.current} of {uploadProgress.total}
                </small>
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

function renderAnswer(
  result: ChatResult,
  onCitation: (citation: Citation) => void,
): ReactNode {
  const citations = new Map(
    result.citations.map((citation) => [citation.number, citation]),
  );
  return result.answer.split(/(\[\d+])/g).map((part, index) => {
    const match = /^\[(\d+)]$/.exec(part);
    const citation = match ? citations.get(Number(match[1])) : undefined;
    if (!citation) return <span key={`${part}-${index}`}>{part}</span>;
    return (
      <button
        aria-label={`Open ${citation.document_name}, page ${citation.page_number}`}
        className="citation"
        key={`${part}-${index}`}
        onClick={() => onCitation(citation)}
        title={`${citation.document_name} · page ${citation.page_number}`}
        type="button"
      >
        {citation.number}
      </button>
    );
  });
}

function messageOf(reason: unknown): string {
  return reason instanceof Error
    ? reason.message
    : 'Something went wrong. Please try again.';
}

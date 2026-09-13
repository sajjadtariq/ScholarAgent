export type PdfDocument = {
  id: string;
  name: string;
  size_bytes: number;
  page_count: number;
  chunk_count: number;
};

export type Citation = {
  number: number;
  chunk_id: string;
  document_id: string;
  document_name: string;
  page_number: number;
  text: string;
  reranker_score: number;
};

export type ChatResult = { answer: string; citations: Citation[] };

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

async function errorFrom(response: Response): Promise<Error> {
  const body = (await response.json().catch(() => null)) as {
    detail?: string;
  } | null;
  return new Error(body?.detail ?? `Request failed (${response.status}).`);
}

export async function getDocuments(): Promise<PdfDocument[]> {
  const response = await fetch(`${API_URL}/documents`);
  if (!response.ok) throw await errorFrom(response);
  return response.json() as Promise<PdfDocument[]>;
}

export async function uploadPdf(file: File): Promise<PdfDocument> {
  const form = new FormData();
  form.append('file', file, file.name);
  const response = await fetch(`${API_URL}/documents/upload`, {
    method: 'POST',
    body: form,
  });
  if (!response.ok) throw await errorFrom(response);
  return response.json() as Promise<PdfDocument>;
}

export async function askQuestion(question: string): Promise<ChatResult> {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
  if (!response.ok) throw await errorFrom(response);
  return response.json() as Promise<ChatResult>;
}

export function pdfUrl(documentId: string, page = 1): string {
  return `${API_URL}/documents/${documentId}/file#page=${page}&view=FitH`;
}

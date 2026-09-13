import { beforeEach, describe, expect, it, vi } from 'vitest';

import { askQuestion, getDocuments, pdfUrl, uploadPdf } from './api';

describe('PDF Research Chat API client', () => {
  beforeEach(() => vi.restoreAllMocks());

  it('uploads a PDF with multipart form data', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          id: 'doc-1',
          name: 'paper.pdf',
          page_count: 2,
          size_bytes: 20,
          chunk_count: 3,
        }),
        { status: 201 },
      ),
    );
    const file = new File(['%PDF'], 'paper.pdf', { type: 'application/pdf' });
    expect((await uploadPdf(file)).name).toBe('paper.pdf');
    expect(fetchMock.mock.calls[0]?.[1]?.method).toBe('POST');
    expect(fetchMock.mock.calls[0]?.[1]?.body).toBeInstanceOf(FormData);
  });

  it('lists PDFs and sends a normal chat request', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response('[]', { status: 200 }))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ answer: 'Answer [1]', citations: [] }), {
          status: 200,
        }),
      );
    expect(await getDocuments()).toEqual([]);
    expect((await askQuestion('What is the result?')).answer).toBe(
      'Answer [1]',
    );
    expect(fetchMock.mock.calls[1]?.[0]).toBe('http://localhost:8000/chat');
  });

  it('builds a PDF viewer URL for the cited page', () => {
    expect(pdfUrl('doc-1', 7)).toBe(
      'http://localhost:8000/documents/doc-1/file#page=7&view=FitH',
    );
  });
});

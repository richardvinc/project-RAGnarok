"use client";

import { useState } from "react";

const DEFAULT_QUERY = "who is mowgli's enemy in the story?";

function splitResponse(text, onCitationClick) {
  const parts = [];
  const citationRegex = /\[source:\s*([^#]+)#chunk:([0-9,\s]+)\]/g;
  let lastIndex = 0;
  let match;

  while ((match = citationRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(
        <span key={`text-${lastIndex}`}>
          {text.slice(lastIndex, match.index)}
        </span>,
      );
    }

    const chunkIds = match[2]
      .split(",")
      .map((value) => Number(value.trim()))
      .filter((value) => Number.isFinite(value));

    parts.push(
      <span key={`citation-${match.index}`} className="citation-group">
        <span>(</span>
        {chunkIds.map((chunkId, index) => (
          <span key={`${match.index}-${chunkId}`}>
            <button
              className="citation"
              onClick={() => onCitationClick(chunkId)}
              type="button"
            >
              source #{chunkId}
            </button>
            {index < chunkIds.length - 1 ? <span>, </span> : null}
          </span>
        ))}
        <span>)</span>
      </span>,
    );

    lastIndex = citationRegex.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(<span key={`text-${lastIndex}`}>{text.slice(lastIndex)}</span>);
  }

  return parts;
}

function getImageSource(data) {
  return (
    data?.image_url ??
    data?.generated_image_url ??
    data?.image_path ??
    data?.generated_image_path ??
    ""
  );
}

function formatDecisionValue(value) {
  if (typeof value === "string") {
    return value;
  }

  return JSON.stringify(value, null, 2);
}

function Section({ title, open, onToggle, children }) {
  return (
    <section className="panel">
      <button className="section-button" type="button" onClick={onToggle}>
        <span>{title}</span>
        <span>{open ? "▲" : "▼"}</span>
      </button>
      <div className="section-body" hidden={!open}>
        {children}
      </div>
    </section>
  );
}

function DecisionLogSection({ decisionLog, runSummary, open, onToggle }) {
  return (
    <Section title="Decision Log" open={open} onToggle={onToggle}>
      <div className="stack">
        <div className="summary-grid">
          <div className="summary-card">
            <span className="summary-label">Used Tools</span>
            <strong>{runSummary?.used_tools ? "Yes" : "No"}</strong>
          </div>
          <div className="summary-card">
            <span className="summary-label">Tool Names</span>
            <strong>
              {runSummary?.tool_names?.length
                ? runSummary.tool_names.join(", ")
                : "None"}
            </strong>
          </div>
          <div className="summary-card">
            <span className="summary-label">Total Steps</span>
            <strong>{runSummary?.total_steps ?? 0}</strong>
          </div>
        </div>

        <div className="decision-list">
          {decisionLog?.map((event, index) => (
            <article className="decision-card" key={`${event.step}-${index}`}>
              <div className="decision-header">
                <div>
                  <div className="decision-eyebrow">
                    Step {index + 1} • {event.stage}
                  </div>
                  <h3 className="decision-title">{event.step}</h3>
                </div>
                <div className={`decision-status status-${event.status}`}>
                  {event.status}
                </div>
              </div>

              <p className="decision-text">{event.decision}</p>

              <div className="decision-meta">
                <span>{new Date(event.timestamp).toLocaleString()}</span>
                {event.tool_name ? <span>Tool: {event.tool_name}</span> : null}
              </div>

              {Object.keys(event.details || {}).length ? (
                <details className="decision-details">
                  <summary>View details</summary>
                  <div className="decision-detail-grid">
                    {Object.entries(event.details).map(([key, value]) => (
                      <div className="decision-detail" key={key}>
                        <strong>{key}</strong>
                        <pre>{formatDecisionValue(value)}</pre>
                      </div>
                    ))}
                  </div>
                </details>
              ) : null}
            </article>
          ))}
        </div>
      </div>
    </Section>
  );
}

export default function HomePage() {
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [k, setK] = useState(8);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);
  const [selectedChunkId, setSelectedChunkId] = useState(null);
  const [openSections, setOpenSections] = useState({
    image: false,
    decisionLog: true,
    embedding: false,
    chunks: false,
    context: false,
    prompt: false,
    response: true,
  });

  const toggleSection = (key) => {
    setOpenSections((current) => ({
      ...current,
      [key]: !current[key],
    }));
  };

  const runQuery = async (event) => {
    event.preventDefault();

    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      setError("Please enter a query.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: trimmedQuery, k }),
      });

      if (!response.ok) {
        throw new Error("Failed to get a response from the backend.");
      }

      const result = await response.json();
      setData(result);
      setSelectedChunkId(result.retrieved_chunks?.[0]?.id ?? null);
      setOpenSections({
        image: Boolean(
          result.image_url ??
          result.generated_image_url ??
          result.image_path ??
          result.generated_image_path,
        ),
        decisionLog: true,
        embedding: true,
        chunks: true,
        context: true,
        prompt: true,
        response: true,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const selectedChunk =
    data?.retrieved_chunks?.find((chunk) => chunk.id === selectedChunkId) ??
    data?.retrieved_chunks?.[0] ??
    null;

  const imageSource = getImageSource(data);

  return (
    <main className="page">
      <div className="container stack">
        <header className="hero">
          <h1>Project RAGnarok</h1>
          <p>
            A grounded RAG demo over The Jungle Book with structured LLM
            decision logging, tool tracing, and a frontend inspector for every
            execution step.
          </p>
        </header>

        <form className="panel" onSubmit={runQuery}>
          <div className="query-row">
            <label className="stack" style={{ gap: 6 }}>
              <span className="hint">Query</span>
              <input
                className="input"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Ask your question here"
              />
            </label>

            <label className="stack" style={{ gap: 6 }}>
              <span className="hint">Chunks limit</span>
              <input
                className="small-input"
                type="number"
                min="1"
                max="20"
                value={k}
                onChange={(event) => setK(Number(event.target.value) || 1)}
              />
            </label>
            <button className="button" type="submit" disabled={loading}>
              {loading ? "Searching..." : "Search"}
            </button>
          </div>

          {error ? <div className="error">{error}</div> : null}
        </form>

        {data ? (
          <>
            <Section
              title="LLM Response"
              open={openSections.response}
              onToggle={() => toggleSection("response")}
            >
              <div className="response mono-box">
                {splitResponse(data.llm_response, setSelectedChunkId)}
              </div>
            </Section>

            <section className="detail-card">
              <h2 className="detail-title">Source Detail</h2>
              {selectedChunk ? (
                <div>
                  <div className="badge">Chunk ID {selectedChunk.id}</div>
                  <p className="source-meta">
                    <strong>Source:</strong> {selectedChunk.source}
                  </p>
                  {selectedChunk.section_path ? (
                    <p className="source-meta">
                      <strong>Section:</strong> {selectedChunk.section_path}
                    </p>
                  ) : null}
                  <div className="content-box">{selectedChunk.content}</div>
                </div>
              ) : (
                <p className="hint">
                  Select a citation or chunk to inspect the source.
                </p>
              )}
            </section>

            {imageSource ? (
              <Section
                title="Generated Image"
                open={openSections.image}
                onToggle={() => toggleSection("image")}
              >
                <div className="stack">
                  <a
                    className="hint"
                    href={imageSource}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {imageSource}
                  </a>
                  <div
                    className="content-box"
                    style={{ padding: 0, overflow: "hidden" }}
                  >
                    <img
                      src={imageSource}
                      alt="Generated visual"
                      style={{
                        display: "block",
                        width: "100%",
                        height: "auto",
                      }}
                    />
                  </div>
                </div>
              </Section>
            ) : null}

            <DecisionLogSection
              decisionLog={data.decision_log}
              runSummary={data.run_summary}
              open={openSections.decisionLog}
              onToggle={() => toggleSection("decisionLog")}
            />

            {/* <Section
              title="Query Embedding"
              open={openSections.embedding}
              onToggle={() => toggleSection("embedding")}
            >
              <div className="mono-box embedding-box">
                [
                {data.query_embedding
                  .slice(0, 20)
                  .map((value) => value.toFixed(4))
                  .join(", ")}{" "}
                ...]
              </div>
            </Section> */}

            <Section
              title="Retrieved Chunks"
              open={openSections.chunks}
              onToggle={() => toggleSection("chunks")}
            >
              <div className="chunk-grid">
                {data.retrieved_chunks.map((chunk, index) => (
                  <article className="chunk-card" key={chunk.id}>
                    <div>
                      <strong>Chunk {index + 1}</strong>{" "}
                      <span className="badge">ID {chunk.id}</span>
                    </div>
                    <div className="chunk-meta">
                      <div>
                        <strong>Source:</strong> {chunk.source}
                      </div>
                      {chunk.section_path ? (
                        <div>
                          <strong>Section:</strong> {chunk.section_path}
                        </div>
                      ) : null}
                    </div>
                    <div className="content-box">{chunk.content}</div>
                  </article>
                ))}
              </div>
            </Section>

            <Section
              title="Formatted Context"
              open={openSections.context}
              onToggle={() => toggleSection("context")}
            >
              <div className="mono-box">{data.formatted_context}</div>
            </Section>

            <Section
              title="Final Prompt"
              open={openSections.prompt}
              onToggle={() => toggleSection("prompt")}
            >
              <div className="mono-box">{data.final_prompt}</div>
            </Section>
          </>
        ) : null}
      </div>
    </main>
  );
}

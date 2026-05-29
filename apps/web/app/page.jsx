"use client";

import { useMemo, useState } from "react";

const DEFAULT_QUERY =
  "Generate me an image of Mowgli, show me the source data used and translate it to indonesian";

const CITATION_REGEX = /\[source:\s*([^#]+)#chunk:([0-9,\s]+)\]/g;

function splitResponse(text, onCitationClick) {
  const parts = [];
  let lastIndex = 0;
  let match;

  CITATION_REGEX.lastIndex = 0;

  while ((match = CITATION_REGEX.exec(text)) !== null) {
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

    lastIndex = CITATION_REGEX.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(<span key={`text-${lastIndex}`}>{text.slice(lastIndex)}</span>);
  }

  return parts;
}

function hasCitations(text) {
  if (!text) {
    return false;
  }
  CITATION_REGEX.lastIndex = 0;
  return CITATION_REGEX.test(text);
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

function DecisionLogSection({ decisionLogHistory, open, onToggle }) {
  return (
    <Section title="Decision Log" open={open} onToggle={onToggle}>
      <div className="stack">
        {decisionLogHistory.length ? (
          decisionLogHistory.map((turn) => (
            <section className="turn-group" key={`turn-${turn.turnIndex}`}>
              <div className="turn-header">
                <div>
                  <div className="decision-eyebrow">Turn {turn.turnIndex}</div>
                  <h3 className="turn-title">{turn.query}</h3>
                </div>
              </div>

              <div className="summary-grid">
                <div className="summary-card">
                  <span className="summary-label">Used Tools</span>
                  <strong>{turn.runSummary?.used_tools ? "Yes" : "No"}</strong>
                </div>
                <div className="summary-card">
                  <span className="summary-label">Tool Names</span>
                  <strong>
                    {turn.runSummary?.tool_names?.length
                      ? turn.runSummary.tool_names.join(", ")
                      : "None"}
                  </strong>
                </div>
                <div className="summary-card">
                  <span className="summary-label">Total Steps</span>
                  <strong>{turn.runSummary?.total_steps ?? 0}</strong>
                </div>
              </div>

              <div className="decision-list">
                {turn.decisionLog?.map((event, index) => (
                  <article
                    className="decision-card"
                    key={`${turn.turnIndex}-${event.step}-${index}`}
                  >
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
                      {event.tool_name ? (
                        <span>Tool: {event.tool_name}</span>
                      ) : null}
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
            </section>
          ))
        ) : (
          <p className="hint">
            Decision logs will appear after the first query.
          </p>
        )}
      </div>
    </Section>
  );
}

function HistoryStateSection({ history, previousTurnContext, open, onToggle }) {
  return (
    <Section title="History State" open={open} onToggle={onToggle}>
      <div className="stack">
        <div className="history-list">
          {history.length ? (
            history.map((turn, index) => (
              <article className="history-card" key={`history-${index}`}>
                <div className="decision-eyebrow">
                  {turn.role} #{index + 1}
                </div>
                <div>{turn.content}</div>
              </article>
            ))
          ) : (
            <p className="hint">No history yet.</p>
          )}
        </div>

        {/* <details className="panel-nested">
          <summary>Previous Turn Context</summary>
          <div className="mono-box">
            {JSON.stringify(previousTurnContext, null, 2)}
          </div>
        </details> */}
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
  const [history, setHistory] = useState([]);
  const [previousTurnContext, setPreviousTurnContext] = useState(null);
  const [decisionLogHistory, setDecisionLogHistory] = useState([]);
  const [selectedChunkId, setSelectedChunkId] = useState(null);
  const [openSections, setOpenSections] = useState({
    image: false,
    decisionLog: true,
    historyState: false,
    chunks: false,
    context: false,
    prompt: false,
    response: true,
  });

  const translatedChunkMap = useMemo(() => {
    const map = new Map();
    for (const chunk of data?.translated_chunks ?? []) {
      map.set(chunk.chunk_id, chunk);
    }
    return map;
  }, [data]);

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
        body: JSON.stringify({
          query: trimmedQuery,
          k,
          history,
          previous_turn_context: previousTurnContext,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get a response from the backend.");
      }

      const result = await response.json();
      const citationSafeResponse = hasCitations(result.translated_llm_response)
        ? result.translated_llm_response
        : result.llm_response;
      const assistantContent =
        citationSafeResponse ?? result.llm_response ?? "";
      const nextHistory = [
        ...history,
        { role: "user", content: trimmedQuery },
        { role: "assistant", content: assistantContent },
      ];
      const nextPreviousTurnContext = {
        assistant_response: assistantContent,
        cited_chunks: (result.cited_chunk_ids ?? [])
          .map((chunkId) =>
            result.retrieved_chunks?.find((chunk) => chunk.id === chunkId),
          )
          .filter(Boolean)
          .map((chunk) => ({
            chunk_id: chunk.id,
            source: chunk.source,
            section_path: chunk.section_path,
            content: chunk.content,
          })),
      };
      const nextDecisionLogEntry = {
        turnIndex: Math.floor(nextHistory.length / 2),
        query: trimmedQuery,
        decisionLog: result.decision_log ?? [],
        runSummary: result.run_summary ?? null,
      };

      setData(result);
      setHistory(nextHistory);
      setPreviousTurnContext(nextPreviousTurnContext);
      setDecisionLogHistory((current) => [...current, nextDecisionLogEntry]);
      setSelectedChunkId(
        result.cited_chunk_ids?.[0] ?? result.retrieved_chunks?.[0]?.id ?? null,
      );
      setOpenSections({
        image: Boolean(getImageSource(result)),
        decisionLog: true,
        historyState: true,
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

  const translatedSelectedChunk = selectedChunk
    ? translatedChunkMap.get(selectedChunk.id)
    : null;
  const imageSource = getImageSource(data);
  const interactiveResponse = hasCitations(data?.translated_llm_response)
    ? data?.translated_llm_response
    : (data?.llm_response ?? "");
  const showOriginalResponse =
    data?.translated_llm_response &&
    data?.translated_llm_response !== data?.llm_response;

  return (
    <main className="page">
      <div className="container stack">
        <header className="hero">
          <h1>Project RAGnarok</h1>
          <p>
            Grounded RAG over The Jungle Book with structured decision logging,
            multi-tool chaining, and translated cited sources.
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

        <HistoryStateSection
          history={history}
          previousTurnContext={previousTurnContext}
          open={openSections.historyState}
          onToggle={() => toggleSection("historyState")}
        />

        {data ? (
          <>
            <Section
              title="LLM Response"
              open={openSections.response}
              onToggle={() => toggleSection("response")}
            >
              <div className="stack">
                <div className="summary-grid">
                  <div className="summary-card">
                    <span className="summary-label">Answer Language</span>
                    <strong>{data.llm_response_language}</strong>
                  </div>
                  <div className="summary-card">
                    <span className="summary-label">Cited Chunks</span>
                    <strong>
                      {data.cited_chunk_ids?.length
                        ? data.cited_chunk_ids.join(", ")
                        : "None"}
                    </strong>
                  </div>
                  <div className="summary-card">
                    <span className="summary-label">Translated Sources</span>
                    <strong>{data.translated_chunks?.length ?? 0}</strong>
                  </div>
                </div>

                <div className="response mono-box">
                  {splitResponse(interactiveResponse, setSelectedChunkId)}
                </div>

                {showOriginalResponse ? (
                  <details className="panel-nested">
                    <summary>Show original response</summary>
                    <div className="mono-box">
                      {splitResponse(data.llm_response, setSelectedChunkId)}
                    </div>
                  </details>
                ) : null}
              </div>
            </Section>

            <section className="detail-card">
              <h2 className="detail-title">Source Detail</h2>
              {selectedChunk ? (
                <div className="stack">
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

                  {translatedSelectedChunk ? (
                    <div className="translated-block">
                      <div className="badge badge-secondary">
                        {translatedSelectedChunk.target_language}
                      </div>
                      <div className="content-box translated-box">
                        {translatedSelectedChunk.translated_content}
                      </div>
                    </div>
                  ) : null}
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
              decisionLogHistory={decisionLogHistory}
              open={openSections.decisionLog}
              onToggle={() => toggleSection("decisionLog")}
            />

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

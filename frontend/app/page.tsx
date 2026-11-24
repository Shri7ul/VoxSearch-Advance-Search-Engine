"use client";

import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import {
  MAX_QUERY_CHARACTERS,
  validateQuery,
} from "@/lib/search-input";
import {
  SearchApiClient,
  createRequestTracker,
  type SearchFilters,
  type SearchResultSnippet,
} from "@/lib/search-api";
import { toResultCardModel } from "@/lib/result-cards";

/**
 * Goal 29 search homepage + Goal 30 API integration.
 *
 * Submitting validates locally, then POSTs the verbatim query to the existing
 * backend endpoint. Result cards (Goal 31), timestamp links (Goal 32) and
 * filters (Goal 33) are intentionally NOT implemented here.
 */
export default function HomePage() {
  const [query, setQuery] = useState("");
  const [language, setLanguage] = useState("");
  const [maxDurationSeconds, setMaxDurationSeconds] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">(
    "idle",
  );
  const [message, setMessage] = useState("");
  const [results, setResults] = useState<SearchResultSnippet[]>([]);

  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  const client = useMemo(() => new SearchApiClient(undefined, baseUrl), [baseUrl]);
  const tracker = useMemo(() => createRequestTracker(), []);
  const inFlight = useRef(false);
  const resultCards = useMemo(
    () => results.map(toResultCardModel),
    [results],
  );

  // Abort in-flight work on unmount so a late response cannot touch dead state.
  useEffect(() => () => tracker.abortAll(), [tracker]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const validation = validateQuery(query);
    if (!validation.ok) {
      tracker.abortAll();
      inFlight.current = false;
      setStatus("error");
      setMessage(validation.error);
      setResults([]);
      return;
    }
    // Ignore a second submit while a request is already in flight.
    if (inFlight.current) return;

    const filters: SearchFilters = {
      language: language || null,
      maxDurationSeconds: maxDurationSeconds
        ? Number(maxDurationSeconds)
        : null,
    };
    const { id, controller } = tracker.next();
    inFlight.current = true;
    setStatus("loading");
    setMessage("Searching…");
    setResults([]);

    try {
      const response = await client.search(validation.value, controller.signal, filters);
      if (!tracker.isCurrent(id)) return; // stale response
      setResults(response.results);
      setStatus("success");
      setMessage(
        response.count === 0
          ? "No matching transcript chunks."
          : `${response.count} matching transcript chunk${response.count === 1 ? "" : "s"}.`,
      );
    } catch (error) {
      if (!tracker.isCurrent(id)) return; // stale failure
      if (error instanceof Error && error.name === "AbortError") return;
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Search failed.");
      setResults([]);
    } finally {
      if (tracker.isCurrent(id)) inFlight.current = false;
    }
  }

  function markFiltersChanged() {
    tracker.abortAll();
    inFlight.current = false;
    setStatus("idle");
    setMessage("Filters changed — submit to search.");
    setResults([]);
  }

  function resetFilters() {
    setLanguage("");
    setMaxDurationSeconds("");
    markFiltersChanged();
  }

  return (
    <main className="home">
      <section className="hero">
        <h1>VoxSearch</h1>
        <p className="tagline">
          Search what was actually said in a video — not just its title.
        </p>

        <form className="search-form" onSubmit={handleSubmit} noValidate>
          <label className="sr-only" htmlFor="search-query">
            Search query
          </label>
          <input
            id="search-query"
            name="query"
            type="search"
            value={query}
            maxLength={MAX_QUERY_CHARACTERS}
            placeholder="Describe what you remember from a video…"
            autoComplete="off"
            aria-busy={status === "loading"}
            onChange={(event) => setQuery(event.target.value)}
          />
          <button type="submit" disabled={status === "loading"}>
            {status === "loading" ? "Searching…" : "Search transcripts"}
          </button>
        </form>

        <fieldset className="filters" disabled={status === "loading"}>
          <legend>Refine results</legend>
          <div className="filters__row">
            <label htmlFor="filter-language">Language</label>
            <select
              id="filter-language"
              value={language}
              onChange={(event) => {
                setLanguage(event.target.value);
                markFiltersChanged();
              }}
            >
              <option value="">Any language</option>
              <option value="en">English</option>
              <option value="bn">Bangla</option>
            </select>

            <label htmlFor="filter-duration">Video duration</label>
            <select
              id="filter-duration"
              value={maxDurationSeconds}
              onChange={(event) => {
                setMaxDurationSeconds(event.target.value);
                markFiltersChanged();
              }}
            >
              <option value="">Any duration</option>
              <option value="300">Up to 5 minutes</option>
              <option value="900">Up to 15 minutes</option>
              <option value="3600">Up to 1 hour</option>
            </select>

            <button type="button" className="filters__reset" onClick={resetFilters}>
              Clear filters
            </button>
          </div>
          {(language !== "" || maxDurationSeconds !== "") && (
            <p className="filters__active" data-testid="active-filters">
              Active filters: {language || "any language"} ·{" "}
              {maxDurationSeconds ? `≤ ${Number(maxDurationSeconds) / 60} min` : "any duration"}
            </p>
          )}
        </fieldset>

        <p
          className={`status ${status}`}
          role="status"
          aria-live="polite"
          data-status={status}
        >
          {message}
        </p>

        {status === "success" && resultCards.length > 0 && (
          <ol className="results" aria-label="Search results">
            {resultCards.map((card) => (
              <li key={card.id} className="result-card">
                <div className="result-card__head">
                  <span className="result-card__rank" aria-hidden="true">
                    {card.rank}
                  </span>
                  <div className="result-card__meta">
                    {card.videoLabel !== null && (
                      <span>Video {card.videoLabel}</span>
                    )}
                    {card.transcriptLabel !== null && (
                      <span>Transcript {card.transcriptLabel}</span>
                    )}
                    {card.chunkLabel !== null && <span>{card.chunkLabel}</span>}
                  </div>
                </div>
                <p className="result-card__text">{card.text}</p>
                <div className="result-card__foot">
                  {card.timestampUrl !== null ? (
                    <a
                      className="result-card__timestamp"
                      href={card.timestampUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-label={`Open video at ${card.timing}`}
                    >
                      {card.timing}
                    </a>
                  ) : (
                    <span>{card.timing}</span>
                  )}
                  {card.relevance !== null && <span>{card.relevance}</span>}
                  {card.precisionLabel !== null && (
                    <span>{card.precisionLabel}</span>
                  )}
                  {card.modelLabel !== null && <span>{card.modelLabel}</span>}
                </div>
              </li>
            ))}
          </ol>
        )}

        <p className="note">
          This homepage validates locally and calls the search API. Timestamp
          links and filters are tracked in later goals.
        </p>
      </section>
    </main>
  );
}

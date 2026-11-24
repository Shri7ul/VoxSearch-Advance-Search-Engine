"use client";

import { useState, type FormEvent } from "react";
import {
  MAX_QUERY_CHARACTERS,
  formatAcceptedMessage,
  validateQuery,
} from "@/lib/search-input";

/**
 * Goal 29 — the search homepage UI only.
 *
 * Submitting validates the query and shows local UI state. There is deliberately
 * NO fetch / API call: live results are Goal 30, result cards are Goal 31 and
 * timestamp links are Goal 32.
 */
export default function HomePage() {
  const [query, setQuery] = useState("");
  const [message, setMessage] = useState<string | null>(null);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const validation = validateQuery(query);
    setMessage(
      validation.ok ? formatAcceptedMessage(validation.value) : validation.error,
    );
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
            onChange={(event) => setQuery(event.target.value)}
          />
          <button type="submit">Search transcripts</button>
        </form>

        <p className="status" role="status" aria-live="polite">
          {message ?? ""}
        </p>

        <p className="note">
          This homepage is the Goal 29 search UI. API integration, result cards,
          timestamp links and filters are tracked in later goals.
        </p>
      </section>
    </main>
  );
}

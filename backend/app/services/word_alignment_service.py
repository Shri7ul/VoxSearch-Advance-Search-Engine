"""Word-alignment orchestration + persistence service (Goal 12).

This is the application/business boundary for word-level timestamp preservation.
It composes the Goal 08 repository abstractions inside SERVICE-OWNED
transactions (see :func:`app.services.transaction`), reuses the fail-closed
:class:`~app.services.content_rights_service.ContentRightsService.is_active`
authorization guard, and delegates every pure/validation decision to
:mod:`app.transcripts.word_alignment` / :mod:`app.transcripts.word_contract`.
It owns NO SQL and NO provider/model code.

Two-phase lifecycle (approved; retry + concurrency safe)
--------------------------------------------------------
Phase A — validate, then create + COMMIT a RUNNING attempt:
    * fail-closed authorization (video exists; the transcript asset and audio
      asset exist, belong to the SAME video, and are authorized by the SAME
      active right; the right is active/unexpired/unrevoked),
    * fail-closed language gate (Bengali / unknown / unsupported codes rejected
      with ZERO provider calls, ZERO runs, ZERO word rows).
    Initial rights/language rejection creates NO alignment run at all.
    A ``running`` row is committed so the attempt is durable/auditable.

Provider execution — OUTSIDE any DB transaction. A ``ProviderCallable`` receives
    the pinned :class:`AlignmentRequest` dict and returns a result mapping. The
    core runtime imports NO torch/whisperx; the default tests inject a FAKE
    provider. The optional isolated worker is a separate process.

Phase B — recheck authorization, validate, persist ATOMICALLY:
    * re-read the right; if revoked/expired now -> terminal ``failed`` with
      ``rights_revoked`` and ZERO new word rows (previous success preserved),
    * strictly validate provider output (no interpolated/backfilled timings),
    * a ZERO-word result -> terminal ``failed`` (previous success preserved),
    * a >0-word result -> WIN the current-success slot in ONE transaction:
      supersede (and delete words of) the previous current success, insert the
      new word set, set final status ``completed`` (no skips) / ``partial``
      (some skips). Only a SUCCESSFUL replacement may supersede/replace.
    * a concurrent-winner UNIQUE violation (partial index) is detected by
      re-querying; the losing run is finalized as ``failed`` with
      ``superseded_by_concurrent_run``. Genuine schema bugs are re-raised, never
      hidden.

Known limitation: a process crash between the Phase A commit and provider
completion leaves a stale ``running`` run. Reaping stale runs is a future goal;
they carry no word rows and never block a later successful attempt (they are
excluded from the partial index).

Goal 13 integration contract
----------------------------
:meth:`WordAlignmentService.get_aligned_sentences` is a READ-ONLY composition of
Goal 10 :func:`segment_sentences`, :func:`tokenize_original`, and
:func:`derive_aligned_sentences`. It upgrades a sentence to ``WORD_ALIGNED``
ONLY when every original token in its span has a valid aligned word, and NEVER
populates Goal 10's ``exact_start_seconds`` / ``exact_end_seconds`` (those stay
``None``). Exact timing is expressed on :class:`AlignedSentence` in integer ms.

Scope & safety: NO chunking, embeddings, reranking, search, ASR download, or
automatic audio fetching. No credentials / URLs / secrets are logged. Integer
milliseconds are a representation choice, not a 1 ms accuracy claim. Only
authorized sources are processed.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.repositories.content_rights_repository import ContentRightsRepository
from app.repositories.source_audio_asset_repository import SourceAudioAssetRepository
from app.repositories.transcript_alignment_run_repository import (
    TranscriptAlignmentRunRepository,
)
from app.repositories.transcript_asset_repository import TranscriptAssetRepository
from app.repositories.transcript_segment_repository import TranscriptSegmentRepository
from app.repositories.transcript_word_repository import TranscriptWordRepository
from app.repositories.video_repository import VideoRepository
from app.services import (
    ContentRightsNotEligibleError,
    TranscriptAssetNotFoundError,
    TranscriptAssetOwnershipError,
    VoxSearchServiceError,
    transaction,
)
from app.services.content_rights_service import ContentRightsService
from app.transcripts.sentence_segmentation import (
    SentenceSegmentationInput,
    segment_sentences,
)
from app.transcripts.sentence_timestamps import (
    AlignedSentence,
    derive_aligned_sentences,
)
from app.transcripts.word_alignment import (
    SegmentData,
    SkippedSegment,
    build_request,
    prepare_segments,
    validate_provider_segment,
)
from app.transcripts.word_contract import (
    ALIGNMENT_PROVIDER,
    ALIGNMENT_PROVIDER_VERSION,
    COMPLETED_STATUS,
    FAILED_STATUS,
    PARTIAL_STATUS,
    REASON_ALL_SEGMENTS_SKIPPED,
    REASON_INVALID_WINDOW,
    REASON_NO_ALIGNABLE_SEGMENTS,
    REASON_PROVIDER_ERROR,
    REASON_RIGHTS_REVOKED,
    REASON_SUPERSEDED_BY_CONCURRENT_RUN,
    REASON_UNSUPPORTED_LANGUAGE,
    RUNNING_STATUS,
    UNSUPPORTED_LANGUAGE_STATUS,
    AlignedWord,
    SourceAudioAssetInput,
    language_gate_reason,
    tokenize_original,
)

# A provider receives the bounded AlignmentRequest dict and returns the
# service-shaped result mapping. The default test provider is a FAKE; the
# optional isolated worker implements the same protocol via a subprocess.
ProviderCallable = Callable[[dict[str, Any]], Mapping[str, Any]]


# --- Domain exceptions (credential-free) --------------------------------------

class WordAlignmentError(VoxSearchServiceError):
    """Base for explicit Goal 12 alignment failures."""


class AlignmentVideoNotFoundError(WordAlignmentError):
    """The video referenced by an alignment request does not exist."""


class SourceAudioAssetNotFoundError(WordAlignmentError):
    """The referenced authorized audio asset does not exist."""


class UnsupportedAlignmentLanguageError(WordAlignmentError):
    """The requested language is not supported by whisperx==3.8.6 alignment."""


# --- Result value objects ------------------------------------------------------

@dataclass(frozen=True)
class SkippedSegmentInfo:
    """One segment deliberately not aligned, with a closed reason."""

    segment_ref: str
    reason: str


@dataclass(frozen=True)
class AlignmentResult:
    """Clear, credential-free outcome of one alignment attempt."""

    run_id: uuid.UUID
    transcript_asset_id: uuid.UUID
    source_audio_asset_id: uuid.UUID
    video_id: uuid.UUID
    status: str
    failure_reason: str | None
    word_count: int
    skipped: tuple[SkippedSegmentInfo, ...]
    superseded_run_id: uuid.UUID | None


@dataclass(frozen=True)
class RegisteredAudioAsset:
    """Credential-free confirmation of an authorized audio asset registration.

    Mirrors the spec-compliant :class:`SourceAudioAssetInput` field set: the
    OPAQUE ``storage_uri`` (never a URL / secret), provenance fields, the
    OPTIONAL ``sample_rate_hz`` (``None`` = unknown), and the timeline offset
    plus available audio window.
    """

    audio_asset_id: uuid.UUID
    video_id: uuid.UUID
    content_right_id: uuid.UUID
    storage_uri: str
    checksum_sha256: str
    mime_type: str
    rights_basis: str
    sample_rate_hz: int | None
    timeline_offset_ms: int
    duration_ms: int | None


def _is_language_error(text: str) -> bool:
    lowered = text.lower()
    return "language" in lowered or "align-model" in lowered


class WordAlignmentService:
    """Atomic, fail-closed word alignment. Owns the two-phase transactions."""

    def __init__(
        self,
        session: Session,
        *,
        videos: VideoRepository | None = None,
        rights: ContentRightsRepository | None = None,
        assets: TranscriptAssetRepository | None = None,
        segments: TranscriptSegmentRepository | None = None,
        audio_assets: SourceAudioAssetRepository | None = None,
        runs: TranscriptAlignmentRunRepository | None = None,
        words: TranscriptWordRepository | None = None,
    ) -> None:
        self._session = session
        self.videos = videos or VideoRepository(session)
        self.rights = rights or ContentRightsRepository(session)
        self.assets = assets or TranscriptAssetRepository(session)
        self.segments = segments or TranscriptSegmentRepository(session)
        self.audio_assets = audio_assets or SourceAudioAssetRepository(session)
        self.runs = runs or TranscriptAlignmentRunRepository(session)
        self.words = words or TranscriptWordRepository(session)

    # --- Audio-asset registration (authorized only) --------------------------

    def register_source_audio_asset(
        self,
        payload: SourceAudioAssetInput | Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> RegisteredAudioAsset:
        """Validate + register an AUTHORIZED source audio asset in one transaction.

        Reuses the same fail-closed rights guard as alignment: the right must
        exist, belong to this video, and currently be active. The opaque
        ``storage_uri`` is stored verbatim and never interpreted as a URL; the
        checksum / MIME / rights-provenance fields are persisted unchanged.
        """
        contract = (
            payload
            if isinstance(payload, SourceAudioAssetInput)
            else SourceAudioAssetInput.model_validate(payload)
        )
        if self.videos.get_video_by_id(contract.video_id) is None:
            raise AlignmentVideoNotFoundError(
                "The video referenced by this audio asset does not exist."
            )
        rights = self.rights.get_rights_by_id(contract.content_right_id)
        if (
            rights is None
            or rights.video_id != contract.video_id
            or not ContentRightsService.is_active(rights, now=now)
        ):
            raise ContentRightsNotEligibleError(
                "No valid, currently active rights record for this video."
            )
        with transaction(self._session):
            asset = self.audio_assets.create_source_audio_asset(
                video_id=contract.video_id,
                content_right_id=contract.content_right_id,
                storage_uri=contract.storage_uri,
                checksum_sha256=contract.checksum_sha256,
                mime_type=contract.mime_type,
                rights_basis=contract.rights_basis,
                sample_rate_hz=contract.sample_rate_hz,
                timeline_offset_ms=contract.timeline_offset_ms,
                duration_ms=contract.duration_ms,
            )
            audio_id = asset.id
        return RegisteredAudioAsset(
            audio_asset_id=audio_id,
            video_id=contract.video_id,
            content_right_id=contract.content_right_id,
            storage_uri=contract.storage_uri,
            checksum_sha256=contract.checksum_sha256,
            mime_type=contract.mime_type,
            rights_basis=contract.rights_basis,
            sample_rate_hz=contract.sample_rate_hz,
            timeline_offset_ms=contract.timeline_offset_ms,
            duration_ms=contract.duration_ms,
        )

    # --- Main alignment orchestration -----------------------------------------

    def align(
        self,
        *,
        transcript_asset_id: uuid.UUID,
        source_audio_asset_id: uuid.UUID,
        video_id: uuid.UUID,
        language: str,
        provider: ProviderCallable,
        now: datetime | None = None,
    ) -> AlignmentResult:
        """Run one word-alignment attempt via the two-phase lifecycle.

        Raises a :class:`WordAlignmentError` / :class:`ContentRightsNotEligibleError`
        only for INITIAL gate rejections (no run created). All post-attempt
        outcomes are returned as a terminal :class:`AlignmentResult`.
        """
        # ---------- Phase A gates (read-only, NO writes) ----------
        if self.videos.get_video_by_id(video_id) is None:
            raise AlignmentVideoNotFoundError(
                "The video referenced by this alignment does not exist."
            )

        asset = self.assets.get_asset_by_id(transcript_asset_id)
        if asset is None:
            raise TranscriptAssetNotFoundError(
                "The transcript asset does not exist."
            )
        if asset.video_id != video_id:
            raise TranscriptAssetOwnershipError(
                "The transcript asset belongs to a different video."
            )

        audio = self.audio_assets.get_audio_asset_by_id(source_audio_asset_id)
        if audio is None:
            raise SourceAudioAssetNotFoundError(
                "The source audio asset does not exist."
            )
        if audio.video_id != video_id:
            raise TranscriptAssetOwnershipError(
                "The source audio asset belongs to a different video."
            )

        # Exact authorization lineage: both composite FKs on the run share ONE
        # content_right_id, so the asset and audio MUST be under the SAME right.
        content_right_id = asset.content_right_id
        if audio.content_right_id != content_right_id:
            raise ContentRightsNotEligibleError(
                "Transcript asset and audio asset are authorized by different "
                "rights records."
            )

        if not self._authorization_active(content_right_id, video_id, now):
            raise ContentRightsNotEligibleError(
                "No valid, currently active rights record for this video."
            )

        gate = language_gate_reason(language)
        if gate is not None:
            # ZERO provider calls, ZERO runs, ZERO word rows.
            raise UnsupportedAlignmentLanguageError(gate)

        # ---------- Phase A: create + COMMIT the RUNNING attempt ----------
        started_at = now if now is not None else datetime.now(timezone.utc)
        with transaction(self._session):
            # Scoped to (asset, audio, provider, version): a stable, gap-free
            # 1-based attempt number that satisfies uq_alignment_runs_attempt
            # and ck_alignment_runs_attempt_number.
            attempt_number = self.runs.next_attempt_number(
                transcript_asset_id,
                source_audio_asset_id,
                provider=ALIGNMENT_PROVIDER,
                provider_version=ALIGNMENT_PROVIDER_VERSION,
            )
            run = self.runs.create_run(
                video_id=video_id,
                transcript_asset_id=transcript_asset_id,
                source_audio_asset_id=source_audio_asset_id,
                content_right_id=content_right_id,
                language=language,
                status=RUNNING_STATUS,
                attempt_number=attempt_number,
                provider=ALIGNMENT_PROVIDER,
                provider_version=ALIGNMENT_PROVIDER_VERSION,
                audio_duration_ms=audio.duration_ms,
                started_at=started_at,
            )
            run_id = run.id

        # ---------- Pure preparation + bounded provider request ----------
        segment_rows = self.segments.list_all_segments_by_asset(
            transcript_asset_id
        )
        segment_count = len(segment_rows)
        segment_data = tuple(
            SegmentData(
                segment_id=s.id,
                segment_ref=str(s.id),
                start_seconds=s.start_seconds,
                end_seconds=s.end_seconds,
                original_text=s.original_text,
            )
            for s in segment_rows
        )
        skipped: list[SkippedSegmentInfo] = []

        # Fail closed if audio/video synchronization cannot be established: an
        # unknown (NULL) or empty audio window means NO segment can be verified
        # against available audio (full task §7). The authorized registration
        # path requires a positive duration, so this guards direct DB inserts.
        if audio.duration_ms is None or audio.duration_ms <= 0:
            return self._finalize_failure(
                run_id, transcript_asset_id, source_audio_asset_id,
                video_id, status=FAILED_STATUS,
                reason=REASON_INVALID_WINDOW, skipped=skipped,
                segment_count=segment_count, now=now,
            )

        preparation = prepare_segments(
            segment_data,
            timeline_offset_ms=audio.timeline_offset_ms,
            audio_duration_ms=audio.duration_ms,
        )
        skipped.extend(
            SkippedSegmentInfo(sk.segment_ref, sk.reason)
            for sk in preparation.skipped
        )

        # No alignable segments -> terminal failure WITHOUT a provider call.
        if not preparation.prepared:
            reason = (
                REASON_NO_ALIGNABLE_SEGMENTS
                if not segment_rows
                else REASON_ALL_SEGMENTS_SKIPPED
            )
            return self._finalize_failure(
                run_id, transcript_asset_id, source_audio_asset_id,
                video_id, status=FAILED_STATUS, reason=reason,
                skipped=skipped, segment_count=segment_count, now=now,
            )

        request = build_request(language, preparation)

        # ---------- Provider execution OUTSIDE any DB transaction ----------
        try:
            raw_result = provider(request.to_dict())
        except Exception:  # noqa: BLE001 — fail closed, never leak provider errors
            raw_result = {
                "segments": [],
                "provider_error": REASON_PROVIDER_ERROR,
            }

        # ---------- Phase B: recheck authorization (after provider) ----------
        if not self._authorization_active(content_right_id, video_id, now):
            return self._finalize_failure(
                run_id,
                transcript_asset_id,
                source_audio_asset_id,
                video_id,
                status=FAILED_STATUS,
                reason=REASON_RIGHTS_REVOKED,
                skipped=skipped,
                segment_count=segment_count,
                now=now,
            )

        # Provider-level error (whole attempt).
        provider_error = (
            raw_result.get("provider_error")
            if isinstance(raw_result, Mapping)
            else REASON_PROVIDER_ERROR
        )
        if provider_error:
            error_text = str(provider_error)
            if _is_language_error(error_text):
                return self._finalize_failure(
                    run_id, transcript_asset_id, source_audio_asset_id,
                    video_id, status=UNSUPPORTED_LANGUAGE_STATUS,
                    reason=REASON_UNSUPPORTED_LANGUAGE, skipped=skipped,
                    segment_count=segment_count, now=now,
                )
            return self._finalize_failure(
                run_id, transcript_asset_id, source_audio_asset_id,
                video_id, status=FAILED_STATUS,
                reason=REASON_PROVIDER_ERROR, skipped=skipped,
                segment_count=segment_count, now=now,
            )

        # ---------- Strict per-segment validation ----------
        by_id: dict[str, Mapping[str, Any]] = {}
        provider_segments = (
            raw_result.get("segments") if isinstance(raw_result, Mapping) else None
        )
        if isinstance(provider_segments, Sequence) and not isinstance(
            provider_segments, (str, bytes)
        ):
            for ps in provider_segments:
                if (
                    isinstance(ps, Mapping)
                    and isinstance(ps.get("segment_id"), str)
                ):
                    by_id[ps["segment_id"]] = ps

        validated: list[AlignedWord] = []
        validated_segment_count = 0
        for prepared in preparation.prepared:
            outcome = validate_provider_segment(
                prepared,
                by_id.get(str(prepared.segment_id)),
                timeline_offset_ms=audio.timeline_offset_ms,
            )
            if isinstance(outcome, SkippedSegment):
                skipped.append(
                    SkippedSegmentInfo(outcome.segment_ref, outcome.reason)
                )
            else:
                validated.extend(outcome)
                validated_segment_count += 1

        total_words = len(validated)

        # A ZERO-word result is a FAILED retry: previous success is preserved
        # untouched (no supersede, no delete, no new word rows).
        if total_words == 0:
            return self._finalize_failure(
                run_id, transcript_asset_id, source_audio_asset_id,
                video_id, status=FAILED_STATUS,
                reason=REASON_ALL_SEGMENTS_SKIPPED, skipped=skipped,
                segment_count=segment_count, now=now,
            )

        # ---------- SUCCESSFUL replacement in ONE transaction ----------
        status = COMPLETED_STATUS if not skipped else PARTIAL_STATUS
        try:
            with transaction(self._session):
                current = self.runs.find_current_success(
                    transcript_asset_id,
                    source_audio_asset_id,
                    exclude_run_id=run_id,
                )
                superseded_id: uuid.UUID | None = None
                if current is not None:
                    # Order matters at each flush: remove the OLD row from the
                    # partial UNIQUE index BEFORE promoting the new one.
                    self.runs.supersede_run(current.id)  # flush
                    self.words.delete_words_by_run(current.id)  # flush
                    superseded_id = current.id
                self.words.add_words(
                    validated,
                    alignment_run_id=run_id,
                    transcript_asset_id=transcript_asset_id,
                    video_id=video_id,
                )
                self.runs.set_run_outcome(
                    run_id,
                    status=status,
                    failure_reason=None,
                    word_count=total_words,
                    segment_count=segment_count,
                    aligned_segment_count=validated_segment_count,
                    skipped_segment_count=len(skipped),
                    finished_at=(
                        now if now is not None else datetime.now(timezone.utc)
                    ),
                )
        except IntegrityError:
            # A concurrent attempt won the partial UNIQUE index. Confirm a
            # DIFFERENT run is now the current success; otherwise this is a
            # genuine schema/data bug that must NOT be hidden -> re-raise.
            winner = self.runs.find_current_success(
                transcript_asset_id,
                source_audio_asset_id,
                exclude_run_id=run_id,
            )
            if winner is None:
                raise
            return self._finalize_failure(
                run_id, transcript_asset_id, source_audio_asset_id,
                video_id, status=FAILED_STATUS,
                reason=REASON_SUPERSEDED_BY_CONCURRENT_RUN, skipped=skipped,
                segment_count=segment_count, now=now,
            )

        return AlignmentResult(
            run_id=run_id,
            transcript_asset_id=transcript_asset_id,
            source_audio_asset_id=source_audio_asset_id,
            video_id=video_id,
            status=status,
            failure_reason=None,
            word_count=total_words,
            skipped=tuple(skipped),
            superseded_run_id=superseded_id,
        )

    # --- Goal 13 integration (read-only) ---------------------------------------

    def get_aligned_sentences(
        self, alignment_run_id: uuid.UUID
    ) -> tuple[AlignedSentence, ...]:
        """Read-only composition of Goal 10 sentences + Goal 12 aligned words.

        For the success run, loads its word rows and the asset's segments,
        re-segments each segment's VERBATIM text with Goal 10, and upgrades to
        ``WORD_ALIGNED`` ONLY fully covered sentences. Pure composition of
        :func:`segment_sentences`, :func:`tokenize_original`, and
        :func:`derive_aligned_sentences`. NO transaction, NO writes, NO
        interpolation; Goal 10 ``exact_start_seconds`` / ``exact_end_seconds``
        are NEVER populated.
        """
        run = self.runs.get_run_by_id(alignment_run_id)
        if run is None:
            return ()
        segment_rows = self.segments.list_all_segments_by_asset(
            run.transcript_asset_id
        )
        word_rows = self.words.list_words_by_run(alignment_run_id)

        words_by_segment: dict[uuid.UUID, list[AlignedWord]] = {}
        for row in word_rows:
            words_by_segment.setdefault(row.segment_id, []).append(
                AlignedWord(
                    segment_id=row.segment_id,
                    ordinal=row.word_index,
                    text=row.word,
                    char_start=row.char_start,
                    char_end=row.char_end,
                    start_ms=row.start_ms,
                    end_ms=row.end_ms,
                )
            )

        result: list[AlignedSentence] = []
        for seg in segment_rows:
            # Segments with no captured original fall back to normalized text
            # for SEGMENTATION only; they carry no tokens/words, so every
            # sentence stays SEGMENT_LEVEL (no fabricated exact timing).
            source_text = (
                seg.original_text if seg.original_text else seg.segment_text
            )
            if not source_text or not source_text.strip():
                continue
            sentences = segment_sentences(
                SentenceSegmentationInput(
                    source_text=source_text,
                    source_segment_ref=str(seg.id),
                    start_seconds=seg.start_seconds,
                    end_seconds=seg.end_seconds,
                    language=run.language,
                )
            )
            tokens = (
                tokenize_original(seg.original_text)
                if seg.original_text
                else ()
            )
            words = tuple(words_by_segment.get(seg.id, ()))
            result.extend(derive_aligned_sentences(sentences, words, tokens))
        return tuple(result)

    # --- Internal helpers ------------------------------------------------------

    def _authorization_active(
        self,
        content_right_id: uuid.UUID,
        video_id: uuid.UUID,
        now: datetime | None,
    ) -> bool:
        # FRESH read (populate_existing) is REQUIRED for the Phase B recheck:
        # the session uses expire_on_commit=False, so a plain get() would return
        # the identity-map object cached in Phase A and hide a revocation that
        # happened during the out-of-transaction provider execution.
        rights = self.rights.get_rights_by_id_fresh(content_right_id)
        return (
            rights is not None
            and rights.video_id == video_id
            and ContentRightsService.is_active(rights, now=now)
        )

    def _finalize_failure(
        self,
        run_id: uuid.UUID,
        transcript_asset_id: uuid.UUID,
        source_audio_asset_id: uuid.UUID,
        video_id: uuid.UUID,
        *,
        status: str,
        reason: str,
        skipped: Sequence[SkippedSegmentInfo],
        segment_count: int,
        now: datetime | None = None,
    ) -> AlignmentResult:
        """Commit a terminal failure for the Phase A run (no words persisted).

        No words are persisted, so ``word_count`` / ``aligned_segment_count``
        are honestly ``0``; ``skipped_segment_count`` reflects every skipped
        segment; ``finished_at`` closes the attempt. The previous successful
        alignment (if any) is left untouched — this only finalizes THIS run.
        """
        finished_at = now if now is not None else datetime.now(timezone.utc)
        with transaction(self._session):
            self.runs.set_run_outcome(
                run_id,
                status=status,
                failure_reason=reason,
                word_count=0,
                segment_count=segment_count,
                aligned_segment_count=0,
                skipped_segment_count=len(skipped),
                finished_at=finished_at,
            )
        return AlignmentResult(
            run_id=run_id,
            transcript_asset_id=transcript_asset_id,
            source_audio_asset_id=source_audio_asset_id,
            video_id=video_id,
            status=status,
            failure_reason=reason,
            word_count=0,
            skipped=tuple(skipped),
            superseded_run_id=None,
        )


__all__ = [
    "AlignmentResult",
    "AlignmentVideoNotFoundError",
    "ProviderCallable",
    "RegisteredAudioAsset",
    "SkippedSegmentInfo",
    "SourceAudioAssetNotFoundError",
    "UnsupportedAlignmentLanguageError",
    "WordAlignmentError",
    "WordAlignmentService",
]

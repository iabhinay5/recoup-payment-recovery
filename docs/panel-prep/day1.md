# Day 1 explain-back — taxonomy and LLM abstraction

Answer from memory, without opening the code. Where you cannot, say so plainly and
mark it — that flag is the useful output, not a failure.

## Taxonomy

1. Why does classifying decline reasons matter at all? Give the concrete money argument,
   not the abstract one.
2. Name the five decline classes and what distinguishes each. Which one is the real
   sequential decision problem, and why are the other four not?
3. `card_expired` and `payment_risk_check_failed` both have a cap of zero attempts but sit
   in different classes. Why is that not a modelling mistake?
4. What happens when Razorpay ships a decline code we have never seen? Why is that the
   correct default rather than the cautious-seeming alternative?
5. Why is `credit_failed` escalated rather than retried? (This one is about double-charge
   risk — make sure you can state the mechanism.)
6. Why does the taxonomy module have no third-party dependencies?

## LLM layer

7. Why is there a provider abstraction at all? Give the data-residency argument before
   the cost one.
8. What are the three cache modes, and what does REPLAY do on a miss?
9. Why does a cache miss raise instead of falling back to a live call? What specifically
   breaks if it falls back?
10. Why is the model name part of the cache key?
11. Why is caching viable here in a way it would not be for a general-purpose agent?

## The question most likely to be asked

12. "You have an AI track submission where the AI does not make the main decision.
    Explain that." — this is ADR-003. Have a two-sentence version and a two-minute version.

---

## Your answers

<!-- write below this line -->

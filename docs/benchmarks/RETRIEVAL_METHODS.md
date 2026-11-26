# Embedding Model: `Qwen/Qwen3_Embedding_0.6B`

| Method | Group | N | Hit@5 | Recall@5 | MRR |
|---|---|---:|---:|---:|---:|
| bm25 | overall | 500 | 0.594 | 0.473 | 0.450 |
| bm25 | temporal | 133 | 0.579 | 0.415 | 0.438 |
| bm25 | multi-session | 133 | 0.459 | 0.268 | 0.291 |
| bm25 | knowledge-update | 78 | 0.808 | 0.643 | 0.614 |
| bm25 | single-session-preference | 30 | 0.133 | 0.117 | 0.064 |
| bm25 | single-session-assistant | 56 | 0.804 | 0.804 | 0.709 |
| bm25 | single-session-user | 70 | 0.671 | 0.671 | 0.549 |
| embedding | overall | 500 | 0.644 | 0.513 | 0.458 |
| embedding | temporal | 133 | 0.579 | 0.442 | 0.380 |
| embedding | multi-session | 133 | 0.624 | 0.379 | 0.451 |
| embedding | knowledge-update | 78 | 0.731 | 0.573 | 0.514 |
| embedding | single-session-preference | 30 | 0.467 | 0.389 | 0.322 |
| embedding | single-session-assistant | 56 | 0.875 | 0.875 | 0.613 |
| embedding | single-session-user | 70 | 0.600 | 0.600 | 0.488 |
| hybrid w=0.3 | overall | 500 | 0.676 | 0.546 | 0.490 |
| hybrid w=0.3 | temporal | 133 | 0.632 | 0.470 | 0.444 |
| hybrid w=0.3 | multi-session | 133 | 0.639 | 0.401 | 0.433 |
| hybrid w=0.3 | knowledge-update | 78 | 0.795 | 0.669 | 0.587 |
| hybrid w=0.3 | single-session-preference | 30 | 0.333 | 0.261 | 0.204 |
| hybrid w=0.3 | single-session-assistant | 56 | 0.875 | 0.875 | 0.694 |
| hybrid w=0.3 | single-session-user | 70 | 0.686 | 0.686 | 0.537 |
| hybrid w=0.5 | overall | 500 | 0.682 | 0.546 | 0.509 |
| hybrid w=0.5 | temporal | 133 | 0.654 | 0.479 | 0.478 |
| hybrid w=0.5 | multi-session | 133 | 0.639 | 0.397 | 0.417 |
| hybrid w=0.5 | knowledge-update | 78 | 0.821 | 0.688 | 0.628 |
| hybrid w=0.5 | single-session-preference | 30 | 0.300 | 0.228 | 0.178 |
| hybrid w=0.5 | single-session-assistant | 56 | 0.857 | 0.857 | 0.741 |
| hybrid w=0.5 | single-session-user | 70 | 0.686 | 0.686 | 0.565 |
| hybrid w=0.7 | overall | 500 | 0.658 | 0.532 | 0.510 |
| hybrid w=0.7 | temporal | 133 | 0.632 | 0.473 | 0.491 |
| hybrid w=0.7 | multi-session | 133 | 0.602 | 0.370 | 0.407 |
| hybrid w=0.7 | knowledge-update | 78 | 0.821 | 0.682 | 0.632 |
| hybrid w=0.7 | single-session-preference | 30 | 0.133 | 0.117 | 0.117 |
| hybrid w=0.7 | single-session-assistant | 56 | 0.857 | 0.857 | 0.755 |
| hybrid w=0.7 | single-session-user | 70 | 0.700 | 0.700 | 0.580 |
| embedding --no prompt-- | overall | 500 | 0.604 | 0.466 | 0.421 |
| embedding --no prompt-- | temporal | 133 | 0.541 | 0.419 | 0.361 |
| embedding --no prompt-- | multi-session | 133 | 0.602 | 0.335 | 0.414 |
| embedding --no prompt-- | knowledge-update | 78 | 0.756 | 0.564 | 0.559 |
| embedding --no prompt-- | single-session-preference | 30 | 0.367 | 0.311 | 0.236 |
| embedding --no prompt-- | single-session-assistant | 56 | 0.696 | 0.696 | 0.469 |
| embedding --no prompt-- | single-session-user | 70 | 0.586 | 0.579 | 0.434 |
| hybrid w=0.3 --no prompt-- | overall | 500 | 0.670 | 0.541 | 0.489 |
| hybrid w=0.3 --no prompt-- | temporal | 133 | 0.624 | 0.488 | 0.446 |
| hybrid w=0.3 --no prompt-- | multi-session | 133 | 0.654 | 0.413 | 0.453 |
| hybrid w=0.3 --no prompt-- | knowledge-update | 78 | 0.821 | 0.656 | 0.605 |
| hybrid w=0.3 --no prompt-- | single-session-preference | 30 | 0.300 | 0.244 | 0.176 |
| hybrid w=0.3 --no prompt-- | single-session-assistant | 56 | 0.821 | 0.821 | 0.609 |
| hybrid w=0.3 --no prompt-- | single-session-user | 70 | 0.657 | 0.657 | 0.546 |
| hybrid w=0.5 --no prompt-- | overall | 500 | 0.684 | 0.554 | 0.520 |
| hybrid w=0.5 --no prompt-- | temporal | 133 | 0.647 | 0.493 | 0.490 |
| hybrid w=0.5 --no prompt-- | multi-session | 133 | 0.662 | 0.428 | 0.466 |
| hybrid w=0.5 --no prompt-- | knowledge-update | 78 | 0.808 | 0.656 | 0.607 |
| hybrid w=0.5 --no prompt-- | single-session-preference | 30 | 0.267 | 0.217 | 0.176 |
| hybrid w=0.5 --no prompt-- | single-session-assistant | 56 | 0.839 | 0.839 | 0.707 |
| hybrid w=0.5 --no prompt-- | single-session-user | 70 | 0.714 | 0.714 | 0.576 |
| hybrid w=0.7 --no prompt-- | overall | 500 | 0.676 | 0.552 | 0.523 |
| hybrid w=0.7 --no prompt-- | temporal | 133 | 0.639 | 0.495 | 0.496 |
| hybrid w=0.7 --no prompt-- | multi-session | 133 | 0.639 | 0.415 | 0.458 |
| hybrid w=0.7 --no prompt-- | knowledge-update | 78 | 0.821 | 0.675 | 0.609 |
| hybrid w=0.7 --no prompt-- | single-session-preference | 30 | 0.200 | 0.144 | 0.139 |
| hybrid w=0.7 --no prompt-- | single-session-assistant | 56 | 0.857 | 0.857 | 0.740 |
| hybrid w=0.7 --no prompt-- | single-session-user | 70 | 0.714 | 0.714 | 0.590 |
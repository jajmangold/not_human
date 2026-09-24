# Live speech contract

The live demo uses the exact Qwen3.8-27B non-thinking profile documented by the
publisher: temperature `0.7`, top-p `0.80`, top-k `20`, min-p `0`, presence
penalty `1.5`, and repetition penalty `1`. Both current and preserved thinking
are disabled because this lane needs immediate spoken output, not a hidden
reasoning preamble. The public gateway model name remains `qwen27b`; the UI
identifies the underlying family as Qwen3.8-27B.

Qwen receives a speech contract, not a generic chat prompt. It emits concise
spoken prose with ordinary punctuation and no Markdown, URLs, SSML, stage
directions, or pseudo-emotion tags. The displayed transcript remains the exact
model output.

Before synthesis, the Kokoro service applies conservative semantic text
normalization to high-confidence formats such as North American phone numbers,
ISO dates, clock times, currency, percentages, and common units. It then runs
the official Kokoro-specific Misaki English G2P with eSpeak as the
out-of-dictionary fallback. The previous service sent raw text directly to
kokoro-onnx's bare eSpeak tokenizer.

Punctuation is retained because it is Kokoro's pacing interface. Streaming
prefers full sentence boundaries, holds very short sentences long enough to
join a following sentence, and bounds long run-ons at a clause boundary. This
balances first-audio latency against the Kokoro voice guide's warning that very
short utterances are weak.

References:

- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M)
- [Kokoro voice guide](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)
- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx)
- [Misaki G2P](https://github.com/hexgrad/misaki)
- [Kokoro prompting notes](https://github.com/ghchinoy/kokoro-rs/blob/main/docs/prompting.md)

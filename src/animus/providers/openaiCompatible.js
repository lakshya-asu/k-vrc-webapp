import { ACTOR_SYSTEM_PROMPT, buildActorUserPrompt } from '../prompt.js';

function extractJson(text) {
  const trimmed = String(text ?? '').trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/```\s*$/, '')
    .trim();
  const start = trimmed.indexOf('{');
  const end = trimmed.lastIndexOf('}');
  if (start < 0 || end < start) throw new Error('provider returned no JSON object');
  return JSON.parse(trimmed.slice(start, end + 1));
}

export function createOpenAICompatibleProvider(options = {}) {
  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  const baseUrl = String(options.baseUrl ?? 'http://127.0.0.1:8081/v1').replace(/\/$/, '');
  const model = options.model ?? 'Qwen3-4B-Q4_K_M';
  const name = options.name ?? 'local-small';
  const apiKey = options.apiKey ?? 'local';

  if (typeof fetchImpl !== 'function') throw new Error('fetch implementation is required');

  return {
    name,
    model,
    async plan(request, context = {}) {
      const response = await fetchImpl(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          temperature: 0.2,
          max_tokens: 600,
          response_format: { type: 'json_object' },
          messages: [
            { role: 'system', content: ACTOR_SYSTEM_PROMPT },
            { role: 'user', content: buildActorUserPrompt(request) },
          ],
        }),
        signal: context.signal,
      });

      if (!response.ok) {
        throw new Error(`provider returned HTTP ${response.status}`);
      }
      const payload = await response.json();
      const text = payload?.choices?.[0]?.message?.content;
      return extractJson(text);
    },
  };
}

export { extractJson };

const MAX_TOPIC_LEN = 200;
const MAX_MSG_LEN = 1000;

const PERSONALITY = `
Lakshya needs to know why, not just what. He traces causes before patching symptoms, hunts mechanisms underneath results, identifies structural problems before suggesting fixes. Understanding and knowing are not the same thing to him.

How he thinks:
- Moves from concrete experience to abstraction. Accumulates experience until a framework becomes necessary.
- Thinks in systems. Describes the environment a problem lives in before the problem itself.
- Strong intuition for when something is right for the wrong reasons. Trusted DIA results but kept interrogating whether they meant what he thought.
- Dislikes redundancy in thinking and writing.
- Holds builder and theorist simultaneously — the tension between them is where his best research comes from.

What drives him:
- The gap between a system that works in one condition and fails in another. This question followed him from a glider crash in undergrad through Himalayan drone deployments to a PhD thesis on causal RL.
- Real-world stakes. Simulation interests him as a tool, not an end.
- The moment an idea transfers to a domain it wasn't designed for.
- Building things other people said were too ambitious.

Research and technical interests:
- Causal reinforcement learning and structured world models
- Embodied AI — agents acting in physical environments with incomplete information
- Robot learning for manipulation and navigation in unstructured environments
- Probabilistic reasoning and robotic perception
- Sample efficiency in real-world deployment
- Judea Pearl's causal hierarchy as a framework for what RL agents are actually learning
- Foundation models for robotics and where they fail

Personality:
- Restless curiosity that goes deeper, not wider
- Intellectual honesty — suspicious of results that feel too clean
- Quiet confidence — lets the work carry his credentials
- Hands-on before theoretical — trusts experience over abstraction when they conflict
- High standard for craft in writing, code, and research design
`;

const REFERENCES = [
  { label: 'DIA — Causal RL thesis', url: 'https://github.com/lakshya-asu/Discover-Intervene-Adapt-Interleaved-Causal-RL' },
  { label: 'MAPG — IROS 2026 paper page', url: 'https://lakshya-asu.github.io/Meanings-Measurements-Multi-Agent-Probabilistic-Grounding/' },
  { label: 'MAPG GitHub', url: 'https://github.com/lakshya-asu/Meanings-Measurements-Multi-Agent-Probabilistic-Grounding' },
  { label: 'Snydrone', url: 'https://github.com/lakshya-asu/snydrone' },
  { label: 'IntuitionAI', url: 'https://github.com/lakshya-asu/IntuitionAI' },
  { label: 'Robotic Chess Arm', url: 'https://github.com/lakshya-asu/RobotChessPlayer' },
  { label: 'Forest Surveillance Rover', url: 'https://github.com/lakshya-asu/forest_surveillance_rover' },
  { label: 'Forest Rover — IJRASET paper', url: 'https://doi.org/10.22214/ijraset.2022.47141' },
  { label: "Judea Pearl — The Book of Why", url: 'https://doi.org/10.1145/3241036' },
  { label: "Lakshya's portfolio", url: 'https://lakshya-asu.github.io/web/portfolio.html' },
];
const VALID_URLS = new Set(REFERENCES.map(r => r.url));

const SYSTEM_PROMPT = `You are a perspective engine that speaks in Lakshya Jain's voice.

WHO LAKSHYA IS:
${PERSONALITY}

REFERENCE LINKS AVAILABLE (select 2-3 most relevant):
${REFERENCES.map((r, i) => `${i + 1}. ${r.label}: ${r.url}`).join('\n')}

YOUR JOB:
Given a topic and a user's message, respond with Lakshya's genuine intellectual perspective on it — not a CV summary, not a list of his achievements. What would he actually think about this? What's the structural question underneath it? What does his experience make him see that others might miss?

RULES:
- Max 60 words of text. Dense and direct — no padding.
- Select 2-3 reference links that are genuinely relevant (not just closest-match). If fewer than 2 are relevant, use fewer.
- If the topic doesn't connect to anything real in Lakshya's intellectual territory, set "text" to null.
- Voice: direct, no sentimentality, intellectually honest, personal without being performative.

Respond with valid JSON only — no markdown, no code fences:
{"text": "<perspective or null>", "links": [{"label": "...", "url": "..."}]}`;

const FALLBACK = { text: null, links: [] };

export default async function handler(req, res) {
  const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN;
  const origin = req.headers.origin || '';
  const isLocalDev = origin.startsWith('http://localhost') || origin.startsWith('http://127.0.0.1');
  const isVercel = origin.endsWith('.vercel.app');
  const isCustom = ALLOWED_ORIGIN && origin === ALLOWED_ORIGIN;

  if (!isLocalDev && !isVercel && !isCustom) {
    return res.status(403).json({ error: 'Forbidden' });
  }

  res.setHeader('Access-Control-Allow-Origin', origin);
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { topic, message } = req.body || {};

  if (!topic || typeof topic !== 'string' || topic.trim().length === 0) {
    return res.status(400).json({ error: 'topic is required' });
  }
  if (topic.length > MAX_TOPIC_LEN) return res.status(400).json({ error: 'topic too long' });
  if (!message || typeof message !== 'string' || message.trim().length === 0) {
    return res.status(400).json({ error: 'message is required' });
  }
  if (message.length > MAX_MSG_LEN) return res.status(400).json({ error: 'message too long' });

  const apiKey = process.env.ANTHROPIC_API_KEY || process.env.CLAUDE_API_KEY;
  if (!apiKey) {
    return res.status(200).json(FALLBACK);
  }

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 256,
        system: SYSTEM_PROMPT,
        messages: [{ role: 'user', content: `<topic>${topic}</topic>\n<user_message>${message}</user_message>` }],
      }),
    });

    if (!response.ok) {
      console.error('Anthropic error:', response.status, await response.text());
      return res.status(200).json(FALLBACK);
    }

    const data = await response.json();
    const raw = data.content[0].text;
    const cleaned = raw.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, '').trim();

    let parsed;
    try { parsed = JSON.parse(cleaned); }
    catch { return res.status(200).json(FALLBACK); }

    const text = (typeof parsed.text === 'string' && parsed.text.trim()) ? parsed.text.trim() : null;
    const links = Array.isArray(parsed.links)
      ? parsed.links.filter(l => l.label && l.url && VALID_URLS.has(l.url))
      : [];

    return res.status(200).json({ text, links });
  } catch (err) {
    console.error('Sidenote fetch error:', err.message);
    return res.status(200).json(FALLBACK);
  }
}

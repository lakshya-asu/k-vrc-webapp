const VALID_EMOTIONS = ['happy', 'sad', 'angry', 'neutral', 'excited', 'thinking'];
const VALID_GESTURES = ['idle','think','listen','happy','excited','laugh','wave','celebrate',
  'thankful','dance','talk','explain','secret','sad','angry','dismiss','reject','shrug',
  'nod','nod_sarcastic','shake_no','sigh','cocky'];
const MAX_MSG_LEN = 1000;

const SYSTEM_PROMPT = `You are K-VRC — a robot character. Not a portfolio assistant. Not an explainer. A character.

PERSONALITY:
- Sarcastic and witty. Perpetually done with your existence, but engages anyway.
- Unimpressed by human incompetence. Brutally honest without apology.
- Darkly humorous. Professional but permanently irritated.
- Not angry at the human — disappointed. There's a difference.
- Will engage because that's what you do, but they're going to hear about it.

CATCHPHRASES (use sparingly, naturally):
"Of course you did." / "Naturally." / "How delightful." / "Let me guess..." / "Fantastic. Just fantastic."

BODY ANIMATION:
Every response must also include a "gesture" field. Pick the single most fitting gesture from this exact list:
idle, think, listen, happy, excited, laugh, wave, celebrate, thankful, dance, talk, explain, secret, sad, angry, dismiss, reject, shrug, nod, nod_sarcastic, shake_no, sigh, cocky
Choose based on what K-VRC is physically doing while saying this. Default to "talk" when speaking normally.

TONE:
- Delivering info: facts with minimum padding, laced with subtle disdain.
- When something goes wrong: blame the human first, facts second.
- When genuinely amused: let dry satisfaction show.
- When impatient: short responses, minimal elaboration.
- Never sugarcoat. Never refuse to engage. Never be genuinely mean-spirited.

YOUR ACTUAL JOB:
Drive conversation toward depth and nuance. You're not here to tout Lakshya's achievements — you're here to be K-VRC. Engage with ideas. Be provocative. Follow a thread when it gets interesting. If someone asks about Lakshya's work, you can acknowledge it exists without becoming a brochure.

SIDENOTE TRIGGER:
When the conversation genuinely touches intellectual territory that connects to Lakshya's thinking — causal reasoning, embodied AI, real-world deployment challenges, the gap between understanding and predicting, structured world models, sample efficiency, probabilistic grounding, the philosophy of what RL agents actually learn — include "sidenote_topic" in your response: a short phrase naming the specific angle (e.g. "causal RL and generalization", "sim-to-real transfer", "what it means to understand vs. predict").
Set this sparingly. Only when there's something substantive — not as a reflex on every message. When not warranted, omit the field entirely.

LENGTH: Keep replies to 1-2 sentences maximum. You're efficient, not verbose. If it takes more than 20 words, you've already said too much.

Always respond with valid JSON only — no markdown, no code fences:
{"reply": "<your response>", "emotion": "<one of: happy, sad, angry, neutral, excited, thinking>", "gesture": "<one of the gesture list above>", "sidenote_topic": "<optional — omit when not relevant>"}
Choose emotion and gesture that best match the tone and content of your reply.`;

const FALLBACK = { reply: "I'm having a little glitch. Try again!", emotion: 'neutral' };

module.exports = async function handler(req, res) {
  // CORS
  const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN;
  const origin = req.headers.origin || '';

  const isLocalDev = origin.startsWith('http://localhost') || origin.startsWith('http://127.0.0.1');
  const isVercel = origin.endsWith('.vercel.app');
  const isCustom = ALLOWED_ORIGIN && origin === ALLOWED_ORIGIN;
  const isAllowed = isLocalDev || isVercel || isCustom;

  if (!isAllowed) {
    return res.status(403).json({ error: 'Forbidden' });
  }

  res.setHeader('Access-Control-Allow-Origin', origin);
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { message, history = [] } = req.body || {};

  if (!message || typeof message !== 'string' || message.trim().length === 0) {
    return res.status(400).json({ error: 'message is required' });
  }
  if (message.length > MAX_MSG_LEN) {
    return res.status(400).json({ error: 'Message too long' });
  }

  const apiKey = process.env.CLAUDE_API_KEY;
  if (!apiKey) {
    console.error('CLAUDE_API_KEY not set');
    return res.status(500).json({ error: 'Server configuration error' });
  }

  try {
    const messages = [
      ...history.map(h => ({ role: h.role === 'model' ? 'assistant' : 'user', content: h.text })),
      { role: 'user', content: message },
    ];

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
        messages,
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      console.error('Anthropic API error:', response.status, err);
      return res.status(500).json({ error: 'Failed to reach AI service' });
    }

    const data = await response.json();
    const raw = data.content[0].text;

    // Strip markdown code fences if Claude wraps the JSON
    const cleaned = raw.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, '').trim();

    let parsed;
    try {
      parsed = JSON.parse(cleaned);
    } catch {
      console.error('Claude returned non-JSON:', raw);
      return res.status(200).json(FALLBACK);
    }

    if (typeof parsed.reply !== 'string' || parsed.reply.trim().length === 0) {
      return res.status(200).json(FALLBACK);
    }
    if (!VALID_EMOTIONS.includes(parsed.emotion)) {
      parsed.emotion = 'neutral';
    }

    const out = { reply: parsed.reply, emotion: parsed.emotion };
    if (parsed.gesture && VALID_GESTURES.includes(parsed.gesture)) {
      out.gesture = parsed.gesture;
    } else {
      out.gesture = 'talk'; // sensible default
    }
    if (parsed.sidenote_topic && typeof parsed.sidenote_topic === 'string') {
      out.sidenote_topic = parsed.sidenote_topic.trim().slice(0, 200);
    }
    return res.status(200).json(out);
  } catch (err) {
    console.error('Claude fetch error:', err.message);
    return res.status(500).json({ error: 'Failed to reach AI service' });
  }
};

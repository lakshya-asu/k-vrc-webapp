// api/tts.js
const MAX_TEXT_LEN = 1000;

export default async function handler(req, res) {
  // CORS — identical to api/chat.js
  const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN;
  const origin = req.headers.origin || '';
  const isLocalDev = origin.startsWith('http://localhost') || origin.startsWith('http://127.0.0.1');
  const isVercel = origin.endsWith('.vercel.app');
  const isCustom = ALLOWED_ORIGIN && origin === ALLOWED_ORIGIN;
  if (!isLocalDev && !isVercel && !isCustom) return res.status(403).json({ error: 'Forbidden' });

  res.setHeader('Access-Control-Allow-Origin', origin);
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { text } = req.body || {};
  if (!text || typeof text !== 'string' || text.trim().length === 0) {
    return res.status(400).json({ error: 'text is required' });
  }
  if (text.length > MAX_TEXT_LEN) {
    return res.status(400).json({ error: 'Text too long' });
  }

  const apiKey = process.env.ELEVEN_LABS_API_KEY;
  const voiceId = process.env.ELEVEN_VOICE_ID;
  if (!apiKey || !voiceId) {
    console.error('ELEVEN_LABS_API_KEY or ELEVEN_VOICE_ID not set');
    return res.status(500).json({ error: 'TTS not configured' });
  }

  try {
    const elevenRes = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
      method: 'POST',
      headers: {
        'xi-api-key': apiKey,
        'Content-Type': 'application/json',
        'Accept': 'audio/mpeg',
      },
      body: JSON.stringify({
        text: text.trim(),
        model_id: 'eleven_flash_v2_5',
        voice_settings: { stability: 0.5, similarity_boost: 0.8, style: 0.0, use_speaker_boost: true },
      }),
    });

    if (!elevenRes.ok) {
      const err = await elevenRes.text();
      console.error('ElevenLabs TTS error:', elevenRes.status, err);
      return res.status(500).json({ error: 'TTS service error' });
    }

    const audioData = await elevenRes.arrayBuffer();
    res.setHeader('Content-Type', 'audio/mpeg');
    res.setHeader('Content-Length', audioData.byteLength);
    return res.end(Buffer.from(audioData));
  } catch (err) {
    console.error('TTS fetch error:', err.message);
    return res.status(500).json({ error: 'TTS request failed' });
  }
}

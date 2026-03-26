// scripts/create-elevenlabs-voice.mjs
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { join, dirname } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

const API_KEY = process.env.ELEVEN_LABS_API_KEY;
if (!API_KEY) { console.error('ELEVEN_LABS_API_KEY not set'); process.exit(1); }

const samplePath = join(__dirname, '../tools/kvrc-voice-sample.mp3');
const audioBytes = readFileSync(samplePath);
const form = new FormData();
form.append('name', 'K-VRC');
form.append('description', 'K-VRC robot character voice');
form.append('files', new Blob([audioBytes], { type: 'audio/mpeg' }), 'kvrc-voice-sample.mp3');

const res = await fetch('https://api.elevenlabs.io/v1/voices/add', {
  method: 'POST',
  headers: { 'xi-api-key': API_KEY },
  body: form,
});

if (!res.ok) {
  const err = await res.text();
  console.error('ElevenLabs error:', res.status, err);
  process.exit(1);
}

const data = await res.json();
console.log('\n✅ Voice created!');
console.log('voice_id:', data.voice_id);
console.log('\nAdd to .env:\nELEVEN_VOICE_ID=' + data.voice_id);

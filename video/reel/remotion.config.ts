import {Config} from '@remotion/cli/config';

Config.setVideoImageFormat('jpeg');
Config.setOverwriteOutput(true);
// The scene MP4s carry the AAC voice; keep audio untouched (they are
// loudness-normalized by tools/build-data.mjs before they reach public/).
Config.setCodec('h264');

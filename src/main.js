import * as THREE from 'three';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { EMOTION_MAP, applyEmotion } from './emotions.js';
import { initRobot, updateRobot } from './robot.js';
import { initChat } from './chat.js';
import './style.css';

// ── Renderer ─────────────────────────────────────────────────
const canvas = document.getElementById('three-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.8;

// ── Scene ─────────────────────────────────────────────────────
const scene = new THREE.Scene();

// ── Camera ────────────────────────────────────────────────────
const CAM_BASE = { y: 2.25, z: 4.4 };
const CAM_LOOK = { y: -0.65, z: -1.8 };
const CAM_MIN_Z = 3.2;   // closest
const CAM_MAX_Z = 6.5;   // furthest
const CAM_MIN_Y = 1.4;
const CAM_MAX_Y = 3.4;

const camera = new THREE.PerspectiveCamera(46, window.innerWidth / window.innerHeight, 0.1, 100);
let camZ = CAM_BASE.z;
let camY = CAM_BASE.y;
camera.position.set(0, camY, camZ);
camera.lookAt(0, CAM_LOOK.y, CAM_LOOK.z);

// Scroll: up = farther + higher, down = closer + lower
window.addEventListener('wheel', (e) => {
  const delta = e.deltaY * 0.004;
  camZ = THREE.MathUtils.clamp(camZ + delta, CAM_MIN_Z, CAM_MAX_Z);
  camY = THREE.MathUtils.clamp(camY + delta * 0.5, CAM_MIN_Y, CAM_MAX_Y);
  camera.position.set(0, camY, camZ);
  camera.lookAt(0, CAM_LOOK.y, CAM_LOOK.z);
}, { passive: true });

// ── HDR environment (lighting + background) ───────────────────
RectAreaLightUniformsLib.init();

new RGBELoader().load('/brown_photostudio_02_4k.hdr', (hdrTex) => {
  hdrTex.mapping = THREE.EquirectangularReflectionMapping;
  scene.environment = hdrTex;
});

// ── Background scene ──────────────────────────────────────────
new GLTFLoader().load('/lowpoly_room.glb', (gltf) => {
  const bg = gltf.scene;
  bg.scale.setScalar(0.008);
  bg.position.set(-1, -1.55, -2);
  bg.rotation.y = THREE.MathUtils.degToRad(11);
  bloomPass.strength = 0.5;

  bg.traverse(obj => {
    if (!obj.isMesh) return;
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    mats.forEach(m => { if (m) m.side = THREE.DoubleSide; });
  });

  scene.add(bg);
});

// ── Lights ────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0xffffff, 0.2));

const keyLight = new THREE.DirectionalLight(0xffeedd, 0.8);
keyLight.position.set(0.5, 3, 5);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xffffff, 0.2);
fillLight.position.set(-3, 1.5, 2);
scene.add(fillLight);

// Rim light from behind — emotion-reactive cyan accent
export const rimLight = new THREE.RectAreaLight(0x00e5ff, 1.0, 3, 4);
rimLight.position.set(0, 1.0, -1.5);
rimLight.lookAt(0, 0.6, 0);
scene.add(rimLight);

// Face fill — subtle cyan tint on face
export const faceLight = new THREE.PointLight(0x00e5ff, 0.3, 5);
faceLight.position.set(0, 1.4, 3);
scene.add(faceLight);

// ── Post-processing ───────────────────────────────────────────
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
export const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  0.5, 0.4, 0.92
);
composer.addPass(bloomPass);

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});

// ── Boot ──────────────────────────────────────────────────────
const clock = new THREE.Clock();

(async () => {
  try {
    const robot = await initRobot(scene, EMOTION_MAP);
    applyEmotion('neutral', { rimLight, faceLight, bloomPass });
    initChat(robot, { rimLight, faceLight, bloomPass });
  } catch (err) {
    console.error('K-VRC boot error:', err);
  } finally {
    document.getElementById('loading-overlay')?.classList.add('hidden');
  }

  renderer.setAnimationLoop(() => {
    updateRobot(clock.getDelta());
    composer.render();
  });
})();

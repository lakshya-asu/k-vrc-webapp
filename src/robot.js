import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EMOTION_MAP } from './emotions.js';
import { attachFaceScreen, updateFaceScreen, tickFaceScreen } from './faceScreen.js';
import { AnimationController } from './animationController.js';

const BONES = {
  head:    'Head',
  chest:   'Spine2',
  armL:    'LeftArm',
  forearmL:'LeftForeArm',
  armR:    'RightArm',
  forearmR:'RightForeArm',
};

let robotRoot = null;
let bones = {};
let baseY = 0;
let clock = 0;
let currentEmotionCfg = null;
let blinkTimer = rand(3, 6);
let targetHeadYaw = 0, targetHeadPitch = 0, targetChestYaw = 0;
let activeMotionType = null;
let activeMotionId = null;
let activeShakeTimeout = null;
let armRestQ = {};
const animCtrl = new AnimationController();

function rand(a, b) { return a + Math.random() * (b - a); }

export async function initRobot(scene, emotionMap) {
  currentEmotionCfg = emotionMap.neutral;

  const gltf = await new Promise((resolve, reject) =>
    new GLTFLoader().load('/models/kvrc.glb', resolve, undefined, reject));

  robotRoot = gltf.scene;
  scene.add(robotRoot);

  robotRoot.position.set(0, -1.4, 0);
  baseY = robotRoot.position.y;

  // Debug: log scene structure
  console.log('GLB children:', gltf.scene.children.map(c => `${c.name}(${c.type})`));
  console.log('Animations loaded:', gltf.animations?.map(a => a.name));

  // --- Deduplicate robot instances ---
  // Each complete robot in Blender exports as a direct child group of gltf.scene.
  // Keep only the one named 'KVRCArmature' (or last if not found); hide the rest.
  const rootChildren = [...robotRoot.children];

  // Case 1: multiple root-level groups each containing skinned meshes
  const robotInstances = rootChildren.filter(child => {
    let hasSkin = false;
    child.traverse(c => { if (c.isSkinnedMesh || c.isMesh) hasSkin = true; });
    return hasSkin;
  });
  if (robotInstances.length > 1) {
    console.warn(`${robotInstances.length} robot instances found — hiding extras`);
    const preferred = robotInstances.find(g => /kvrc/i.test(g.name))
      ?? robotInstances[robotInstances.length - 1];
    robotInstances.forEach(g => {
      if (g !== preferred) { g.visible = false; console.log('Hidden:', g.name); }
    });
  }

  // Case 2: multiple skinned meshes at the same root level (different material splits)
  // Only hide if they appear to be full-body duplicates (same vertex count).
  const skinnedMeshes = [];
  robotRoot.traverse(obj => { if (obj.isSkinnedMesh) skinnedMeshes.push(obj); });
  const vcounts = skinnedMeshes.map(m => m.geometry.attributes.position.count);
  const maxVC = Math.max(...vcounts);
  // If multiple meshes share the exact same (largest) vertex count → duplicates
  const fullBodyMeshes = skinnedMeshes.filter(m => m.geometry.attributes.position.count === maxVC);
  if (fullBodyMeshes.length > 1) {
    console.warn(`${fullBodyMeshes.length} identical full-body meshes — hiding extras`);
    fullBodyMeshes.slice(1).forEach(m => { m.visible = false; console.log('Hidden mesh:', m.name); });
  }

  robotRoot.traverse(obj => {
    if (!obj.isBone && obj.type !== 'Bone') return;
    for (const [key, name] of Object.entries(BONES)) {
      if (obj.name === name) bones[key] = obj;
    }
  });

  for (const key of ['armL', 'forearmL', 'armR', 'forearmR']) {
    if (bones[key]) armRestQ[key] = bones[key].quaternion.clone();
  }

  console.log('Bones resolved:', Object.fromEntries(
    Object.entries(BONES).map(([k]) => [k, !!bones[k]])
  ));

  // ── Init animation mixer with all NLA clips ─────────────────
  if (gltf.animations?.length) {
    animCtrl.init(robotRoot, gltf.animations);
    animCtrl.playGesture('idle');
  }

  // Attach face screen FIRST so we can skip it during material assignment
  const faceScreenMesh = attachFaceScreen(robotRoot, scene, bones.head);
  robotRoot.__faceScreenMesh = faceScreenMesh;

  // ── Apply K-VRC colors (Standard — responds to IBL from HDR environment) ──
  const armorMat = new THREE.MeshStandardMaterial({
    color: 0xe03818,
    emissive: 0x1a0400,
    roughness: 0.55,
    metalness: 0.1,
    side: THREE.DoubleSide,
  });
  const jointMat = new THREE.MeshStandardMaterial({
    color: 0x1e1e22,
    emissive: 0x050505,
    roughness: 0.4,
    metalness: 0.6,
    side: THREE.DoubleSide,
  });

  let colored = 0;
  robotRoot.traverse(obj => {
    if (!obj.isMesh) return;
    if (obj === faceScreenMesh) return;
    const n = obj.name.toLowerCase();
    const isJoint = ['neck','hand','toe','foot','hips','pelvis'].some(k => n.includes(k));
    obj.material = isJoint ? jointMat : armorMat;
    colored++;
  });
  console.log(`K-VRC: colored ${colored} mesh objects`);

  window.addEventListener('mousemove', e => {
    const nx =  (e.clientX / window.innerWidth)  * 2 - 1;
    const ny = -(e.clientY / window.innerHeight) * 2 + 1;
    targetHeadYaw   = nx * THREE.MathUtils.degToRad(22);
    targetHeadPitch = -ny * THREE.MathUtils.degToRad(12);
    targetChestYaw  = nx * THREE.MathUtils.degToRad(7);
  });

  return {
    root: robotRoot,
    bones,
    setEmotionCfg: cfg => { currentEmotionCfg = cfg; },
    startBodyMotion,
    triggerHeadJerk,
    playGesture: name => animCtrl.playGesture(name),
    get faceScreenMesh() { return faceScreenMesh; },
  };
}

export function updateRobot(delta) {
  if (!robotRoot) return;
  clock += delta;

  // Float
  if (activeMotionType !== 'bob' && currentEmotionCfg) {
    robotRoot.position.y = baseY
      + Math.sin(clock * currentEmotionCfg.floatSpeed) * currentEmotionCfg.floatAmp;
  }

  // Head + chest tracking
  const headLerp = currentEmotionCfg?.headRapid ? 0.15 : 0.05;
  if (bones.head) {
    bones.head.rotation.y = THREE.MathUtils.lerp(bones.head.rotation.y, targetHeadYaw, headLerp);
    if (activeMotionType !== 'nod')
      bones.head.rotation.x = THREE.MathUtils.lerp(bones.head.rotation.x, targetHeadPitch, headLerp);
  }
  if (bones.chest) {
    bones.chest.rotation.y = THREE.MathUtils.lerp(bones.chest.rotation.y, targetChestYaw, 0.02);
    if (activeMotionType !== 'slump')
      bones.chest.rotation.x = THREE.MathUtils.lerp(bones.chest.rotation.x, 0, 0.03);
  }
  if (activeMotionType !== 'tilt' && bones.head)
    bones.head.rotation.z = THREE.MathUtils.lerp(bones.head.rotation.z, 0, 0.04);

  // Mixer
  animCtrl.update(delta);

  // Face
  tickFaceScreen(delta * 1000);
  updateFaceScreen(robotRoot.__faceScreenMesh);
}

export function startBodyMotion(name) {
  if (activeMotionId) { cancelAnimationFrame(activeMotionId); activeMotionId = null; }
  if (activeShakeTimeout) { clearTimeout(activeShakeTimeout); activeShakeTimeout = null; }
  activeMotionType = name;
  switch (name) {
    case 'sway':  motionSway();  break;
    case 'nod':   motionNod();   break;
    case 'bob':   motionBob();   break;
    case 'slump': motionSlump(); break;
    case 'shake': motionShake(); break;
    case 'tilt':  motionTilt();  break;
  }
}

export function triggerHeadJerk() {
  if (!bones.head) return;
  const orig = bones.head.rotation.y;
  let step = 0;
  const JERK = [0.18, -0.18, 0.1, -0.1, 0];
  const next = () => {
    if (step >= JERK.length) return;
    bones.head.rotation.y = orig + JERK[step++];
    setTimeout(next, 60);
  };
  next();
}

function motionSway() {
  let t = 0;
  const loop = () => {
    if (activeMotionType !== 'sway') return;
    t += 0.016;
    if (bones.chest) bones.chest.rotation.z = Math.sin(t * 0.6) * THREE.MathUtils.degToRad(2);
    activeMotionId = requestAnimationFrame(loop);
  };
  loop();
}

function motionNod() {
  let t = 0;
  const loop = () => {
    if (activeMotionType !== 'nod') return;
    t += 0.016;
    if (bones.head)
      bones.head.rotation.x = THREE.MathUtils.lerp(
        bones.head.rotation.x, Math.sin(t * 1.2) * THREE.MathUtils.degToRad(6), 0.08);
    activeMotionId = requestAnimationFrame(loop);
  };
  loop();
}

function motionBob() {
  let t = 0;
  const loop = () => {
    if (activeMotionType !== 'bob') return;
    t += 0.016;
    const s = Math.sin(t * 2.5);
    robotRoot.position.y = baseY + s * 0.04;
    activeMotionId = requestAnimationFrame(loop);
  };
  loop();
}

function motionSlump() {
  const TARGET = THREE.MathUtils.degToRad(6);
  const loop = () => {
    if (activeMotionType !== 'slump') return;
    if (bones.chest) {
      if (Math.abs(bones.chest.rotation.x - TARGET) < 0.001) { bones.chest.rotation.x = TARGET; return; }
      bones.chest.rotation.x = THREE.MathUtils.lerp(bones.chest.rotation.x, TARGET, 0.04);
    }
    activeMotionId = requestAnimationFrame(loop);
  };
  loop();
}

function motionShake() {
  const SHAKE = [10,-10,8,-8,5,-5,0].map(d => THREE.MathUtils.degToRad(d));
  let step = 0;
  const next = () => {
    if (activeMotionType !== 'shake' || step >= SHAKE.length) return;
    if (bones.head) bones.head.rotation.y = SHAKE[step++];
    activeShakeTimeout = setTimeout(next, 55);
  };
  next();
}

function motionTilt() {
  const loop = () => {
    if (activeMotionType !== 'tilt') return;
    if (bones.head)
      bones.head.rotation.z = THREE.MathUtils.lerp(
        bones.head.rotation.z, THREE.MathUtils.degToRad(8), 0.05);
    activeMotionId = requestAnimationFrame(loop);
  };
  loop();
}

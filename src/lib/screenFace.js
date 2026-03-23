import * as THREE from 'three'
import { expressions } from './expressions.js'

export class ScreenFace {
  constructor() {
    this.canvas = document.createElement('canvas')
    this.canvas.width = 512
    this.canvas.height = 512
    this.ctx = this.canvas.getContext('2d')

    this.texture = new THREE.CanvasTexture(this.canvas)
    this.texture.minFilter = THREE.NearestFilter
    this.texture.magFilter = THREE.NearestFilter
    this.texture.colorSpace = THREE.SRGBColorSpace

    this.material = new THREE.MeshStandardMaterial({
      emissiveMap: this.texture,
      emissive: new THREE.Color(1, 1, 1),
      emissiveIntensity: 1.2,
      color: new THREE.Color(0, 0, 0),
      toneMapped: false,
    })

    this.current = 'neutral'
    this.t = 0
    this.amplitude = 0  // for speaking waveform
  }

  // Apply to the ScreenFace mesh found in a loaded GLTF
  applyToModel(gltf) {
    gltf.scene.traverse((node) => {
      if (node.isMesh) {
        // Target by material name OR mesh name
        const mats = Array.isArray(node.material) ? node.material : [node.material]
        mats.forEach((mat, i) => {
          if (mat && mat.name === 'ScreenFace') {
            const mats2 = Array.isArray(node.material) ? [...node.material] : [node.material]
            mats2[i] = this.material
            node.material = mats2.length === 1 ? mats2[0] : mats2
          }
        })
      }
    })
  }

  setExpression(name) {
    if (expressions[name]) {
      this.current = name
    }
  }

  setAmplitude(val) {
    this.amplitude = Math.max(0, Math.min(1, val))
  }

  // Call every frame from useFrame
  update(delta) {
    this.t += delta
    const draw = expressions[this.current]
    if (draw) {
      draw(this.ctx, this.t, this.amplitude)
      this.texture.needsUpdate = true
    }
  }

  dispose() {
    this.texture.dispose()
    this.material.dispose()
  }
}

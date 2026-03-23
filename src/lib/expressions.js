// Canvas 2D draw functions for K-VRC screen face
// Canvas is 512x512. Dark background, glowing colored features.
// Style reference: simple geometric shapes, LED/pixel aesthetic

const BG = '#020d14'
const CYAN = '#00e5ff'
const GREEN = '#00ff88'
const YELLOW = '#ffe600'
const ORANGE = '#ff8800'
const RED = '#ff2200'
const BLUE = '#4488ff'
const PURPLE = '#bb44ff'
const PINK = '#ff44cc'
const WHITE = '#ffffff'
const TEAL = '#00ccbb'

function clear(ctx) {
  ctx.fillStyle = BG
  ctx.fillRect(0, 0, 512, 512)
}

function glow(ctx, color, blur = 18) {
  ctx.shadowColor = color
  ctx.shadowBlur = blur
}

// Eye helpers
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath()
  ctx.roundRect(x, y, w, h, r)
  ctx.fill()
}

function arc(ctx, cx, cy, rx, ry, startAngle, endAngle) {
  ctx.beginPath()
  ctx.ellipse(cx, cy, rx, ry, 0, startAngle, endAngle)
  ctx.stroke()
}

export const expressions = {
  neutral(ctx) {
    clear(ctx)
    ctx.fillStyle = CYAN
    glow(ctx, CYAN)
    // Two simple dots
    ctx.beginPath(); ctx.arc(168, 256, 28, 0, Math.PI * 2); ctx.fill()
    ctx.beginPath(); ctx.arc(344, 256, 28, 0, Math.PI * 2); ctx.fill()
  },

  happy(ctx) {
    clear(ctx)
    ctx.fillStyle = GREEN
    ctx.strokeStyle = GREEN
    ctx.lineWidth = 14
    glow(ctx, GREEN, 22)
    // ^_^ style arcs
    arc(ctx, 168, 270, 44, 36, Math.PI, 0)
    arc(ctx, 344, 270, 44, 36, Math.PI, 0)
    // Small smile
    ctx.strokeStyle = GREEN
    arc(ctx, 256, 330, 60, 30, 0, Math.PI)
  },

  sad(ctx) {
    clear(ctx)
    ctx.fillStyle = BLUE
    ctx.strokeStyle = BLUE
    ctx.lineWidth = 12
    glow(ctx, BLUE, 20)
    // Droopy eyes (downward arcs)
    arc(ctx, 168, 240, 44, 30, 0, Math.PI)
    arc(ctx, 344, 240, 44, 30, 0, Math.PI)
    // Frown
    ctx.strokeStyle = BLUE
    arc(ctx, 256, 370, 55, 28, Math.PI, 0)
  },

  excited(ctx, t = 0) {
    clear(ctx)
    ctx.fillStyle = YELLOW
    ctx.strokeStyle = YELLOW
    ctx.lineWidth = 10
    glow(ctx, YELLOW, 26)
    // Wide circle eyes
    ctx.beginPath(); ctx.arc(168, 256, 46, 0, Math.PI * 2); ctx.stroke()
    ctx.beginPath(); ctx.arc(344, 256, 46, 0, Math.PI * 2); ctx.stroke()
    // Pulsing inner dots
    const pulse = 0.7 + 0.3 * Math.sin(t * 6)
    ctx.beginPath(); ctx.arc(168, 256, 14 * pulse, 0, Math.PI * 2); ctx.fill()
    ctx.beginPath(); ctx.arc(344, 256, 14 * pulse, 0, Math.PI * 2); ctx.fill()
    // "!!" text
    ctx.font = 'bold 72px monospace'
    ctx.textAlign = 'center'
    ctx.fillText('!!', 256, 380)
  },

  thinking(ctx, t = 0) {
    clear(ctx)
    ctx.fillStyle = PURPLE
    ctx.strokeStyle = PURPLE
    ctx.lineWidth = 12
    glow(ctx, PURPLE, 18)
    // One squinting eye, one normal
    roundRect(ctx, 128, 228, 80, 22, 4)  // squint left
    ctx.beginPath(); ctx.arc(344, 248, 32, 0, Math.PI * 2); ctx.stroke()
    // Animated "..."
    ctx.font = 'bold 56px monospace'
    ctx.textAlign = 'center'
    ctx.fillStyle = PURPLE
    const dots = '.'.repeat(1 + (Math.floor(t * 2) % 3))
    ctx.fillText(dots, 256, 380)
  },

  confused(ctx) {
    clear(ctx)
    ctx.fillStyle = ORANGE
    ctx.strokeStyle = ORANGE
    ctx.lineWidth = 12
    glow(ctx, ORANGE, 20)
    // Mismatched eyes
    ctx.beginPath(); ctx.arc(168, 248, 38, 0, Math.PI * 2); ctx.stroke()
    ctx.beginPath(); ctx.arc(344, 248, 22, 0, Math.PI * 2); ctx.stroke()
    // WTF text
    ctx.font = 'bold 64px monospace'
    ctx.textAlign = 'center'
    ctx.fillStyle = ORANGE
    ctx.fillText('WTF', 256, 380)
  },

  angry(ctx) {
    clear(ctx)
    ctx.fillStyle = RED
    ctx.strokeStyle = RED
    ctx.lineWidth = 14
    glow(ctx, RED, 24)
    // Sharp V-shape brows + narrow eyes
    ctx.beginPath()
    ctx.moveTo(128, 210); ctx.lineTo(168, 230); ctx.lineTo(208, 210)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(304, 210); ctx.lineTo(344, 230); ctx.lineTo(384, 210)
    ctx.stroke()
    // Narrow eyes
    roundRect(ctx, 130, 240, 78, 18, 3)
    roundRect(ctx, 306, 240, 78, 18, 3)
    // Zigzag mouth
    ctx.beginPath()
    ctx.moveTo(176, 350)
    for (let i = 0; i < 5; i++) {
      ctx.lineTo(196 + i * 28, i % 2 === 0 ? 330 : 370)
    }
    ctx.lineTo(336, 350)
    ctx.stroke()
  },

  listening(ctx, t = 0) {
    clear(ctx)
    ctx.fillStyle = TEAL
    ctx.strokeStyle = TEAL
    ctx.lineWidth = 10
    glow(ctx, TEAL, 16)
    // Half-closed eyes
    roundRect(ctx, 130, 248, 78, 20, 10)
    roundRect(ctx, 306, 248, 78, 20, 10)
    // Small waveform
    const bars = 7
    const barW = 14
    const spacing = 24
    const startX = 256 - (bars * spacing) / 2
    for (let i = 0; i < bars; i++) {
      const h = 10 + 14 * Math.abs(Math.sin(t * 3 + i * 0.8))
      roundRect(ctx, startX + i * spacing, 348 - h / 2, barW, h, 3)
    }
  },

  speaking(ctx, t = 0, amplitude = 0.5) {
    clear(ctx)
    ctx.fillStyle = WHITE
    ctx.strokeStyle = WHITE
    ctx.lineWidth = 10
    glow(ctx, WHITE, 14)
    // Normal eyes
    ctx.beginPath(); ctx.arc(168, 248, 30, 0, Math.PI * 2); ctx.fill()
    ctx.beginPath(); ctx.arc(344, 248, 30, 0, Math.PI * 2); ctx.fill()
    // Audio waveform driven by amplitude
    ctx.strokeStyle = WHITE
    ctx.lineWidth = 5
    ctx.beginPath()
    const w = 220, cx = 256, cy = 360
    ctx.moveTo(cx - w / 2, cy)
    for (let x = 0; x <= w; x += 4) {
      const phase = (x / w) * Math.PI * 4 + t * 8
      const y = cy + Math.sin(phase) * 28 * amplitude
      ctx.lineTo(cx - w / 2 + x, y)
    }
    ctx.stroke()
  },

  love(ctx, t = 0) {
    clear(ctx)
    ctx.fillStyle = PINK
    glow(ctx, PINK, 24)
    // Heart eyes using two arcs + triangle
    function heart(cx, cy, size) {
      ctx.save()
      ctx.translate(cx, cy)
      ctx.scale(size, size)
      ctx.beginPath()
      ctx.moveTo(0, 10)
      ctx.bezierCurveTo(-30, -10, -50, 10, -30, 30)
      ctx.bezierCurveTo(-15, 45,   0,  50,  0, 50)
      ctx.bezierCurveTo( 15, 50,  30, 45,  30, 30)
      ctx.bezierCurveTo( 50, 10,  30,-10,   0, 10)
      ctx.fill()
      ctx.restore()
    }
    const pulse = 0.85 + 0.15 * Math.sin(t * 4)
    heart(168, 230, pulse * 0.7)
    heart(344, 230, pulse * 0.7)
  },
}

export const EXPRESSION_NAMES = Object.keys(expressions)

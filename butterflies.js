/**
 * MorphoLLM: Ethereal Butterfly & Stardust Canvas Simulation
 * Realistic fluttering, iridescent wing gradients, stardust trails, and ambient physics.
 */

class ButterflySimulation {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.butterflies = [];
    this.stardust = [];
    this.active = true;
    this.mouse = { x: -1000, y: -1000, active: false };
    this.lastTime = performance.now();

    this.init();
  }

  init() {
    this.resize();
    window.addEventListener('resize', () => this.resize());
    
    // Track gentle mouse movement
    window.addEventListener('mousemove', (e) => {
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
      this.mouse.active = true;
      // Occasionally emit stardust on mouse move
      if (this.active && Math.random() < 0.25) {
        this.addStardust(e.clientX, e.clientY, 1, 1.2, '#caf0f8');
      }
    });

    window.addEventListener('mouseleave', () => {
      this.mouse.active = false;
    });

    // Gentle butterfly burst on click
    window.addEventListener('click', (e) => {
      if (!this.active) return;
      this.createBurst(e.clientX, e.clientY, 12);
    });

    // Populate ambient butterflies
    const count = window.innerWidth < 768 ? 6 : 14;
    for (let i = 0; i < count; i++) {
      this.butterflies.push(this.createButterfly());
    }

    // Start animation loop
    requestAnimationFrame((t) => this.render(t));
  }

  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  createButterfly(startX, startY) {
    const isMobile = window.innerWidth < 768;
    const size = (isMobile ? 12 : 16) + Math.random() * 12;
    return {
      x: startX !== undefined ? startX : Math.random() * window.innerWidth,
      y: startY !== undefined ? startY : Math.random() * window.innerHeight,
      size: size,
      vx: (Math.random() - 0.5) * 1.4,
      vy: (Math.random() - 0.5) * 1.2 - 0.3, // slight upward float
      heading: Math.random() * Math.PI * 2,
      targetHeading: Math.random() * Math.PI * 2,
      wingPhase: Math.random() * Math.PI * 2,
      flapSpeed: 0.16 + Math.random() * 0.12,
      glideTimer: 0,
      opacity: 0.75 + Math.random() * 0.25,
      // Colors: Navy blue, sea blue, light blue, ethereal white
      palette: this.getRandomPalette()
    };
  }

  getRandomPalette() {
    const palettes = [
      {
        tip: '#caf0f8',
        mid: '#00b4d8',
        base: '#0077b6',
        deep: '#03045e',
        glow: 'rgba(72, 202, 228, 0.7)'
      },
      {
        tip: '#e0f2fe',
        mid: '#38bdf8',
        base: '#0284c7',
        deep: '#0c2340',
        glow: 'rgba(56, 189, 248, 0.8)'
      },
      {
        tip: '#f0f9ff',
        mid: '#48cae4',
        base: '#0a9396',
        deep: '#005f73',
        glow: 'rgba(10, 147, 150, 0.75)'
      },
      {
        tip: '#f8fafc',
        mid: '#818cf8',
        base: '#0077b6',
        deep: '#020617',
        glow: 'rgba(129, 140, 248, 0.65)'
      }
    ];
    return palettes[Math.floor(Math.random() * palettes.length)];
  }

  addStardust(x, y, count = 1, sizeMult = 1, color = '#caf0f8') {
    for (let i = 0; i < count; i++) {
      this.stardust.push({
        x: x + (Math.random() - 0.5) * 10,
        y: y + (Math.random() - 0.5) * 10,
        vx: (Math.random() - 0.5) * 0.8,
        vy: (Math.random() - 0.5) * 0.8 + 0.2,
        size: (1 + Math.random() * 2.2) * sizeMult,
        alpha: 0.9,
        decay: 0.012 + Math.random() * 0.018,
        color: color
      });
    }
  }

  createBurst(x, y, count = 8) {
    for (let i = 0; i < count; i++) {
      const angle = (Math.PI * 2 / count) * i + Math.random() * 0.5;
      const speed = 1.5 + Math.random() * 2.5;
      this.stardust.push({
        x: x,
        y: y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        size: 2.5 + Math.random() * 2.5,
        alpha: 1,
        decay: 0.015,
        color: Math.random() > 0.4 ? '#48cae4' : '#e0f2fe'
      });
    }
  }

  toggle() {
    this.active = !this.active;
    if (!this.active) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    return this.active;
  }

  render(timestamp) {
    requestAnimationFrame((t) => this.render(t));
    if (!this.active) return;

    const dt = Math.min((timestamp - this.lastTime) / 1000, 0.1);
    this.lastTime = timestamp;

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // 1. Update & Render Stardust
    for (let i = this.stardust.length - 1; i >= 0; i--) {
      const p = this.stardust[i];
      p.x += p.vx;
      p.y += p.vy;
      p.alpha -= p.decay;

      if (p.alpha <= 0) {
        this.stardust.splice(i, 1);
        continue;
      }

      this.ctx.save();
      this.ctx.globalAlpha = p.alpha;
      this.ctx.fillStyle = p.color;
      this.ctx.shadowBlur = 6;
      this.ctx.shadowColor = p.color;
      this.ctx.beginPath();
      this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.restore();
    }

    // Keep stardust count bounded
    if (this.stardust.length > 250) {
      this.stardust.splice(0, this.stardust.length - 250);
    }

    // 2. Update & Render Butterflies
    for (let b of this.butterflies) {
      // Natural steering & gentle wind
      if (Math.random() < 0.03) {
        b.targetHeading += (Math.random() - 0.5) * 1.2;
      }

      // Gentle attraction towards mouse if within 250px
      if (this.mouse.active) {
        const dx = this.mouse.x - b.x;
        const dy = this.mouse.y - b.y;
        const dist = Math.hypot(dx, dy);
        if (dist < 260 && dist > 30) {
          b.targetHeading = Math.atan2(dy, dx) + (Math.random() - 0.5) * 0.5;
        } else if (dist <= 30) {
          // Flutter away if too close
          b.targetHeading = Math.atan2(-dy, -dx);
        }
      }

      // Smooth heading interpolation
      let angleDiff = b.targetHeading - b.heading;
      while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;
      while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
      b.heading += angleDiff * 0.05;

      // Speed & movement
      const speed = 1.3;
      b.vx = Math.cos(b.heading) * speed;
      b.vy = Math.sin(b.heading) * speed - 0.2; // slight upward lift

      b.x += b.vx;
      b.y += b.vy;

      // Wrap around canvas edges with margin
      const margin = 50;
      if (b.x < -margin) b.x = this.canvas.width + margin;
      if (b.x > this.canvas.width + margin) b.x = -margin;
      if (b.y < -margin) b.y = this.canvas.height + margin;
      if (b.y > this.canvas.height + margin) b.y = -margin;

      // Flapping logic with occasional gliding pause
      b.wingPhase += b.flapSpeed;
      const wingScale = Math.sin(b.wingPhase); // -1 to 1

      // Emit stardust trail occasionally from wingtips
      if (Math.random() < 0.18) {
        this.addStardust(b.x, b.y, 1, 0.9, b.palette.tip);
      }

      // Render Butterfly
      this.drawButterfly(b, wingScale);
    }
  }

  drawButterfly(b, wingScale) {
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(b.x, b.y);
    ctx.rotate(b.heading + Math.PI / 2);
    ctx.globalAlpha = b.opacity;

    const s = b.size;
    const flapWidth = Math.abs(wingScale); // 0 (closed) to 1 (flat open)

    // Outer Glow / Ethereal Shadow
    ctx.shadowColor = b.palette.glow;
    ctx.shadowBlur = 10;

    // --- Right Wings ---
    ctx.save();
    ctx.scale(flapWidth, 1);
    this.drawForewing(ctx, s, b.palette, 1);
    this.drawHindwing(ctx, s, b.palette, 1);
    ctx.restore();

    // --- Left Wings ---
    ctx.save();
    ctx.scale(-flapWidth, 1);
    this.drawForewing(ctx, s, b.palette, -1);
    this.drawHindwing(ctx, s, b.palette, -1);
    ctx.restore();

    // --- Body & Antennae ---
    ctx.shadowBlur = 4;
    ctx.fillStyle = '#e0f2fe';
    ctx.beginPath();
    ctx.ellipse(0, 0, s * 0.1, s * 0.45, 0, 0, Math.PI * 2);
    ctx.fill();

    // Antennae
    ctx.strokeStyle = '#caf0f8';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(-1, -s * 0.35);
    ctx.quadraticCurveTo(-s * 0.3, -s * 0.8, -s * 0.4, -s * 0.75);
    ctx.moveTo(1, -s * 0.35);
    ctx.quadraticCurveTo(s * 0.3, -s * 0.8, s * 0.4, -s * 0.75);
    ctx.stroke();

    ctx.restore();
  }

  drawForewing(ctx, s, palette, side) {
    const grad = ctx.createRadialGradient(s * 0.2, -s * 0.3, 2, s * 0.5, -s * 0.5, s * 1.3);
    grad.addColorStop(0, palette.tip);
    grad.addColorStop(0.35, palette.mid);
    grad.addColorStop(0.7, palette.base);
    grad.addColorStop(1, palette.deep);

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(0, -s * 0.1);
    ctx.bezierCurveTo(s * 0.4, -s * 0.8, s * 1.1, -s * 0.9, s * 1.25, -s * 0.4);
    ctx.bezierCurveTo(s * 1.3, -s * 0.1, s * 0.9, s * 0.2, s * 0.2, 0);
    ctx.closePath();
    ctx.fill();

    // Delicate wing veins
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(0, -s * 0.1);
    ctx.quadraticCurveTo(s * 0.6, -s * 0.4, s * 1.1, -s * 0.5);
    ctx.moveTo(0, -s * 0.1);
    ctx.quadraticCurveTo(s * 0.5, -s * 0.2, s * 0.9, -s * 0.1);
    ctx.stroke();
  }

  drawHindwing(ctx, s, palette, side) {
    const grad = ctx.createRadialGradient(s * 0.2, s * 0.2, 1, s * 0.4, s * 0.4, s * 0.9);
    grad.addColorStop(0, palette.tip);
    grad.addColorStop(0.4, palette.mid);
    grad.addColorStop(0.8, palette.base);
    grad.addColorStop(1, palette.deep);

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.bezierCurveTo(s * 0.4, s * 0.1, s * 0.85, s * 0.3, s * 0.75, s * 0.75);
    ctx.bezierCurveTo(s * 0.65, s * 1.05, s * 0.2, s * 0.85, 0, s * 0.35);
    ctx.closePath();
    ctx.fill();
  }
}

// Global initialization
window.butterflySim = null;
document.addEventListener('DOMContentLoaded', () => {
  window.butterflySim = new ButterflySimulation('butterflyCanvas');
});

// services/wallpaper.service.ts

export interface WallpaperServiceOptions {
    velocity?: number;
    density?: number;
    netLineDistance?: number;
    netLineColor?: string;
    particleColors?: string[];
    spawnQuantity?: number;
}

type Nullable<T> = T | null;

class Particle {
    x: number;
    y: number;
    radius: number;
    opacity: number;
    velocity: { x: number; y: number };
    particleColor: string;

    constructor(
        private network: ParticleNetwork,
        x?: number,
        y?: number
    ) {
        this.particleColor = this.returnRandomArrayItem(
            this.network.options.particleColors
        );
        this.radius = this.getLimitedRandom(1.5, 2.5);
        this.opacity = 0;

        this.x = x ?? Math.random() * this.network.canvas.width;
        this.y = y ?? Math.random() * this.network.canvas.height;

        this.velocity = {
            x: (Math.random() - 0.5) * this.network.options.velocity,
            y: (Math.random() - 0.5) * this.network.options.velocity,
        };
    }

    update() {
        if (this.opacity < 1) this.opacity += 0.01;
        else this.opacity = 1;

        if (
            this.x > this.network.canvas.width + 100 ||
            this.x < -100
        ) {
            this.velocity.x = -this.velocity.x;
        }

        if (
            this.y > this.network.canvas.height + 100 ||
            this.y < -100
        ) {
            this.velocity.y = -this.velocity.y;
        }

        this.x += this.velocity.x;
        this.y += this.velocity.y;
    }

    draw() {
        const ctx = this.network.ctx;
        ctx.beginPath();
        ctx.fillStyle = this.particleColor;
        ctx.globalAlpha = this.opacity;
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fill();
    }

    private getLimitedRandom(min: number, max: number) {
        return Math.random() * (max - min) + min;
    }

    private returnRandomArrayItem(array: string[]) {
        return array[Math.floor(Math.random() * array.length)];
    }
}

class ParticleNetwork {
    options: Required<WallpaperServiceOptions>;
    canvas: HTMLCanvasElement;
    ctx: CanvasRenderingContext2D;
    particles: Particle[] = [];
    interactionParticle?: Particle;
    animationFrame = 0;
    createIntervalId: ReturnType<typeof setInterval> | null = null;

    spawnQuantity = 3;
    mouseIsDown = false;
    touchIsMoving = false;

    onMouseMove?: (e: MouseEvent) => void;
    onTouchMove?: (e: TouchEvent) => void;
    onMouseDown?: (e: MouseEvent) => void;
    onTouchStart?: (e: TouchEvent) => void;
    onMouseUp?: (e: MouseEvent) => void;
    onMouseOut?: (e: MouseEvent) => void;
    onTouchEnd?: (e: TouchEvent) => void;

    constructor(
        private root: HTMLElement,
        options?: WallpaperServiceOptions
    ) {
        this.options = {
            velocity: options?.velocity ?? 1,
            density: options?.density ?? 15000,
            netLineDistance: options?.netLineDistance ?? 200,
            netLineColor: options?.netLineColor ?? "#929292",
            particleColors: options?.particleColors ?? ["#aaa"],
            spawnQuantity: options?.spawnQuantity ?? 3,
        };

        this.canvas = document.createElement("canvas");
        const ctx = this.canvas.getContext("2d");
        if (!ctx) {
            throw new Error("No se pudo crear el contexto 2D del canvas.");
        }
        this.ctx = ctx;

        this.root.appendChild(this.canvas);
        this.sizeCanvas();
        this.createParticles(true);
        this.bindUiActions();
        this.update = this.update.bind(this);
        this.animationFrame = requestAnimationFrame(this.update);
    }

    sizeCanvas() {
        this.canvas.width = this.root.offsetWidth;
        this.canvas.height = this.root.offsetHeight;
    }

    createParticles(isInitial = false) {
        this.particles = [];
        const quantity =
            (this.canvas.width * this.canvas.height) / this.options.density;

        if (isInitial) {
            let counter = 0;

            if (this.createIntervalId) clearInterval(this.createIntervalId);

            this.createIntervalId = setInterval(() => {
                if (counter < quantity - 1) {
                    this.particles.push(new Particle(this));
                } else if (this.createIntervalId) {
                    clearInterval(this.createIntervalId);
                    this.createIntervalId = null;
                }
                counter++;
            }, 250);
        } else {
            for (let i = 0; i < quantity; i++) {
                this.particles.push(new Particle(this));
            }
        }
    }

    createInteractionParticle() {
        this.interactionParticle = new Particle(this);
        this.interactionParticle.velocity = { x: 0, y: 0 };
        this.particles.push(this.interactionParticle);
        return this.interactionParticle;
    }

    removeInteractionParticle() {
        if (!this.interactionParticle) return;

        const index = this.particles.indexOf(this.interactionParticle);
        if (index > -1) {
            this.particles.splice(index, 1);
        }
        this.interactionParticle = undefined;
    }

    update() {
        if (!this.canvas) {
            cancelAnimationFrame(this.animationFrame);
            return;
        }

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.globalAlpha = 1;

        for (let i = 0; i < this.particles.length; i++) {
            for (let j = this.particles.length - 1; j > i; j--) {
                const p1 = this.particles[i];
                const p2 = this.particles[j];

                let distance = Math.min(
                    Math.abs(p1.x - p2.x),
                    Math.abs(p1.y - p2.y)
                );

                if (distance > this.options.netLineDistance) continue;

                distance = Math.sqrt(
                    Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2)
                );

                if (distance > this.options.netLineDistance) continue;

                this.ctx.beginPath();
                this.ctx.strokeStyle = this.options.netLineColor;
                this.ctx.globalAlpha =
                    ((this.options.netLineDistance - distance) /
                        this.options.netLineDistance) *
                    p1.opacity *
                    p2.opacity;
                this.ctx.lineWidth = 0.7;
                this.ctx.moveTo(p1.x, p1.y);
                this.ctx.lineTo(p2.x, p2.y);
                this.ctx.stroke();
            }
        }

        for (let i = 0; i < this.particles.length; i++) {
            this.particles[i].update();
            this.particles[i].draw();
        }

        if (this.options.velocity !== 0) {
            this.animationFrame = requestAnimationFrame(this.update);
        }
    }

    bindUiActions() {
        this.onMouseMove = (e: MouseEvent) => {
            if (!this.interactionParticle) this.createInteractionParticle();

            if (this.interactionParticle) {
                const rect = this.canvas.getBoundingClientRect();
                this.interactionParticle.x = e.clientX - rect.left;
                this.interactionParticle.y = e.clientY - rect.top;
            }
        };

        this.onTouchMove = (e: TouchEvent) => {
            e.preventDefault();
            this.touchIsMoving = true;

            const touch = e.changedTouches[0];
            if (!this.interactionParticle) this.createInteractionParticle();

            if (this.interactionParticle) {
                const rect = this.canvas.getBoundingClientRect();
                this.interactionParticle.x = touch.clientX - rect.left;
                this.interactionParticle.y = touch.clientY - rect.top;
            }
        };

        this.onMouseDown = () => {
            this.mouseIsDown = true;
            let counter = 0;
            let quantity = this.spawnQuantity;

            const intervalId = setInterval(() => {
                if (!this.mouseIsDown) {
                    clearInterval(intervalId);
                    return;
                }

                if (counter === 1) quantity = 1;

                for (let i = 0; i < quantity; i++) {
                    if (this.interactionParticle) {
                        this.particles.push(
                            new Particle(
                                this,
                                this.interactionParticle.x,
                                this.interactionParticle.y
                            )
                        );
                    }
                }

                counter++;
            }, 50);
        };

        this.onTouchStart = (e: TouchEvent) => {
            e.preventDefault();
            setTimeout(() => {
                if (!this.touchIsMoving) {
                    const touch = e.changedTouches[0];
                    const rect = this.canvas.getBoundingClientRect();

                    for (let i = 0; i < this.spawnQuantity; i++) {
                        this.particles.push(
                            new Particle(
                                this,
                                touch.clientX - rect.left,
                                touch.clientY - rect.top
                            )
                        );
                    }
                }
            }, 200);
        };

        this.onMouseUp = () => {
            this.mouseIsDown = false;
        };

        this.onMouseOut = () => {
            this.removeInteractionParticle();
        };

        this.onTouchEnd = (e: TouchEvent) => {
            e.preventDefault();
            this.touchIsMoving = false;
            this.removeInteractionParticle();
        };

        this.canvas.addEventListener("mousemove", this.onMouseMove);
        this.canvas.addEventListener("touchmove", this.onTouchMove, { passive: false });
        this.canvas.addEventListener("mousedown", this.onMouseDown);
        this.canvas.addEventListener("touchstart", this.onTouchStart, { passive: false });
        this.canvas.addEventListener("mouseup", this.onMouseUp);
        this.canvas.addEventListener("mouseout", this.onMouseOut);
        this.canvas.addEventListener("touchend", this.onTouchEnd);
    }

    destroy() {
        cancelAnimationFrame(this.animationFrame);

        if (this.createIntervalId) {
            clearInterval(this.createIntervalId);
            this.createIntervalId = null;
        }

        if (this.onMouseMove) this.canvas.removeEventListener("mousemove", this.onMouseMove);
        if (this.onTouchMove) this.canvas.removeEventListener("touchmove", this.onTouchMove);
        if (this.onMouseDown) this.canvas.removeEventListener("mousedown", this.onMouseDown);
        if (this.onTouchStart) this.canvas.removeEventListener("touchstart", this.onTouchStart);
        if (this.onMouseUp) this.canvas.removeEventListener("mouseup", this.onMouseUp);
        if (this.onMouseOut) this.canvas.removeEventListener("mouseout", this.onMouseOut);
        if (this.onTouchEnd) this.canvas.removeEventListener("touchend", this.onTouchEnd);

        if (this.canvas.parentElement) {
            this.canvas.parentElement.removeChild(this.canvas);
        }
    }
}

export class WallpaperService {
    private instance: ParticleNetwork | null = null;
    private resizeHandler: (() => void) | null = null;

    start(root: HTMLElement, options?: WallpaperServiceOptions) {
        this.stop();
        this.instance = new ParticleNetwork(root, options);

        this.resizeHandler = () => {
            if (!this.instance) return;
            this.instance.ctx.clearRect(
                0,
                0,
                this.instance.canvas.width,
                this.instance.canvas.height
            );
            this.instance.sizeCanvas();
            this.instance.createParticles(false);
        };

        window.addEventListener("resize", this.resizeHandler);
    }

    stop() {
        if (this.resizeHandler) {
            window.removeEventListener("resize", this.resizeHandler);
            this.resizeHandler = null;
        }

        if (this.instance) {
            this.instance.destroy();
            this.instance = null;
        }
    }
}
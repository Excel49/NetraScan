// 3D Point Cloud Visualizer using Three.js

class PointCloudVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.pointCloud = null;
        this.points = [];

        this.init();
    }

    init() {
        if (!this.container) {
            console.error('Container not found');
            return;
        }

        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0f172a);

        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.resetView(); // Set initial view

        // Fetch custom config
        this.defaultCameraPosition = new THREE.Vector3(5, 5, 5);
        this.fetchConfig().then(() => {
            // Apply config immediately on load
            this.resetView();
        });

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        // Add orbit controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        // Add lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        this.scene.add(directionalLight);

        // Add coordinate axes with custom colors and labels
        this.addColoredAxes(5); // Length = 5 units

        // DEBUG: 4 titik referensi untuk cek sumbu
        const debugPoints = [
            { pos: [0, 0, 0], color: 0xffffff, label: 'Origin' },      // Putih = pusat
            { pos: [2, 0, 0], color: 0x3b82f6, label: 'X+2' },         // Biru = X positif
            { pos: [0, 2, 0], color: 0x22c55e, label: 'Y+2' },         // Hijau = Y positif (atas)
            { pos: [0, 0, 2], color: 0xef4444, label: 'Z+2' },         // Merah = Z positif
        ];
        for (const dp of debugPoints) {
            const geo = new THREE.SphereGeometry(0.15, 16, 16);
            const mat = new THREE.MeshBasicMaterial({ color: dp.color });
            const sphere = new THREE.Mesh(geo, mat);
            sphere.position.set(dp.pos[0], dp.pos[1], dp.pos[2]);
            this.scene.add(sphere);

            const sprite = this.makeTextSprite(dp.label, dp.color);
            sprite.position.set(dp.pos[0], dp.pos[1] + 0.35, dp.pos[2]);
            this.scene.add(sprite);
        }

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Start animation loop
        this.animate();
    }

    addColoredAxes(length) {
        const axisConfig = [
            { dir: new THREE.Vector3(1, 0, 0), color: 0x3b82f6, label: 'X' }, // Blue
            { dir: new THREE.Vector3(0, 1, 0), color: 0x22c55e, label: 'Y' }, // Green
            { dir: new THREE.Vector3(0, 0, 1), color: 0xef4444, label: 'Z' }, // Red
        ];

        const origin = new THREE.Vector3(0, 0, 0);

        for (const axis of axisConfig) {
            const end = axis.dir.clone().multiplyScalar(length);

            // Line
            const lineGeo = new THREE.BufferGeometry().setFromPoints([origin, end]);
            const lineMat = new THREE.LineBasicMaterial({ color: axis.color, linewidth: 2 });
            this.scene.add(new THREE.Line(lineGeo, lineMat));

            // Arrow cone at tip
            const coneGeo = new THREE.ConeGeometry(0.12, 0.4, 8);
            const coneMat = new THREE.MeshBasicMaterial({ color: axis.color });
            const cone = new THREE.Mesh(coneGeo, coneMat);
            cone.position.copy(end);
            // Orient cone along axis direction
            cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis.dir);
            this.scene.add(cone);

            // Text label
            const sprite = this.makeTextSprite(axis.label, axis.color);
            sprite.position.copy(end.clone().add(axis.dir.clone().multiplyScalar(0.6)));
            this.scene.add(sprite);
        }

        // Grid on XZ plane (floor)
        const gridHelper = new THREE.GridHelper(20, 20, 0x888888, 0x222222);
        gridHelper.position.y = -0.01;
        this.scene.add(gridHelper);
    }

    makeTextSprite(text, hexColor) {
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');

        // Convert hex color to CSS string
        const colorStr = '#' + new THREE.Color(hexColor).getHexString();

        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(0, 0, 128, 64);
        ctx.font = 'Bold 48px Inter, Arial, sans-serif';
        ctx.fillStyle = colorStr;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 64, 32);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
        const sprite = new THREE.Sprite(material);
        sprite.scale.set(1, 0.5, 1);
        return sprite;
    }

    updatePointCloud(points) {
        // Clear existing point cloud
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
        }

        if (!points || points.length === 0) {
            console.log('No points to visualize');
            return;
        }

        this.points = points;

        // Prepare geometry
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(points.length * 3);
        const colors = new Float32Array(points.length * 3);

        // Find bounds for normalization
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;

        for (let i = 0; i < points.length; i++) {
            const point = points[i];
            const x = point[0] || 0;
            const y = point[1] || 0;
            const z = point[2] || 0;

            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
            minZ = Math.min(minZ, z);
            maxZ = Math.max(maxZ, z);
        }

        // Calculate center and scale
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const centerZ = (minZ + maxZ) / 2;

        const rangeX = maxX - minX;
        const rangeY = maxY - minY;
        const rangeZ = maxZ - minZ;
        const maxRange = Math.max(rangeX, rangeY, rangeZ);
        const scale = maxRange > 0 ? 10 / maxRange : 1;

        // Fill buffers
        for (let i = 0; i < points.length; i++) {
            const point = points[i];
            const idx = i * 3;

            // Normalize positions
            positions[idx] = ((point[0] || 0) - centerX) * scale;
            positions[idx + 1] = ((point[1] || 0) - centerY) * scale;
            positions[idx + 2] = ((point[2] || 0) - centerZ) * scale;

            // Set colors (use provided colors or generate based on height)
            if (point.length >= 6) {
                // Use provided RGB colors
                colors[idx] = (point[3] || 255) / 255;
                colors[idx + 1] = (point[4] || 0) / 255;
                colors[idx + 2] = (point[5] || 0) / 255;
            } else {
                // Generate color based on height (blue to red gradient)
                const normalizedY = (positions[idx + 1] + 5) / 10; // Normalize to 0-1
                colors[idx] = normalizedY; // Red increases with height
                colors[idx + 1] = 0;
                colors[idx + 2] = 1 - normalizedY; // Blue decreases with height
            }
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        // Create material
        const material = new THREE.PointsMaterial({
            size: 0.05,
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        });

        // Create point cloud
        this.pointCloud = new THREE.Points(geometry, material);
        this.scene.add(this.pointCloud);

        // Update info display
        this.updateInfoDisplay(points.length);

        console.log(`Point cloud updated: ${points.length} points`);
    }

    updateInfoDisplay(pointCount) {
        const infoElement = document.getElementById('pointCloudInfo');
        if (infoElement) {
            infoElement.innerHTML = `
                <div class="info-item">
                    <i class="fas fa-cube"></i>
                    <span>Points: ${pointCount.toLocaleString()}</span>
                </div>
                <div class="info-item">
                    <i class="fas fa-arrows-alt-h"></i>
                    <span>Click and drag to rotate</span>
                </div>
                <div class="info-item">
                    <i class="fas fa-mouse-pointer"></i>
                    <span>Scroll to zoom</span>
                </div>
            `;
        }
    }

    onWindowResize() {
        if (!this.camera || !this.renderer) return;

        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.controls) {
            this.controls.update();
        }

        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }



    changePointSize(size) {
        if (this.pointCloud) {
            this.pointCloud.material.size = size;
            this.pointCloud.material.needsUpdate = true;
        }
    }

    changeColorMode(mode) {
        if (!this.pointCloud || !this.points.length) return;

        const geometry = this.pointCloud.geometry;
        const colors = geometry.attributes.color.array;
        const positions = geometry.attributes.position.array;

        switch (mode) {
            case 'height':
                // Color by height (Y coordinate)
                for (let i = 0; i < this.points.length; i++) {
                    const idx = i * 3;
                    const y = positions[idx + 1];
                    const normalizedY = (y + 5) / 10; // Normalize to 0-1

                    colors[idx] = normalizedY; // Red
                    colors[idx + 1] = 0;
                    colors[idx + 2] = 1 - normalizedY; // Blue
                }
                break;

            case 'distance':
                // Color by distance from center
                for (let i = 0; i < this.points.length; i++) {
                    const idx = i * 3;
                    const x = positions[idx];
                    const y = positions[idx + 1];
                    const z = positions[idx + 2];

                    const distance = Math.sqrt(x * x + y * y + z * z);
                    const normalizedDist = Math.min(distance / 10, 1);

                    colors[idx] = 1 - normalizedDist; // Red decreases with distance
                    colors[idx + 1] = normalizedDist; // Green increases with distance
                    colors[idx + 2] = 0;
                }
                break;

            case 'single':
                // Single color
                for (let i = 0; i < this.points.length; i++) {
                    const idx = i * 3;
                    colors[idx] = 1;     // Red
                    colors[idx + 1] = 0; // Green
                    colors[idx + 2] = 0; // Blue
                }
                break;
        }

        geometry.attributes.color.needsUpdate = true;
    }

    exportScreenshot() {
        if (!this.renderer) return;

        // Create temporary link
        const link = document.createElement('a');
        link.style.display = 'none';
        document.body.appendChild(link);

        // Take screenshot
        this.renderer.render(this.scene, this.camera);
        const dataURL = this.renderer.domElement.toDataURL('image/png');

        // Trigger download
        link.href = dataURL;
        link.download = `pointcloud_${Date.now()}.png`;
        link.click();

        // Clean up
        document.body.removeChild(link);
    }

    getStats() {
        if (!this.pointCloud) {
            return { points: 0 };
        }

        const positions = this.pointCloud.geometry.attributes.position.array;
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;

        for (let i = 0; i < positions.length; i += 3) {
            const x = positions[i];
            const y = positions[i + 1];
            const z = positions[i + 2];

            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
            minZ = Math.min(minZ, z);
            maxZ = Math.max(maxZ, z);
        }

        return {
            points: positions.length / 3,
            bounds: {
                x: { min: minX, max: maxX, range: maxX - minX },
                y: { min: minY, max: maxY, range: maxY - minY },
                z: { min: minZ, max: maxZ, range: maxZ - minZ }
            },
            center: {
                x: (minX + maxX) / 2,
                y: (minY + maxY) / 2,
                z: (minZ + maxZ) / 2
            }
        };
    }

    destroy() {
        // Clean up Three.js resources
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
        }

        if (this.renderer) {
            this.renderer.dispose();
            this.container.removeChild(this.renderer.domElement);
        }

        if (this.controls) {
            this.controls.dispose();
        }

        window.removeEventListener('resize', () => this.onWindowResize());
    }
    async fetchConfig() {
        try {
            const response = await fetch('/api/config');
            if (response.ok) {
                const config = await response.json();
                if (config.camera_position) {
                    this.defaultCameraPosition = new THREE.Vector3(...config.camera_position);
                    // Update current view if not yet interacted? 
                    // Better just set it for resetView usage
                }
            }
        } catch (e) {
            console.error("Failed to load config", e);
        }
    }

    resetView() {
        if (this.camera) {
            const pos = this.defaultCameraPosition || new THREE.Vector3(5, 5, 5);
            this.camera.position.copy(pos);
            this.camera.lookAt(0, 0, 0);
            if (this.controls) {
                this.controls.target.set(0, 0, 0);
                this.controls.update();
            }
        }
    }

    setTopView() {
        if (this.camera) {
            this.camera.position.set(0, 15, 0.01);
            this.camera.lookAt(0, 0, 0);
            if (this.controls) {
                this.controls.target.set(0, 0, 0);
                this.controls.update();
            }
        }
    }

    setFrontView() {
        if (this.camera) {
            this.camera.position.set(0, 3, 10);
            this.camera.lookAt(0, 0, 0);
            if (this.controls) {
                this.controls.target.set(0, 0, 0);
                this.controls.update();
            }
        }
    }
}

// Export for use in main.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PointCloudVisualizer;
}

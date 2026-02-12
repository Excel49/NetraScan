document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    // Elements
    const calibrationStatus = document.getElementById('calib-status');
    const calibrationProgress = document.getElementById('calib-progress');
    const startCalibBtn = document.getElementById('btn-start-calib');

    const scanStatus = document.getElementById('scan-status');
    const scanProgress = document.getElementById('scan-progress');
    const startScanBtn = document.getElementById('btn-start-scan');
    const stopScanBtn = document.getElementById('btn-stop-scan');
    const btnCalibCamera = document.getElementById('btn-start-calib');
    const btnCalibLaser = document.getElementById('btn-calib-laser');

    // --- BUTTON HANDLERS ---

    // Camera Calibration (Stage 1)
    if (btnCalibCamera) {
        btnCalibCamera.addEventListener('click', async () => {
            try {
                const response = await fetch('/api/calibrate/camera', { method: 'POST' });
                const data = await response.json();
                if (!data.success) alert(data.error);
            } catch (error) {
                console.error('Error starting camera calibration:', error);
            }
        });
    }

    // Laser Calibration (Stage 2)
    if (btnCalibLaser) {
        btnCalibLaser.addEventListener('click', async () => {
            try {
                const response = await fetch('/api/calibrate/laser', { method: 'POST' });
                const data = await response.json();
                if (!data.success) alert(data.error);
            } catch (error) {
                console.error('Error starting laser calibration:', error);
            }
        });
    }

    // Toggle Axis View
    const btnToggleAxis = document.getElementById('btn-toggle-axis');
    const calibVideoFeed = document.getElementById('calibVideoFeed');
    if (btnToggleAxis && calibVideoFeed) {
        btnToggleAxis.addEventListener('click', () => {
            if (calibVideoFeed.src.includes('mode=axis')) {
                calibVideoFeed.src = '/video_feed';
                btnToggleAxis.textContent = 'Show Axis';
            } else {
                calibVideoFeed.src = '/video_feed?mode=axis';
                btnToggleAxis.textContent = 'Hide Axis';
            }
        });
    }

    // Legacy Calibration Logic removed - handled by specific buttons above

    socket.on('calibration_status', (data) => {
        if (data.message) calibrationStatus.textContent = data.message;
        if (data.progress !== undefined) {
            calibrationProgress.style.width = data.progress + '%';
        }

        if (data.status === 'completed' || data.status === 'error') {
            startCalibBtn.disabled = false;
            if (data.status === 'completed') {
                calibrationStatus.innerHTML = "✅ " + data.message;
            }
        }
    });

    // --- Threshold Slider ---
    const laserThreshold = document.getElementById('laserThreshold');
    const thresholdValue = document.getElementById('thresholdValue');

    if (laserThreshold) {
        laserThreshold.addEventListener('input', function () {
            if (thresholdValue) thresholdValue.textContent = this.value;
        });

        laserThreshold.addEventListener('change', async function () {
            try {
                await fetch('/api/settings/threshold', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ threshold: this.value })
                });
            } catch (e) {
                console.error("Error setting threshold:", e);
            }
        });
    }

    // --- Exposure Slider ---
    const cameraExposure = document.getElementById('cameraExposure');
    const exposureValue = document.getElementById('exposureValue');

    if (cameraExposure) {
        cameraExposure.addEventListener('input', function () {
            if (exposureValue) exposureValue.textContent = this.value;
        });

        cameraExposure.addEventListener('change', async function () {
            try {
                await fetch('/api/settings/exposure', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ exposure: this.value })
                });
            } catch (e) {
                console.error("Error setting exposure:", e);
            }
        });
    }

    // --- Visualizer Initialization ---
    let pointCloudVisualizer = null;
    if (document.getElementById('pointCloudCanvas')) {
        pointCloudVisualizer = new PointCloudVisualizer('pointCloudCanvas');
    }

    // --- View Buttons ---
    const btnViewReset = document.getElementById('btn-view-reset');
    const btnViewTop = document.getElementById('btn-view-top');
    const btnViewFront = document.getElementById('btn-view-front');

    if (btnViewReset && pointCloudVisualizer) {
        btnViewReset.addEventListener('click', () => pointCloudVisualizer.resetView());
    }
    if (btnViewTop && pointCloudVisualizer) {
        btnViewTop.addEventListener('click', () => pointCloudVisualizer.setTopView());
    }
    if (btnViewFront && pointCloudVisualizer) {
        btnViewFront.addEventListener('click', () => pointCloudVisualizer.setFrontView());
    }

    // --- Laser Overlay Logic ---
    const toggleLaserOverlay = document.getElementById('toggleLaserOverlay');
    const videoStream = document.getElementById('videoStream');
    const laserFeed = document.getElementById('laserFeed');

    if (toggleLaserOverlay) {
        toggleLaserOverlay.addEventListener('change', function () {
            if (this.checked) {
                videoStream.style.display = 'none';
                laserFeed.style.display = 'block';
                laserFeed.src = '/laser_overlay?t=' + new Date().getTime();
            } else {
                laserFeed.style.display = 'none';
                videoStream.style.display = 'block';
                videoStream.src = '/video_feed?t=' + new Date().getTime();
            }
        });
    }

    // --- Scanning Logic ---
    let allPoints = []; // Store cumulative points

    const downloadScanBtn = document.getElementById('btn-download-scan');

    startScanBtn.addEventListener('click', async () => {
        try {
            startScanBtn.disabled = true;
            stopScanBtn.disabled = false;
            // Hide download button during scan
            if (downloadScanBtn) downloadScanBtn.style.display = 'none';

            scanStatus.textContent = "Starting scan...";

            // Reset points
            allPoints = [];

            // Clear previous points
            if (pointCloudVisualizer) {
                pointCloudVisualizer.updatePointCloud([]);
                document.getElementById('pointCloudInfo').textContent = "Points: 0 | Drag to rotate";
            }

            const res = await fetch('/api/scan/start', { method: 'POST' });
            const data = await res.json();

            if (!data.success) {
                scanStatus.textContent = "Error: " + data.error;
                startScanBtn.disabled = false;
                stopScanBtn.disabled = true;
            }
        } catch (e) {
            console.error(e);
            startScanBtn.disabled = false;
        }
    });

    stopScanBtn.addEventListener('click', async () => {
        await fetch('/api/scan/stop', { method: 'POST' });
        scanStatus.textContent = "Stopping...";
    });

    socket.on('scan_progress', (data) => {
        scanProgress.style.width = data.progress + '%';
        scanStatus.textContent = `Scanning... ${Math.round(data.progress)}%`;

        // Use accumulated count
        if (pointCloudVisualizer) {
            const count = allPoints.length;
            document.getElementById('pointCloudInfo').textContent = `Points: ${count.toLocaleString()} | Drag to rotate`;
        }
    });

    socket.on('pointcloud_data', (data) => {
        if (pointCloudVisualizer && data.points) {
            // ACCUMULATE POINTS
            // data.points is an array of [x,y,z,r,g,b]
            for (let p of data.points) {
                allPoints.push(p);
            }

            // Render full cloud
            pointCloudVisualizer.updatePointCloud(allPoints);
            document.getElementById('pointCloudInfo').textContent = `Points: ${allPoints.length.toLocaleString()} | Drag to rotate`;
        }
    });

    socket.on('scan_complete', (data) => {
        startScanBtn.disabled = false;
        stopScanBtn.disabled = true;
        scanStatus.innerHTML = "✅ Scan " + data.status;
        scanProgress.style.width = '100%';

        if (data.status === 'completed' && data.download_url && downloadScanBtn) {
            downloadScanBtn.href = data.download_url;
            downloadScanBtn.style.display = 'inline-block';
            downloadScanBtn.innerHTML = "Download Result (.PLY)";
        }

        // Final sync if needed
        if (data.points_count && data.points_count !== allPoints.length) {
            console.log(`Note: Backend reported ${data.points_count} points, Frontend has ${allPoints.length}`);
        }
    });
});

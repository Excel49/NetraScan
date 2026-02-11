# NetraScan-Lite

3D Scanner Control Software with Turntable Integration and Dual-Board Extrinsic Calibration.

## Setup Installation (Device Baru)

### 1. Clone Repository (Dari GitHub)
```bash
git clone https://github.com/Excel49/NetraScan.git
cd NetraScan/NetraScan-Lite
```

### 2. Setup Python Environment
Pastikan sudah install Python 3.10 ke atas.

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```bash
python app.py
```
Akses web control di: `http://localhost:5000`

## Kalibrasi
Lihat file `calibrate_dual_board.py` untuk kalibrasi ekstrinsik.

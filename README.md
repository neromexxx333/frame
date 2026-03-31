# Portal 2D Reliability Analysis Application

## Deskripsi
Aplikasi untuk menganalisis portal 2D dengan metode matriks kekakuan langsung 
dan integrasi Monte Carlo untuk analisis keandalan probabilistik.

## Fitur
- Analisis struktur portal 2D menggunakan direct stiffness method
- Pembebanan: beban merata (stokastik) + beban nodal (deterministik)
- Simulasi Monte Carlo dengan random variables:
  - Mutu beton: distribusi log-normal
  - Mutu baja: distribusi normal
  - Beban mati: distribusi normal
  - Beban hidup: distribusi log-normal
- Perhitungan Pf (probability of failure) dan Beta (reliability index)
- Analisis sensitivitas variabel random
- Dashboard Streamlit untuk menampilkan input/output simulasi terakhir

## Struktur Folder
```
d:/Reli_DSM/
├── main.py                    # Aplikasi utama
├── requirements.txt           # Python dependencies
├── input_template.xlsx        # Template input Excel
├── create_sample_excel.py     # Script membuat template
├── modules/
│   ├── excel_reader.py        # Baca input Excel
│   ├── stiffness_matrix.py    # Metode matriks kekakuan
│   ├── analysis.py            # Analisis struktural
│   ├── monte_carlo.py         # Simulasi Monte Carlo
│   ├── reliability.py         # Analisis keandalan
│   └── plotting.py            # Visualisasi
└── output/
    ├── reliability_report.txt
    └── analysis_results.json
```

## Cara Menggunakan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Buat Input Data Excel
```bash
python create_sample_excel.py
```
Ini akan membuat file `input_template.xlsx` dengan struktur dan data contoh.

### 3. Edit Data Input
Buka `input_template.xlsx` dan sesuaikan dengan data struktur Anda.

### 4. Jalankan Analisis
```bash
python main.py input_template.xlsx
```

### 5. Jalankan Dashboard Streamlit
```bash
streamlit run streamlit_app.py
```

### 6. Lihat Hasil
Output akan disimpan di folder `output/`:
- `reliability_report.txt` - Laporan keandalan
- `analysis_results.json` - Hasil numerik
- `failure_probability.png` - Grafik distribusi
- `sensitivity_analysis.png` - Grafik sensitivitas
- `last_simulation_deformation.png` - Plot deformasi simulasi terakhir
- `last_simulation_axial.png` - Diagram gaya aksial simulasi terakhir
- `last_simulation_shear.png` - Diagram gaya geser simulasi terakhir
- `last_simulation_moment.png` - Diagram momen simulasi terakhir

## Format Excel Input

### Sheet: Geometri
- Element_ID: Nomor elemen
- Node_Start, Node_End: Node awal dan akhir
- Area: Luas penampang (mm²)
- Inertia: Momen inersia (mm⁴)
- E_Mean: Modulus elastisitas (MPa)

### Sheet: Nodes
- Node_ID: Nomor node
- X, Y: Koordinat (mm)

### Sheet: Boundary_Condition
- Node_ID: Nomor node
- Restrain_X, Y, Rz: 1=tertahan, 0=bebas

### Sheet: Mutu_Beton
- Mean: Rata-rata (MPa)
- StdDev: Deviasi standart
- Distribution: 'lognormal'

### Sheet: Mutu_Baja
- Mendukung format lama berbasis `Tipe` (`Tarik`, `Tekan`, opsional `Geser`)
- Mendukung format baru per elemen: `Element_ID`, `Mean/StdDev/Distribution` untuk `tarik`, `tekan`, dan `geser`
- Kolom deterministic per elemen juga didukung: `Deterministic_tarik`, `Deterministic_tekan`, `Deterministic_geser`

### Sheet: Beban_Mati dan Beban_Hidup
- Element_ID: Elemen yang dibebani
- Mean: Beban rata-rata (kN/m)
- StdDev: Deviasi
- Distribution: 'normal' (mati) atau 'lognormal' (hidup)

### Sheet: Beban_Nodal
- Node_ID: Nomor node
- Fx, Fy: Gaya horizontal dan vertikal (kN)
- Mz: Momen (kN.m)

## Output Report

Laporan keandalan mencakup:
- Probability of Failure (Pf)
- Reliability Index (Beta)
- Klasifikasi keamanan
- Interpretasi hasil
- Ranking sensitivitas variabel

## Contoh Output
```
============================================================
RELIABILITY ASSESSMENT REPORT
============================================================

Simulation Statistics:
  - Number of simulations: 10000
  - Number of failures: 25

Results:
  - Probability of Failure (Pf): 0.002500 (0.25%)
  - Reliability Index (Beta): 2.8070
  - Safety Classification: Safe
  - Target Beta (ULS): 3.0
  - Safety Status: ✗ UNSAFE (below target)

Interpretation:
  - Expected 1 failure in every 400 structures
  - Structure is Safe relative to target
```

## Notes
- Porto dan gaya dalam dihitung menggunakan metode matriks kekakuan
- Beban merata dikonversi ke gaya nodal ekuivalen di tengah elemen
- Total load = dead load + live load
- Semua beban positif mengarah ke bawah (Y negatif)

## Contact
Untuk pertanyaan atau bugs, silakan laporkan di repository.

// Ödev 3 - Haftaya görevi (2. kısım): Beş sorunun CUDA (GPU) sürümleri, elle yazılmış GPU çekirdekleriyle.
//
//   1) Doğrusal kontrast germe       (min/max indirgeme çekirdeği + ölçekleme çekirdeği)
//   2) Histogram eşitleme            (paylaşılan bellekte blok histogramı + atomik toplama; CDF CPU'da)
//   3) CLAHE (8x8, clip 2.0)         (her kare için bir blok: histogram, kırpma, tablo; sonra piksel başına interpolasyon)
//   4) 3x3 ortalama filtre           (piksel başına bir GPU thread'i)
//   5) 5x5 medyan filtre             (piksel başına bir GPU thread'i, ekleme sıralaması)
//
// Bu dosya, aynı klasördeki odev3_thread.cpp'yi içine alır; böylece aynı çalıştırmada CPU sürümleri
// (1 thread ve çok thread) da ölçülür ve GPU sonucu CPU sonucuyla birebir karşılaştırılır.
//
// Süreler iki biçimde verilir:
//   GPU çekirdek : görüntü zaten GPU belleğindeyken sadece hesaplama (küçük 256 elemanlık kopyalar dahil)
//   GPU toplam   : görüntüyü CPU'dan GPU'ya yüklemek ve sonucu geri indirmek dahil
//
// Kullanım: ./cuda_test <veri_dizini> [cikti_dizini=cikti_cuda] [cpu_tekrar=5] [gpu_tekrar=50]
//
// Derleme (Google Colab, GPU çalışma zamanı; OpenCV: apt-get install libopencv-dev):
//   nvcc -O2 -std=c++17 --fmad=false -arch=native odev3_cuda.cu -o cuda_test \
//        $(pkg-config --cflags --libs opencv4) -Xcompiler -pthread
// --fmad=false: GPU'nun çarp-topla kaynaştırmasını kapatır, böylece ondalıklı hesaplar CPU ile aynı
// sırada ve aynı yuvarlamayla yapılır (sonuçların birebir aynı çıkması için).

#define THREAD_TEST_NO_MAIN
#include "odev3_thread.cpp"

#include <cuda_runtime.h>

#include <cstdlib>

#define BLK 256

#define CUDA_CHECK(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t err__ = (call);                                                                       \
        if (err__ != cudaSuccess) {                                                                       \
            std::cerr << "CUDA hatasi: " << cudaGetErrorString(err__) << " (satir " << __LINE__ << ")\n"; \
            std::exit(1);                                                                                 \
        }                                                                                                 \
    } while (0)

// ---------------------------------------------------------------------------------------------
// GPU yardımcı fonksiyonları
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ int d_reflect101(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
    }
    return i;
}
__device__ __forceinline__ int d_clamp(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }
__device__ __forceinline__ uchar d_sat(int v) { return static_cast<uchar>(v < 0 ? 0 : (v > 255 ? 255 : v)); }

// ---------------------------------------------------------------------------------------------
// Görev 1: kontrast germe
// ---------------------------------------------------------------------------------------------
__global__ void minMaxKernel(const uchar* in, int n, int* gmin, int* gmax) {
    __shared__ int smin[BLK];
    __shared__ int smax[BLK];
    int lo = 255, hi = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int v = in[i];
        lo = min(lo, v);
        hi = max(hi, v);
    }
    smin[threadIdx.x] = lo;
    smax[threadIdx.x] = hi;
    __syncthreads();
    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smin[threadIdx.x] = min(smin[threadIdx.x], smin[threadIdx.x + s]);
            smax[threadIdx.x] = max(smax[threadIdx.x], smax[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMin(gmin, smin[0]);
        atomicMax(gmax, smax[0]);
    }
}

__global__ void scaleKernel(const uchar* in, uchar* out, int n, int lo, double scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<uchar>(__dadd_rn(__dmul_rn(static_cast<double>(static_cast<int>(in[i]) - lo), scale), 0.5));
}

// ---------------------------------------------------------------------------------------------
// Görev 2: histogram eşitleme
// ---------------------------------------------------------------------------------------------
__global__ void histKernel(const uchar* in, int n, unsigned int* ghist) {
    __shared__ unsigned int sh[256];
    sh[threadIdx.x] = 0;  // blockDim.x == 256
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) atomicAdd(&sh[in[i]], 1u);
    __syncthreads();
    atomicAdd(&ghist[threadIdx.x], sh[threadIdx.x]);
}

__global__ void lutApplyKernel(const uchar* in, uchar* out, int n, const uchar* lut) {
    __shared__ uchar sl[256];
    sl[threadIdx.x] = lut[threadIdx.x];  // blockDim.x == 256
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sl[in[i]];
}

// ---------------------------------------------------------------------------------------------
// Görev 3: CLAHE
// ---------------------------------------------------------------------------------------------
// Her blok bir kareyi işler: histogram (paylaşılan bellekte), kırpma ve dağıtma, dönüşüm tablosu.
__global__ void claheLutKernel(const uchar* src, int W, int H, int gridX, int tileW, int tileH, int clipLimit,
                               float lutScale, uchar* luts) {
    __shared__ int hist[256];
    const int t = blockIdx.x;
    const int ty = t / gridX, tx = t % gridX;
    hist[threadIdx.x] = 0;  // blockDim.x == 256
    __syncthreads();

    const int total = tileW * tileH;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int yy = ty * tileH + idx / tileW;
        const int xx = tx * tileW + idx % tileW;
        atomicAdd(&hist[src[d_reflect101(yy, H) * W + d_reflect101(xx, W)]], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {  // 256 elemanlık küçük seri iş: kırpma, dağıtma, tablo
        int clipped = 0;
        for (int i = 0; i < 256; ++i) {
            if (hist[i] > clipLimit) {
                clipped += hist[i] - clipLimit;
                hist[i] = clipLimit;
            }
        }
        const int redistBatch = clipped / 256;
        int residual = clipped - redistBatch * 256;
        for (int i = 0; i < 256; ++i) hist[i] += redistBatch;
        if (residual != 0) {
            const int residualStep = max(256 / residual, 1);
            for (int i = 0; i < 256 && residual > 0; i += residualStep, --residual) ++hist[i];
        }
        uchar* lut = luts + static_cast<size_t>(t) * 256;
        int sum = 0;
        for (int i = 0; i < 256; ++i) {
            sum += hist[i];
            lut[i] = d_sat(__float2int_rn(static_cast<float>(sum) * lutScale));
        }
    }
}

// Piksel başına bir thread: en yakın 4 karenin tablosundan bilineer interpolasyon
__global__ void claheInterpKernel(const uchar* src, uchar* dst, int W, int H, int gridX, int gridY, float invTw,
                                  float invTh, const uchar* luts) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const float txf = x * invTw - 0.5f;
    int tx1 = static_cast<int>(floorf(txf));
    const float xa = txf - tx1;
    int tx2 = tx1 + 1;
    tx1 = max(tx1, 0);
    tx2 = min(tx2, gridX - 1);

    const float tyf = y * invTh - 0.5f;
    int ty1 = static_cast<int>(floorf(tyf));
    const float ya = tyf - ty1;
    int ty2 = ty1 + 1;
    ty1 = max(ty1, 0);
    ty2 = min(ty2, gridY - 1);

    const int v = src[y * W + x];
    const float l11 = luts[(ty1 * gridX + tx1) * 256 + v];
    const float l12 = luts[(ty1 * gridX + tx2) * 256 + v];
    const float l21 = luts[(ty2 * gridX + tx1) * 256 + v];
    const float l22 = luts[(ty2 * gridX + tx2) * 256 + v];
    const float top = l11 * (1.0f - xa) + l12 * xa;
    const float bottom = l21 * (1.0f - xa) + l22 * xa;
    dst[y * W + x] = d_sat(__float2int_rn(top * (1.0f - ya) + bottom * ya));
}

// ---------------------------------------------------------------------------------------------
// Görev 4: 3x3 ortalama filtre
// ---------------------------------------------------------------------------------------------
__global__ void meanKernel(const uchar* in, uchar* out, int W, int H) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const uchar* r0 = in + static_cast<size_t>(d_reflect101(y - 1, H)) * W;
    const uchar* r1 = in + static_cast<size_t>(y) * W;
    const uchar* r2 = in + static_cast<size_t>(d_reflect101(y + 1, H)) * W;
    const int c0 = d_reflect101(x - 1, W), c1 = x, c2 = d_reflect101(x + 1, W);
    const float w = 1.0f / 9.0f;
    const float sum = w * r0[c0] + w * r0[c1] + w * r0[c2] + w * r1[c0] + w * r1[c1] + w * r1[c2] + w * r2[c0] +
                      w * r2[c1] + w * r2[c2];
    out[static_cast<size_t>(y) * W + x] = d_sat(__float2int_rn(sum));
}

// ---------------------------------------------------------------------------------------------
// Görev 5: 5x5 medyan filtre
// ---------------------------------------------------------------------------------------------
__global__ void medianKernel(const uchar* in, uchar* out, int W, int H) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    uchar v[25];
    int n = 0;
    for (int ky = 0; ky < 5; ++ky) {
        const uchar* row = in + static_cast<size_t>(d_clamp(y + ky - 2, H)) * W;
        for (int kx = 0; kx < 5; ++kx) v[n++] = row[d_clamp(x + kx - 2, W)];
    }
    for (int i = 1; i < 25; ++i) {
        const uchar key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            --j;
        }
        v[j + 1] = key;
    }
    out[static_cast<size_t>(y) * W + x] = v[12];
}

// ---------------------------------------------------------------------------------------------
// Ana bilgisayar (host) tarafı: GPU belleği ve görev sarmalayıcıları
// ---------------------------------------------------------------------------------------------
struct GpuCtx {
    int W = 0, H = 0;
    size_t n = 0;
    uchar *d_in = nullptr, *d_out = nullptr, *d_lut = nullptr, *d_luts = nullptr;
    int* d_minmax = nullptr;
    unsigned int* d_hist = nullptr;

    void init(int w, int h) {
        W = w;
        H = h;
        n = static_cast<size_t>(w) * h;
        CUDA_CHECK(cudaMalloc(&d_in, n));
        CUDA_CHECK(cudaMalloc(&d_out, n));
        CUDA_CHECK(cudaMalloc(&d_lut, 256));
        CUDA_CHECK(cudaMalloc(&d_luts, 64 * 256));
        CUDA_CHECK(cudaMalloc(&d_minmax, 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_hist, 256 * sizeof(unsigned int)));
    }
    void release() {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_lut);
        cudaFree(d_luts);
        cudaFree(d_minmax);
        cudaFree(d_hist);
    }
};

// xfer = true: görüntüyü GPU'ya yükle ve sonucu geri indir (toplam süre). false: sadece hesaplama.
static void uploadIfNeeded(const cv::Mat& src, GpuCtx& c, bool xfer) {
    if (xfer) CUDA_CHECK(cudaMemcpy(c.d_in, src.data, c.n, cudaMemcpyHostToDevice));
}
static cv::Mat downloadIfNeeded(const cv::Mat& src, GpuCtx& c, bool xfer) {
    if (!xfer) {
        CUDA_CHECK(cudaDeviceSynchronize());
        return cv::Mat();
    }
    cv::Mat out(src.size(), CV_8UC1);
    CUDA_CHECK(cudaMemcpy(out.data, c.d_out, c.n, cudaMemcpyDeviceToHost));
    return out;
}

static cv::Mat gpuContrast(const cv::Mat& src, GpuCtx& c, bool xfer) {
    uploadIfNeeded(src, c, xfer);
    const int n = static_cast<int>(c.n);
    const int init[2] = {255, 0};
    CUDA_CHECK(cudaMemcpy(c.d_minmax, init, sizeof(init), cudaMemcpyHostToDevice));
    minMaxKernel<<<std::min(1024, (n + BLK - 1) / BLK), BLK>>>(c.d_in, n, c.d_minmax, c.d_minmax + 1);
    CUDA_CHECK(cudaGetLastError());
    int res[2];
    CUDA_CHECK(cudaMemcpy(res, c.d_minmax, sizeof(res), cudaMemcpyDeviceToHost));
    const int lo = res[0], hi = res[1];
    if (hi == lo) {
        CUDA_CHECK(cudaMemcpy(c.d_out, c.d_in, c.n, cudaMemcpyDeviceToDevice));
    } else {
        scaleKernel<<<(n + BLK - 1) / BLK, BLK>>>(c.d_in, c.d_out, n, lo, 255.0 / (hi - lo));
        CUDA_CHECK(cudaGetLastError());
    }
    return downloadIfNeeded(src, c, xfer);
}

static cv::Mat gpuHistEqualize(const cv::Mat& src, GpuCtx& c, bool xfer) {
    uploadIfNeeded(src, c, xfer);
    const int n = static_cast<int>(c.n);
    CUDA_CHECK(cudaMemset(c.d_hist, 0, 256 * sizeof(unsigned int)));
    histKernel<<<std::min(1024, (n + BLK - 1) / BLK), BLK>>>(c.d_in, n, c.d_hist);
    CUDA_CHECK(cudaGetLastError());
    unsigned int h[256];
    CUDA_CHECK(cudaMemcpy(h, c.d_hist, sizeof(h), cudaMemcpyDeviceToHost));

    // CDF ve dönüşüm tablosu (256 elemanlık, CPU'da, CPU sürümüyle aynı formül)
    std::array<unsigned long long, 256> cdf;
    unsigned long long run = 0;
    for (int b = 0; b < 256; ++b) {
        run += h[b];
        cdf[b] = run;
    }
    const unsigned long long N = c.n;
    unsigned long long cdfMin = 0;
    for (int b = 0; b < 256; ++b) {
        if (h[b] > 0) {
            cdfMin = cdf[b];
            break;
        }
    }
    if (cdfMin == N) {
        CUDA_CHECK(cudaMemcpy(c.d_out, c.d_in, c.n, cudaMemcpyDeviceToDevice));
    } else {
        uchar lut[256];
        for (int v = 0; v < 256; ++v)
            lut[v] = (cdf[v] < cdfMin) ? 0
                                       : static_cast<uchar>(static_cast<double>(cdf[v] - cdfMin) * 255.0 /
                                                                static_cast<double>(N - cdfMin) + 0.5);
        CUDA_CHECK(cudaMemcpy(c.d_lut, lut, 256, cudaMemcpyHostToDevice));
        lutApplyKernel<<<(n + BLK - 1) / BLK, BLK>>>(c.d_in, c.d_out, n, c.d_lut);
        CUDA_CHECK(cudaGetLastError());
    }
    return downloadIfNeeded(src, c, xfer);
}

static cv::Mat gpuClahe(const cv::Mat& src, GpuCtx& c, bool xfer) {
    uploadIfNeeded(src, c, xfer);
    const double clipLimitParam = 2.0;
    const int gridX = 8, gridY = 8, W = c.W, H = c.H;
    const int padW = ((W + gridX - 1) / gridX) * gridX;
    const int padH = ((H + gridY - 1) / gridY) * gridY;
    const int tileW = padW / gridX, tileH = padH / gridY;
    const int tileArea = tileW * tileH;
    int clipLimit = static_cast<int>(clipLimitParam * tileArea / 256);
    clipLimit = std::max(clipLimit, 1);
    const float lutScale = static_cast<float>(256 - 1) / tileArea;
    const float invTw = 1.0f / tileW, invTh = 1.0f / tileH;

    claheLutKernel<<<gridX * gridY, 256>>>(c.d_in, W, H, gridX, tileW, tileH, clipLimit, lutScale, c.d_luts);
    CUDA_CHECK(cudaGetLastError());
    const dim3 block(16, 16);
    const dim3 grid((W + 15) / 16, (H + 15) / 16);
    claheInterpKernel<<<grid, block>>>(c.d_in, c.d_out, W, H, gridX, gridY, invTw, invTh, c.d_luts);
    CUDA_CHECK(cudaGetLastError());
    return downloadIfNeeded(src, c, xfer);
}

static cv::Mat gpuMean(const cv::Mat& src, GpuCtx& c, bool xfer) {
    uploadIfNeeded(src, c, xfer);
    const dim3 block(16, 16);
    const dim3 grid((c.W + 15) / 16, (c.H + 15) / 16);
    meanKernel<<<grid, block>>>(c.d_in, c.d_out, c.W, c.H);
    CUDA_CHECK(cudaGetLastError());
    return downloadIfNeeded(src, c, xfer);
}

static cv::Mat gpuMedian(const cv::Mat& src, GpuCtx& c, bool xfer) {
    uploadIfNeeded(src, c, xfer);
    const dim3 block(16, 16);
    const dim3 grid((c.W + 15) / 16, (c.H + 15) / 16);
    medianKernel<<<grid, block>>>(c.d_in, c.d_out, c.W, c.H);
    CUDA_CHECK(cudaGetLastError());
    return downloadIfNeeded(src, c, xfer);
}

// ---------------------------------------------------------------------------------------------
// Ana program
// ---------------------------------------------------------------------------------------------
struct GpuTask {
    std::string name, inputFile, outFile;
    std::function<cv::Mat(const cv::Mat&, GpuCtx&, bool)> gpu;
    std::function<cv::Mat(const cv::Mat&, int)> cpu;  // odev3_thread.cpp'deki CPU sürümü
    int cpuRepeat;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <veri_dizini> [cikti_dizini=cikti_cuda] [cpu_tekrar=5] [gpu_tekrar=50]\n";
        return 1;
    }
    const fs::path dataDir = argv[1];
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_cuda");
    const int cpuRepeat = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 5;
    const int gpuRepeat = (argc >= 5) ? std::max(1, std::stoi(argv[4])) : 50;
    fs::create_directories(outDir);

    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
        std::cerr << "Hata: GPU bulunamadi. Colab'da Calisma zamani turunu GPU (T4) yapin.\n";
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    const int mt = std::max(2, hw);  // CPU'nun çok thread'li ölçümü için thread sayısı

    std::cout << "GPU          : " << prop.name << " (hesaplama yetenegi " << prop.major << "." << prop.minor << ")\n";
    std::cout << "CPU thread   : " << hw << " (cok thread'li olcum " << mt << " thread ile)\n";
    std::cout << "Tekrar       : CPU " << cpuRepeat << " (medyan icin " << std::max(1, cpuRepeat / 3) << "), GPU " << gpuRepeat << "\n\n";

    const std::vector<GpuTask> tasks = {
        {"1. Kontrast germe", "1_dusuk_kontrast.png", "1_kontrast_germe_gpu.png", gpuContrast, contrastStretch, cpuRepeat},
        {"2. Histogram esitleme", "1_dusuk_kontrast.png", "2_histogram_esitleme_gpu.png", gpuHistEqualize, histEqualize, cpuRepeat},
        {"3. CLAHE (8x8, clip 2.0)", "1_dusuk_kontrast.png", "3_clahe_gpu.png", gpuClahe, claheParallel, cpuRepeat},
        {"4. Ortalama filtre 3x3", "2_gauss_gurultulu.png", "4_ortalama_3x3_gpu.png", gpuMean, meanFilterParallel, cpuRepeat},
        {"5. Medyan filtre 5x5", "3_tuz_biber.png", "5_medyan_5x5_gpu.png", gpuMedian, medianParallel, std::max(1, cpuRepeat / 3)},
    };

    GpuCtx ctx;
    {
        const cv::Mat first = cv::imread((dataDir / tasks[0].inputFile).string(), cv::IMREAD_GRAYSCALE);
        if (first.empty()) {
            std::cerr << "Hata: girdi okunamadi: " << (dataDir / tasks[0].inputFile).string() << "\n";
            return 1;
        }
        ctx.init(first.cols, first.rows);
    }

    struct Row {
        double cpu1, cpuN, gpuKernel, gpuTotal;
        double maxDiff;
        int diffPixels;
    };
    std::vector<Row> rows(tasks.size());

    for (size_t k = 0; k < tasks.size(); ++k) {
        const GpuTask& t = tasks[k];
        cv::Mat input = cv::imread((dataDir / t.inputFile).string(), cv::IMREAD_GRAYSCALE);
        if (input.empty() || input.cols != ctx.W || input.rows != ctx.H) {
            std::cerr << "Hata: girdi okunamadi ya da boyutu farkli: " << t.inputFile << "\n";
            return 1;
        }
        if (!input.isContinuous()) input = input.clone();
        std::cout << t.name << " calisiyor...\n";

        // Doğruluk: GPU sonucu ile CPU (tek thread) sonucunu karşılaştır
        const cv::Mat cpuRef = t.cpu(input, 1);
        const cv::Mat gpuOut = t.gpu(input, ctx, true);
        cv::Mat diff;
        cv::absdiff(gpuOut, cpuRef, diff);
        double maxDiff = 0;
        cv::minMaxLoc(diff, nullptr, &maxDiff);
        rows[k].maxDiff = maxDiff;
        rows[k].diffPixels = cv::countNonZero(diff);
        cv::imwrite((outDir / t.outFile).string(), gpuOut);

        // Süre ölçümleri
        rows[k].cpu1 = averageMs([&] { g_sink = g_sink + t.cpu(input, 1).at<uchar>(0, 0); }, t.cpuRepeat);
        rows[k].cpuN = averageMs([&] { g_sink = g_sink + t.cpu(input, mt).at<uchar>(0, 0); }, t.cpuRepeat);
        t.gpu(input, ctx, true);  // girdiyi GPU belleğine yerleştir
        rows[k].gpuKernel = averageMs([&] { t.gpu(input, ctx, false); }, gpuRepeat);
        rows[k].gpuTotal = averageMs([&] { t.gpu(input, ctx, true); }, gpuRepeat);
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nTABLO 1 - Calisma suresi (ms):\n";
    std::cout << "  " << std::left << std::setw(28) << "Gorev" << std::right << std::setw(14) << "CPU 1 thread" << std::setw(5)
              << mt << " thr CPU" << std::setw(14) << "GPU cekirdek" << std::setw(12) << "GPU toplam" << "\n";
    for (size_t k = 0; k < tasks.size(); ++k)
        std::cout << "  " << std::left << std::setw(28) << tasks[k].name << std::right << std::setw(14) << rows[k].cpu1
                  << std::setw(13) << rows[k].cpuN << std::setw(14) << rows[k].gpuKernel << std::setw(12) << rows[k].gpuTotal << "\n";

    std::cout << std::setprecision(2);
    std::cout << "\nTABLO 2 - Hizlanma (CPU suresi / GPU suresi):\n";
    std::cout << "  " << std::left << std::setw(28) << "Gorev" << std::right << std::setw(20) << "CPU 1 / GPU cekirdek"
              << std::setw(18) << "CPU 1 / GPU toplam" << std::setw(18) << "CPU N / GPU toplam" << "\n";
    for (size_t k = 0; k < tasks.size(); ++k)
        std::cout << "  " << std::left << std::setw(28) << tasks[k].name << std::right << std::setw(19)
                  << rows[k].cpu1 / rows[k].gpuKernel << "x" << std::setw(17) << rows[k].cpu1 / rows[k].gpuTotal << "x"
                  << std::setw(17) << rows[k].cpuN / rows[k].gpuTotal << "x\n";

    std::cout << "\nTABLO 3 - Dogruluk (GPU sonucu ile CPU tek thread sonucu):\n";
    for (size_t k = 0; k < tasks.size(); ++k)
        std::cout << "  " << std::left << std::setw(28) << tasks[k].name << std::right << "en buyuk fark = " << rows[k].maxDiff
                  << ", farkli piksel = " << rows[k].diffPixels << (rows[k].diffPixels == 0 ? "  (birebir ayni)" : "") << "\n";

    std::cout << "\nKaydedilenler (" << outDir.string() << "): her gorevin GPU sonucu\n";
    for (const auto& t : tasks) std::cout << "  " << t.outFile << "\n";

    ctx.release();
    return 0;
}

// Ödev 3 - Haftaya görevi (1. kısım): Beş sorunun multi-threaded (çok iş parçacıklı) sürümleri.
//
//   1) Doğrusal kontrast germe       (min/max paralel bulunur, ölçekleme paralel)
//   2) Histogram eşitleme            (her thread kendi histogramını çıkarır, birleştirilir; dönüşüm paralel)
//   3) CLAHE (elle yazılan)          (kareler paralel, interpolasyon satırlara bölünerek paralel)
//   4) 3x3 ortalama filtre           (satırlar thread'lere bölünür)
//   5) 5x5 medyan filtre             (satırlar thread'lere bölünür)
//
// Her görev TEK bir fonksiyonla yazılmıştır ve thread sayısı parametre olarak verilir; thread sayısı 1
// olduğunda iş sırayla (tek thread) yapılır. Böylece 1 thread ile N thread karşılaştırması aynı kodla,
// adil biçimde yapılır. Her thread sayısı için sonucun 1 thread sonucuyla BİREBİR aynı olduğu doğrulanır.
//
// Paralelleştirme yöntemi: işin bölünebilir ekseni (görüntü satırları ya da CLAHE kareleri) thread sayısı
// kadar ardışık parçaya bölünür; her thread kendi parçasını bağımsız işler, ortak veriye yazmaz.
//
// Girdiler (veri_hazirla.exe ile üretilmiş olmalı, varsayılan dizin: veri):
//   veri/1_dusuk_kontrast.png   -> görev 1, 2, 3
//   veri/2_gauss_gurultulu.png  -> görev 4
//   veri/3_tuz_biber.png        -> görev 5
//
// Kullanım: ./thread_test <veri_dizini> [cikti_dizini] [tekrar_sayisi=10]
// Derleme:  g++ -O2 odev3_thread.cpp -o thread_test.exe -std=c++17 -pthread $(pkg-config --cflags --libs opencv5)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static const int HIST = 256;
static volatile double g_sink = 0.0;  // derleyicinin ölçülen işi silmesini önlemek için

// ---------------------------------------------------------------------------------------------
// Yardımcılar
// ---------------------------------------------------------------------------------------------
static int reflect101(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
    }
    return i;
}

static int clampIdx(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

// n parçalık işi en fazla 'threads' parçaya böler (threads <= 1 ise tek parça)
static int numChunks(int n, int threads) { return (threads <= 1 || n <= 1) ? 1 : std::min(threads, n); }

// [0, n) aralığını ardışık parçalara bölüp her parçayı ayrı thread'de çalıştırır.
// fn(parcaNo, baslangic, bitis) biçiminde çağrılır. Ana thread de ilk parçayı işler.
template <typename Fn>
static void parallelFor(int n, int threads, Fn fn) {
    const int c = numChunks(n, threads);
    if (c == 1) {
        fn(0, 0, n);
        return;
    }
    const int chunk = (n + c - 1) / c;
    std::vector<std::thread> workers;
    workers.reserve(c - 1);
    for (int t = 1; t < c; ++t) {
        const int b = t * chunk, e = std::min(n, b + chunk);
        if (b < e) workers.emplace_back([&fn, t, b, e] { fn(t, b, e); });
    }
    fn(0, 0, std::min(n, chunk));
    for (auto& w : workers) w.join();
}

template <typename Fn>
static double averageMs(Fn fn, int repeat) {
    fn();  // ısınma turu
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
}

// ---------------------------------------------------------------------------------------------
// Görev 1: doğrusal kontrast germe
// ---------------------------------------------------------------------------------------------
static cv::Mat contrastStretch(const cv::Mat& src, int threads) {
    const int c = numChunks(src.rows, threads);
    std::vector<int> mins(c, 255), maxs(c, 0);  // her parçanın kendi min/max değeri

    parallelFor(src.rows, threads, [&](int id, int b, int e) {
        int lo = 255, hi = 0;
        for (int y = b; y < e; ++y) {
            const uchar* s = src.ptr<uchar>(y);
            for (int x = 0; x < src.cols; ++x) {
                if (s[x] < lo) lo = s[x];
                if (s[x] > hi) hi = s[x];
            }
        }
        mins[id] = lo;
        maxs[id] = hi;
    });

    int lo = 255, hi = 0;
    for (int i = 0; i < c; ++i) {
        lo = std::min(lo, mins[i]);
        hi = std::max(hi, maxs[i]);
    }

    cv::Mat out(src.size(), CV_8UC1);
    if (hi == lo) {
        src.copyTo(out);
        return out;
    }
    const double scale = 255.0 / (hi - lo);
    parallelFor(src.rows, threads, [&](int, int b, int e) {
        for (int y = b; y < e; ++y) {
            const uchar* s = src.ptr<uchar>(y);
            uchar* d = out.ptr<uchar>(y);
            for (int x = 0; x < src.cols; ++x) d[x] = static_cast<uchar>((s[x] - lo) * scale + 0.5);
        }
    });
    return out;
}

// ---------------------------------------------------------------------------------------------
// Görev 2: histogram eşitleme
// ---------------------------------------------------------------------------------------------
static cv::Mat histEqualize(const cv::Mat& src, int threads) {
    const int c = numChunks(src.rows, threads);
    std::vector<std::array<unsigned int, HIST>> part(c);
    for (auto& p : part) p.fill(0);

    // Her thread kendi parçasının histogramını kendi dizisine çıkarır
    parallelFor(src.rows, threads, [&](int id, int b, int e) {
        auto& h = part[id];
        for (int y = b; y < e; ++y) {
            const uchar* s = src.ptr<uchar>(y);
            for (int x = 0; x < src.cols; ++x) ++h[s[x]];
        }
    });

    // Parça histogramlarını birleştir, CDF'yi çıkar (256 elemanlık, paralelleştirmeye değmez)
    std::array<unsigned long long, HIST> hist, cdf;
    hist.fill(0);
    for (int i = 0; i < c; ++i)
        for (int b = 0; b < HIST; ++b) hist[b] += part[i][b];
    unsigned long long run = 0;
    for (int b = 0; b < HIST; ++b) {
        run += hist[b];
        cdf[b] = run;
    }
    const unsigned long long N = static_cast<unsigned long long>(src.rows) * src.cols;
    unsigned long long cdfMin = 0;
    for (int b = 0; b < HIST; ++b) {
        if (hist[b] > 0) {
            cdfMin = cdf[b];
            break;
        }
    }

    cv::Mat out(src.size(), CV_8UC1);
    if (cdfMin == N) {
        src.copyTo(out);
        return out;
    }
    std::array<uchar, HIST> lut;
    for (int v = 0; v < HIST; ++v) {
        lut[v] = (cdf[v] < cdfMin) ? 0
                                   : static_cast<uchar>(static_cast<double>(cdf[v] - cdfMin) * 255.0 /
                                                            static_cast<double>(N - cdfMin) + 0.5);
    }

    // Dönüşüm tablosunu her piksele paralel uygula
    parallelFor(src.rows, threads, [&](int, int b, int e) {
        for (int y = b; y < e; ++y) {
            const uchar* s = src.ptr<uchar>(y);
            uchar* d = out.ptr<uchar>(y);
            for (int x = 0; x < src.cols; ++x) d[x] = lut[s[x]];
        }
    });
    return out;
}

// ---------------------------------------------------------------------------------------------
// Görev 3: CLAHE (clip limit 2.0, 8x8 kare), elle yazılmış
// ---------------------------------------------------------------------------------------------
static cv::Mat claheParallel(const cv::Mat& src, int threads) {
    const double clipLimitParam = 2.0;
    const int gridX = 8, gridY = 8;
    const int W = src.cols, H = src.rows;

    const int padW = ((W + gridX - 1) / gridX) * gridX;
    const int padH = ((H + gridY - 1) / gridY) * gridY;
    const int tileW = padW / gridX, tileH = padH / gridY;
    const int tileArea = tileW * tileH;

    int clipLimit = static_cast<int>(clipLimitParam * tileArea / HIST);
    clipLimit = std::max(clipLimit, 1);
    const float lutScale = static_cast<float>(HIST - 1) / tileArea;

    std::vector<uchar> luts(static_cast<size_t>(gridX) * gridY * HIST);

    // Karelerin histogram, kırpma ve tablo hesapları birbirinden bağımsızdır: kareler thread'lere bölünür
    parallelFor(gridX * gridY, threads, [&](int, int tb, int te) {
        for (int t = tb; t < te; ++t) {
            const int ty = t / gridX, tx = t % gridX;
            std::array<int, HIST> hist;
            hist.fill(0);
            for (int yy = ty * tileH; yy < (ty + 1) * tileH; ++yy) {
                const uchar* s = src.ptr<uchar>(reflect101(yy, H));
                for (int xx = tx * tileW; xx < (tx + 1) * tileW; ++xx) ++hist[s[reflect101(xx, W)]];
            }

            int clipped = 0;
            for (int i = 0; i < HIST; ++i) {
                if (hist[i] > clipLimit) {
                    clipped += hist[i] - clipLimit;
                    hist[i] = clipLimit;
                }
            }
            const int redistBatch = clipped / HIST;
            int residual = clipped - redistBatch * HIST;
            for (int i = 0; i < HIST; ++i) hist[i] += redistBatch;
            if (residual != 0) {
                const int residualStep = std::max(HIST / residual, 1);
                for (int i = 0; i < HIST && residual > 0; i += residualStep, --residual) ++hist[i];
            }

            uchar* lut = &luts[static_cast<size_t>(t) * HIST];
            int sum = 0;
            for (int i = 0; i < HIST; ++i) {
                sum += hist[i];
                lut[i] = cv::saturate_cast<uchar>(sum * lutScale);
            }
        }
    });

    // Bilineer interpolasyon: her satır bağımsızdır, satırlar thread'lere bölünür
    cv::Mat dst(src.size(), CV_8UC1);
    const float invTw = 1.0f / tileW, invTh = 1.0f / tileH;
    std::vector<int> xi1(W), xi2(W);
    std::vector<float> xa(W);
    for (int x = 0; x < W; ++x) {
        const float txf = x * invTw - 0.5f;
        int tx1 = static_cast<int>(std::floor(txf));
        const float a = txf - tx1;
        int tx2 = tx1 + 1;
        tx1 = std::max(tx1, 0);
        tx2 = std::min(tx2, gridX - 1);
        xi1[x] = tx1;
        xi2[x] = tx2;
        xa[x] = a;
    }

    parallelFor(H, threads, [&](int, int yb, int ye) {
        for (int y = yb; y < ye; ++y) {
            const float tyf = y * invTh - 0.5f;
            int ty1 = static_cast<int>(std::floor(tyf));
            const float ya = tyf - ty1;
            int ty2 = ty1 + 1;
            ty1 = std::max(ty1, 0);
            ty2 = std::min(ty2, gridY - 1);

            const uchar* s = src.ptr<uchar>(y);
            uchar* d = dst.ptr<uchar>(y);
            for (int x = 0; x < W; ++x) {
                const int v = s[x];
                const float l11 = luts[(static_cast<size_t>(ty1) * gridX + xi1[x]) * HIST + v];
                const float l12 = luts[(static_cast<size_t>(ty1) * gridX + xi2[x]) * HIST + v];
                const float l21 = luts[(static_cast<size_t>(ty2) * gridX + xi1[x]) * HIST + v];
                const float l22 = luts[(static_cast<size_t>(ty2) * gridX + xi2[x]) * HIST + v];
                const float top = l11 * (1.0f - xa[x]) + l12 * xa[x];
                const float bottom = l21 * (1.0f - xa[x]) + l22 * xa[x];
                d[x] = cv::saturate_cast<uchar>(top * (1.0f - ya) + bottom * ya);
            }
        }
    });
    return dst;
}

// ---------------------------------------------------------------------------------------------
// Görev 4: 3x3 ortalama filtre (konvolüsyon, her eleman 1/9)
// ---------------------------------------------------------------------------------------------
static cv::Mat meanFilterParallel(const cv::Mat& src, int threads) {
    const int W = src.cols, H = src.rows;
    cv::Mat dst(src.size(), CV_8UC1);
    std::vector<int> colIdx(static_cast<size_t>(W) * 3);
    for (int x = 0; x < W; ++x)
        for (int k = 0; k < 3; ++k) colIdx[static_cast<size_t>(x) * 3 + k] = reflect101(x + k - 1, W);
    const float w = 1.0f / 9.0f;

    parallelFor(H, threads, [&](int, int yb, int ye) {
        for (int y = yb; y < ye; ++y) {
            const uchar* r0 = src.ptr<uchar>(reflect101(y - 1, H));
            const uchar* r1 = src.ptr<uchar>(y);
            const uchar* r2 = src.ptr<uchar>(reflect101(y + 1, H));
            uchar* d = dst.ptr<uchar>(y);
            for (int x = 0; x < W; ++x) {
                const int* ci = &colIdx[static_cast<size_t>(x) * 3];
                const float sum = w * r0[ci[0]] + w * r0[ci[1]] + w * r0[ci[2]] + w * r1[ci[0]] + w * r1[ci[1]] +
                                  w * r1[ci[2]] + w * r2[ci[0]] + w * r2[ci[1]] + w * r2[ci[2]];
                d[x] = cv::saturate_cast<uchar>(sum);
            }
        }
    });
    return dst;
}

// ---------------------------------------------------------------------------------------------
// Görev 5: 5x5 medyan filtre (ekleme sıralamasıyla, kenarlar en yakın kenar pikseliyle doldurulur)
// ---------------------------------------------------------------------------------------------
static cv::Mat medianParallel(const cv::Mat& src, int threads) {
    const int W = src.cols, H = src.rows;
    cv::Mat dst(src.size(), CV_8UC1);
    std::vector<int> colIdx(static_cast<size_t>(W) * 5);
    for (int x = 0; x < W; ++x)
        for (int k = 0; k < 5; ++k) colIdx[static_cast<size_t>(x) * 5 + k] = clampIdx(x + k - 2, W);

    parallelFor(H, threads, [&](int, int yb, int ye) {
        for (int y = yb; y < ye; ++y) {
            const uchar* rows[5];
            for (int ky = 0; ky < 5; ++ky) rows[ky] = src.ptr<uchar>(clampIdx(y + ky - 2, H));
            uchar* d = dst.ptr<uchar>(y);
            for (int x = 0; x < W; ++x) {
                uchar v[25];
                int n = 0;
                for (int ky = 0; ky < 5; ++ky)
                    for (int kx = 0; kx < 5; ++kx) v[n++] = rows[ky][colIdx[static_cast<size_t>(x) * 5 + kx]];
                for (int i = 1; i < 25; ++i) {
                    const uchar key = v[i];
                    int j = i - 1;
                    while (j >= 0 && v[j] > key) {
                        v[j + 1] = v[j];
                        --j;
                    }
                    v[j + 1] = key;
                }
                d[x] = v[12];
            }
        }
    });
    return dst;
}

// ---------------------------------------------------------------------------------------------
// Ana program: her görevi 1, 2, 4, 8 thread ile çalıştır, doğrula, süreleri karşılaştır
// ---------------------------------------------------------------------------------------------
struct Task {
    std::string name;
    std::string inputFile;
    std::string outFile;
    std::function<cv::Mat(const cv::Mat&, int)> fn;
    int repeat;
};

#ifndef THREAD_TEST_NO_MAIN  // CUDA programi bu dosyayi kendi main'i ile birlikte kullanir
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <veri_dizini> [cikti_dizini] [tekrar_sayisi=10]\n";
        return 1;
    }
    const fs::path dataDir = argv[1];
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_thread");
    const int repeat = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 10;
    fs::create_directories(outDir);

    const std::vector<Task> tasks = {
        {"1. Kontrast germe", "1_dusuk_kontrast.png", "1_kontrast_germe_4thread.png", contrastStretch, repeat},
        {"2. Histogram esitleme", "1_dusuk_kontrast.png", "2_histogram_esitleme_4thread.png", histEqualize, repeat},
        {"3. CLAHE (8x8, clip 2.0)", "1_dusuk_kontrast.png", "3_clahe_4thread.png", claheParallel, repeat},
        {"4. Ortalama filtre 3x3", "2_gauss_gurultulu.png", "4_ortalama_3x3_4thread.png", meanFilterParallel, repeat},
        {"5. Medyan filtre 5x5", "3_tuz_biber.png", "5_medyan_5x5_4thread.png", medianParallel, std::max(1, repeat / 3)},
    };
    const std::vector<int> threadCounts = {1, 2, 4, 8};

    std::cout << "Islemci thread sayisi (bilgisayar): " << std::thread::hardware_concurrency() << "\n";
    std::cout << "Tekrar sayisi: " << repeat << " (medyan icin " << std::max(1, repeat / 3) << "), sureler ortalamadir\n\n";

    std::vector<std::vector<double>> times(tasks.size(), std::vector<double>(threadCounts.size(), 0.0));
    std::vector<bool> allSame(tasks.size(), true);

    for (size_t k = 0; k < tasks.size(); ++k) {
        const Task& task = tasks[k];
        const cv::Mat input = cv::imread((dataDir / task.inputFile).string(), cv::IMREAD_GRAYSCALE);
        if (input.empty()) {
            std::cerr << "Hata: girdi okunamadi: " << (dataDir / task.inputFile).string() << "\n";
            return 1;
        }
        std::cout << task.name << " calisiyor (" << input.cols << "x" << input.rows << ")...\n";

        const cv::Mat reference = task.fn(input, 1);
        for (size_t ti = 0; ti < threadCounts.size(); ++ti) {
            const int tc = threadCounts[ti];
            const cv::Mat out = task.fn(input, tc);
            if (cv::norm(out, reference, cv::NORM_INF) != 0.0) allSame[k] = false;
            times[k][ti] = averageMs(
                [&] {
                    const cv::Mat r = task.fn(input, tc);
                    g_sink = g_sink + r.at<uchar>(0, 0);
                },
                task.repeat);
            if (tc == 4) cv::imwrite((outDir / task.outFile).string(), out);
        }
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nTABLO 1 - Calisma suresi (ms):\n";
    std::cout << "  " << std::left << std::setw(28) << "Gorev";
    for (int tc : threadCounts) std::cout << std::right << std::setw(12) << (std::to_string(tc) + " thread");
    std::cout << std::setw(14) << "Sonuc ayni mi\n";
    for (size_t k = 0; k < tasks.size(); ++k) {
        std::cout << "  " << std::left << std::setw(28) << tasks[k].name;
        for (size_t ti = 0; ti < threadCounts.size(); ++ti) std::cout << std::right << std::setw(12) << times[k][ti];
        std::cout << std::setw(12) << (allSame[k] ? "EVET" : "HAYIR") << "\n";
    }

    std::cout << "\nTABLO 2 - Hizlanma (1 thread suresi / N thread suresi):\n";
    std::cout << "  " << std::left << std::setw(28) << "Gorev";
    for (int tc : threadCounts) std::cout << std::right << std::setw(12) << (std::to_string(tc) + " thread");
    std::cout << "\n";
    for (size_t k = 0; k < tasks.size(); ++k) {
        std::cout << "  " << std::left << std::setw(28) << tasks[k].name;
        for (size_t ti = 0; ti < threadCounts.size(); ++ti)
            std::cout << std::right << std::setw(11) << (times[k][0] / times[k][ti]) << "x";
        std::cout << "\n";
    }

    std::cout << "\nKaydedilenler (" << outDir.string() << "): her gorevin 4 thread ile uretilen sonucu\n";
    for (const auto& t : tasks) std::cout << "  " << t.outFile << "\n";
    return 0;
}
#endif  // THREAD_TEST_NO_MAIN

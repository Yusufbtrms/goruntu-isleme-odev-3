// Ödev 3 - Soru 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
//   A) OpenCV'nin hazır cv::createCLAHE fonksiyonu ile düzeltme
//   B) Aynı işlemin HAZIR FONKSİYON KULLANILMADAN elle gerçeklenmesi ve iki sonucun karşılaştırılması
//
// CLAHE'nin adımları (elle yazılan sürüm bunları izler):
//   1) Görüntü gridX x gridY karelere (tile) bölünür. Boyut tam bölünmüyorsa alt/sağ kenar
//      yansıtılarak (reflect101) doldurulur.
//   2) Her karenin kendi 256'lık histogramı çıkarılır.
//   3) Histogram "kırpılır": clipLimit'ten yüksek çubuklar kesilir, fazlalık tüm çubuklara eşit dağıtılır.
//      (Böylece gürültü ve düz bölgeler aşırı güçlendirilmez.)
//   4) Kırpılmış histogramdan CDF çıkarılıp her kare için 256 değerli dönüşüm tablosu (LUT) yapılır.
//   5) Her piksel, kendisine en yakın 4 karenin LUT'larından bilineer interpolasyonla dönüştürülür
//      (kareler arasında keskin sınır oluşmasın diye).
//
// Kullanım: ./clahe <dusuk_kontrastli_goruntu> [cikti_dizini] [clip_limit=2.0] [grid=8]
//
// Derleme:
//   g++ -O2 odev3_3_clahe.cpp -o clahe.exe -std=c++17 $(pkg-config --cflags --libs opencv5)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const int HIST = 256;

// Kenar yansıtma (reflect101): sınırın dışına taşan indeksi görüntünün içine yansıtır.
static int reflect101(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
    }
    return i;
}

// Elle yazılmış CLAHE (yalnızca 8-bit, tek kanallı görüntü)
static cv::Mat myClahe(const cv::Mat& src, double clipLimitParam, int gridX, int gridY) {
    const int W = src.cols, H = src.rows;

    // 1) Karelerin boyutu: görüntü boyutu tam bölünmüyorsa yukarı yuvarla (doldurulmuş boyut)
    const int padW = ((W + gridX - 1) / gridX) * gridX;
    const int padH = ((H + gridY - 1) / gridY) * gridY;
    const int tileW = padW / gridX, tileH = padH / gridY;
    const int tileArea = tileW * tileH;

    // Kırpma sınırı: clipLimit, bir karedeki ortalama çubuk yüksekliğinin kaç katı olabileceğini belirler
    int clipLimit = 0;
    if (clipLimitParam > 0.0) {
        clipLimit = static_cast<int>(clipLimitParam * tileArea / HIST);
        clipLimit = std::max(clipLimit, 1);
    }
    const float lutScale = static_cast<float>(HIST - 1) / tileArea;

    // Her kare için 256 değerli dönüşüm tablosu
    std::vector<uchar> luts(static_cast<size_t>(gridX) * gridY * HIST);

    for (int ty = 0; ty < gridY; ++ty) {
        for (int tx = 0; tx < gridX; ++tx) {
            // 2) Karenin histogramı (doldurulmuş bölge yansıtılarak okunur)
            std::array<int, HIST> hist;
            hist.fill(0);
            for (int yy = ty * tileH; yy < (ty + 1) * tileH; ++yy) {
                const uchar* s = src.ptr<uchar>(reflect101(yy, H));
                for (int xx = tx * tileW; xx < (tx + 1) * tileW; ++xx) ++hist[s[reflect101(xx, W)]];
            }

            // 3) Histogramı kırp ve fazlalığı eşit dağıt
            if (clipLimit > 0) {
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
            }

            // 4) CDF'den dönüşüm tablosu
            uchar* lut = &luts[(static_cast<size_t>(ty) * gridX + tx) * HIST];
            int sum = 0;
            for (int i = 0; i < HIST; ++i) {
                sum += hist[i];
                lut[i] = cv::saturate_cast<uchar>(sum * lutScale);
            }
        }
    }

    // 5) Bilineer interpolasyonla her pikseli dönüştür
    cv::Mat dst(src.size(), CV_8UC1);
    const float invTw = 1.0f / tileW, invTh = 1.0f / tileH;

    // x yönündeki komşu kare indeksleri ve ağırlıkları bir kez hesaplanır
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

    for (int y = 0; y < H; ++y) {
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
    return dst;
}

static void printStatsRow(const std::string& name, const cv::Mat& img) {
    cv::Scalar m, s;
    cv::meanStdDev(img, m, s);
    double mn = 0, mx = 0;
    cv::minMaxLoc(img, &mn, &mx);
    std::cout << std::left << std::setw(26) << name << std::right << std::fixed << std::setprecision(2)
              << std::setw(10) << m[0] << std::setw(12) << s[0] << std::setw(8) << mn << std::setw(8) << mx
              << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <dusuk_kontrastli_goruntu> [cikti_dizini] [clip_limit=2.0] [grid=8]\n";
        return 1;
    }
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_clahe");
    const double clip = (argc >= 4) ? std::stod(argv[3]) : 2.0;
    const int grid = (argc >= 5) ? std::max(1, std::stoi(argv[4])) : 8;
    fs::create_directories(outDir);

    cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Hata: goruntu okunamadi: " << argv[1] << "\n";
        return 1;
    }

    // A) OpenCV'nin hazır CLAHE fonksiyonu
    cv::Mat viaOpenCv;
    auto t0 = std::chrono::steady_clock::now();
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip, cv::Size(grid, grid));
        clahe->apply(src, viaOpenCv);
    }
    auto t1 = std::chrono::steady_clock::now();

    // B) Elle yazılan CLAHE
    cv::Mat mine = myClahe(src, clip, grid, grid);
    auto t2 = std::chrono::steady_clock::now();

    const double msOpenCv = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double msMine = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "Goruntu boyutu : " << src.cols << "x" << src.rows << " (gri seviye)\n";
    std::cout << "Clip limit     : " << clip << ", kare izgarasi: " << grid << "x" << grid << "\n\n";

    std::cout << std::left << std::setw(26) << "Goruntu" << std::right << std::setw(10) << "Ortalama"
              << std::setw(12) << "Std.sapma" << std::setw(8) << "Min" << std::setw(8) << "Maks" << "\n";
    printStatsRow("Girdi (dusuk kontrast)", src);
    printStatsRow("CLAHE (OpenCV hazir)", viaOpenCv);
    printStatsRow("CLAHE (elle yazilan)", mine);

    // Test: iki sonucun karşılaştırılması
    cv::Mat diff;
    cv::absdiff(viaOpenCv, mine, diff);
    double maxDiff = 0;
    cv::minMaxLoc(diff, nullptr, &maxDiff);
    const int differing = cv::countNonZero(diff);
    const double total = static_cast<double>(src.rows) * src.cols;
    std::cout << "\nTEST: elle yazilan CLAHE ile OpenCV CLAHE karsilastirmasi\n";
    std::cout << "  En buyuk mutlak fark      : " << maxDiff << "\n";
    std::cout << "  Ortalama mutlak fark      : " << std::setprecision(4) << cv::mean(diff)[0] << "\n";
    std::cout << "  Farkli piksel sayisi      : " << differing << " (" << std::setprecision(3)
              << differing / total * 100.0 << " %)\n";
    std::cout << "\nSure: OpenCV " << std::setprecision(2) << msOpenCv << " ms, elle yazilan " << msMine
              << " ms (tek calistirma)\n";

    // Görseller
    auto small = [](const cv::Mat& m) {
        cv::Mat r;
        const double s = 480.0 / m.cols;
        cv::resize(m, r, cv::Size(), s, s, cv::INTER_AREA);
        return r;
    };
    cv::Mat montage, diffVis;
    cv::hconcat(std::vector<cv::Mat>{small(src), small(viaOpenCv), small(mine)}, montage);
    diff.convertTo(diffVis, CV_8U, 50.0);  // farkı gözle görülür yapmak için 50 kat büyüt

    cv::imwrite((outDir / "1_dusuk_kontrast.png").string(), src);
    cv::imwrite((outDir / "2_clahe_opencv.png").string(), viaOpenCv);
    cv::imwrite((outDir / "3_clahe_elle.png").string(), mine);
    cv::imwrite((outDir / "4_yan_yana_girdi_opencv_elle.png").string(), montage);
    cv::imwrite((outDir / "5_fark_x50.png").string(), diffVis);

    std::cout << "\nKaydedilenler (" << outDir.string() << "):\n"
              << "  1_dusuk_kontrast.png              : girdi\n"
              << "  2_clahe_opencv.png                : OpenCV hazir CLAHE sonucu\n"
              << "  3_clahe_elle.png                  : elle yazilan CLAHE sonucu\n"
              << "  4_yan_yana_girdi_opencv_elle.png  : soldan saga girdi | OpenCV | elle\n"
              << "  5_fark_x50.png                    : iki sonuc arasindaki fark (50 kat buyutulmus)\n";
    return 0;
}

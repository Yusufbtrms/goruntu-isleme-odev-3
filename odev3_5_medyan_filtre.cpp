// Ödev 3 - Soru 5: Tuz-biber (salt-and-pepper) gürültülü görüntüde gürültü bastırma.
//   A) OpenCV'nin hazır cv::medianBlur fonksiyonu (5x5)
//   B) 5x5 medyan filtrenin HAZIR FONKSİYON KULLANILMADAN elle yazılması
//
// Medyan filtre: Her pikselin etrafındaki 5x5'lik komşuluktaki 25 değer küçükten büyüğe sıralanır,
// ortadaki (13.) değer piksele yazılır. Tuz (255) ve biber (0) gibi uç değerler sıralamada hep
// başa/sona düştüğü için ortadaki değeri etkilemez; bu yüzden bu tür gürültüde ortalama
// filtresinden çok daha başarılıdır.
//
// Kenarlar: Görüntü dışına taşan komşular en yakın kenar pikseliyle doldurulur (replicate);
// cv::medianBlur da aynı yöntemi kullanır.
//
// Kullanım: ./medyan_filtre <tuz_biber_goruntu> [cikti_dizini] [temiz_referans_goruntu]
//
// Derleme:
//   g++ -O2 odev3_5_medyan_filtre.cpp -o medyan_filtre.exe -std=c++17 \
//       $(pkg-config --cflags --libs opencv5)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Sınır dışına taşan indeksi en yakın kenara çeker (replicate)
static int clampIdx(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

// Elle yazılmış 5x5 medyan filtre (hazır sıralama/medyan fonksiyonu yok)
static cv::Mat medianFilter5x5(const cv::Mat& src) {
    const int W = src.cols, H = src.rows;
    cv::Mat dst(src.size(), CV_8UC1);

    std::vector<int> colIdx(static_cast<size_t>(W) * 5);
    for (int x = 0; x < W; ++x)
        for (int k = 0; k < 5; ++k) colIdx[static_cast<size_t>(x) * 5 + k] = clampIdx(x + k - 2, W);

    for (int y = 0; y < H; ++y) {
        const uchar* rows[5];
        for (int ky = 0; ky < 5; ++ky) rows[ky] = src.ptr<uchar>(clampIdx(y + ky - 2, H));
        uchar* d = dst.ptr<uchar>(y);

        for (int x = 0; x < W; ++x) {
            // 5x5 komşuluktaki 25 değeri topla
            uchar v[25];
            int n = 0;
            for (int ky = 0; ky < 5; ++ky)
                for (int kx = 0; kx < 5; ++kx) v[n++] = rows[ky][colIdx[static_cast<size_t>(x) * 5 + kx]];

            // Ekleme sıralaması (insertion sort) ile küçükten büyüğe sırala
            for (int i = 1; i < 25; ++i) {
                const uchar key = v[i];
                int j = i - 1;
                while (j >= 0 && v[j] > key) {
                    v[j + 1] = v[j];
                    --j;
                }
                v[j + 1] = key;
            }

            d[x] = v[12];  // 25 değerin ortancası (13. eleman, indeks 12)
        }
    }
    return dst;
}

static double meanSquaredError(const cv::Mat& a, const cv::Mat& b) {
    double sum = 0.0;
    for (int y = 0; y < a.rows; ++y) {
        const uchar* pa = a.ptr<uchar>(y);
        const uchar* pb = b.ptr<uchar>(y);
        for (int x = 0; x < a.cols; ++x) {
            const double d = static_cast<double>(pa[x]) - pb[x];
            sum += d * d;
        }
    }
    return sum / (static_cast<double>(a.rows) * a.cols);
}

static double psnr(double mse) { return mse <= 0 ? INFINITY : 10.0 * std::log10(255.0 * 255.0 / mse); }

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <tuz_biber_goruntu> [cikti_dizini] [temiz_referans_goruntu]\n";
        return 1;
    }
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_medyan_filtre");
    fs::create_directories(outDir);

    cv::Mat noisy = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (noisy.empty()) {
        std::cerr << "Hata: goruntu okunamadi: " << argv[1] << "\n";
        return 1;
    }

    // Görüntüdeki uç değerli (0 ya da 255) piksellerin oranı (tuz-biber gürültüsünün göstergesi)
    long long extremes = 0;
    for (int y = 0; y < noisy.rows; ++y) {
        const uchar* s = noisy.ptr<uchar>(y);
        for (int x = 0; x < noisy.cols; ++x) extremes += (s[x] == 0 || s[x] == 255) ? 1 : 0;
    }
    const double total = static_cast<double>(noisy.rows) * noisy.cols;

    std::cout << "Goruntu boyutu : " << noisy.cols << "x" << noisy.rows << " (gri seviye)\n";
    std::cout << "0 veya 255 degerli piksel orani (tuz-biber gostergesi): " << std::fixed << std::setprecision(2)
              << extremes / total * 100.0 << " %\n\n";

    // A) OpenCV hazır medyan filtre (5x5)
    cv::Mat viaOpenCv;
    auto t0 = std::chrono::steady_clock::now();
    cv::medianBlur(noisy, viaOpenCv, 5);
    auto t1 = std::chrono::steady_clock::now();

    // B) Elle yazılan 5x5 medyan filtre
    cv::Mat mine = medianFilter5x5(noisy);
    auto t2 = std::chrono::steady_clock::now();

    const double msOpenCv = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double msMine = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Karşılaştırma için: aynı boyutta (5x5) ortalama filtre
    cv::Mat box5;
    cv::blur(noisy, box5, cv::Size(5, 5));

    // Test: elle yazılan ile OpenCV sonucu
    cv::Mat diff;
    cv::absdiff(viaOpenCv, mine, diff);
    double maxDiff = 0;
    cv::minMaxLoc(diff, nullptr, &maxDiff);
    std::cout << "TEST: elle yazilan 5x5 medyan ile cv::medianBlur(5) karsilastirmasi\n";
    std::cout << "  En buyuk mutlak fark : " << maxDiff << "\n";
    std::cout << "  Farkli piksel sayisi : " << cv::countNonZero(diff) << "\n";
    std::cout << "  Sure (tek calistirma): OpenCV " << msOpenCv << " ms, elle yazilan " << msMine << " ms\n";

    // Gürültü bastırma başarısının sayısal ölçümü
    if (argc >= 4) {
        cv::Mat clean = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
        if (clean.empty() || clean.size() != noisy.size()) {
            std::cout << "\nUyari: temiz referans okunamadi ya da boyutu farkli, MSE/PSNR hesaplanmadi.\n";
        } else {
            struct Row {
                std::string name;
                const cv::Mat* img;
            };
            const std::vector<Row> rows = {{"Tuz-biber gurultulu", &noisy},
                                           {"Medyan 5x5 (OpenCV)", &viaOpenCv},
                                           {"Medyan 5x5 (elle yazilan)", &mine},
                                           {"Ortalama 5x5 (karsilastirma)", &box5}};
            std::cout << "\nTemiz referansa gore hata olcumu (MSE dusuk, PSNR yuksek = daha iyi):\n";
            for (const auto& r : rows) {
                const double mse = meanSquaredError(*r.img, clean);
                std::cout << "  " << std::left << std::setw(30) << r.name << std::right << "MSE = " << std::setw(8)
                          << mse << ", PSNR = " << psnr(mse) << " dB\n";
            }
        }
    }

    // Görseller
    auto small = [](const cv::Mat& m) {
        cv::Mat r;
        const double s = 480.0 / m.cols;
        cv::resize(m, r, cv::Size(), s, s, cv::INTER_AREA);
        return r;
    };
    cv::Mat montage;
    cv::hconcat(std::vector<cv::Mat>{small(noisy), small(viaOpenCv), small(mine)}, montage);

    // Yakın plan: ortadan 200x200 bölge 3 kat büyütülmüş (gürültülü | medyan | ortalama 5x5)
    const int cs = std::min(200, std::min(noisy.cols, noisy.rows));
    const cv::Rect roi((noisy.cols - cs) / 2, (noisy.rows - cs) / 2, cs, cs);
    auto zoom3 = [&](const cv::Mat& m) {
        cv::Mat r;
        cv::resize(m(roi), r, cv::Size(), 3.0, 3.0, cv::INTER_NEAREST);
        return r;
    };
    cv::Mat zoom;
    cv::hconcat(std::vector<cv::Mat>{zoom3(noisy), zoom3(mine), zoom3(box5)}, zoom);

    cv::imwrite((outDir / "1_tuz_biber.png").string(), noisy);
    cv::imwrite((outDir / "2_medyan_opencv_5x5.png").string(), viaOpenCv);
    cv::imwrite((outDir / "3_medyan_elle_5x5.png").string(), mine);
    cv::imwrite((outDir / "4_yan_yana_gurultulu_opencv_elle.png").string(), montage);
    cv::imwrite((outDir / "5_yakin_plan_x3_gurultulu_medyan_ortalama.png").string(), zoom);

    std::cout << "\nKaydedilenler (" << outDir.string() << "):\n"
              << "  1_tuz_biber.png                                 : girdi\n"
              << "  2_medyan_opencv_5x5.png                         : OpenCV medyan filtre sonucu\n"
              << "  3_medyan_elle_5x5.png                           : elle yazilan 5x5 medyan filtre sonucu\n"
              << "  4_yan_yana_gurultulu_opencv_elle.png            : soldan saga gurultulu | OpenCV | elle\n"
              << "  5_yakin_plan_x3_gurultulu_medyan_ortalama.png   : yakin plan: gurultulu | medyan (elle) | ortalama 5x5\n";
    return 0;
}

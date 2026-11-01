// Ödev 3 - Soru 4: Gürültülü bir görüntüde 3x3 ortalama (mean/box) filtresini KONVOLÜSYON ile
//                  gezdirerek gürültüyü bastırma.
//
// Konvolüsyon: 3x3'lük bir çekirdek (kernel) görüntü üzerinde piksel piksel gezdirilir.
// Her konumda çekirdek ile altındaki 3x3 komşuluk çarpılıp toplanır, sonuç merkez piksele yazılır.
// Ortalama filtresinde çekirdeğin 9 elemanı da 1/9'dur (komşuların ortalaması).
//
// Kenar pikselleri için görüntü dışına taşan komşular yansıtılarak (reflect101) alınır.
//
// Kullanım: ./ortalama_filtre <gurultulu_goruntu> [cikti_dizini] [temiz_referans_goruntu]
//   Temiz referans verilirse gürültü bastırmanın başarısı MSE ve PSNR ile sayısal ölçülür.
//
// Derleme:
//   g++ -O2 odev3_4_ortalama_filtre.cpp -o ortalama_filtre.exe -std=c++17 \
//       $(pkg-config --cflags --libs opencv5)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Kenar yansıtma (reflect101)
static int reflect101(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
    }
    return i;
}

// 3x3 çekirdekle konvolüsyon (hazır fonksiyon kullanmadan).
// Gerçek konvolüsyonda çekirdek 180 derece çevrilir (k[2-ky][2-kx]); ortalama çekirdeği simetrik
// olduğu için sonuç değişmez.
static cv::Mat convolve3x3(const cv::Mat& src, const float k[3][3]) {
    const int W = src.cols, H = src.rows;
    cv::Mat dst(src.size(), CV_8UC1);

    // Her x için soldaki, kendisi ve sağdaki sütun indeksleri (kenarlarda yansıtılmış)
    std::vector<int> colIdx(static_cast<size_t>(W) * 3);
    for (int x = 0; x < W; ++x)
        for (int kx = 0; kx < 3; ++kx) colIdx[static_cast<size_t>(x) * 3 + kx] = reflect101(x + kx - 1, W);

    for (int y = 0; y < H; ++y) {
        const uchar* rows[3];
        for (int ky = 0; ky < 3; ++ky) rows[ky] = src.ptr<uchar>(reflect101(y + ky - 1, H));
        uchar* d = dst.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx)
                    sum += k[2 - ky][2 - kx] * rows[ky][colIdx[static_cast<size_t>(x) * 3 + kx]];
            d[x] = cv::saturate_cast<uchar>(sum);  // en yakın tam sayıya yuvarlar, 0..255'e sınırlar
        }
    }
    return dst;
}

// Ortalama karesel hata
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
        std::cerr << "Kullanim: " << argv[0] << " <gurultulu_goruntu> [cikti_dizini] [temiz_referans_goruntu]\n";
        return 1;
    }
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_ortalama_filtre");
    fs::create_directories(outDir);

    cv::Mat noisy = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (noisy.empty()) {
        std::cerr << "Hata: goruntu okunamadi: " << argv[1] << "\n";
        return 1;
    }

    // 3x3 ortalama çekirdeği: 9 elemanın hepsi 1/9
    float kernel[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) kernel[i][j] = 1.0f / 9.0f;

    cv::Mat filtered = convolve3x3(noisy, kernel);

    std::cout << "Goruntu boyutu : " << noisy.cols << "x" << noisy.rows << " (gri seviye)\n";
    std::cout << "Cekirdek       : 3x3 ortalama (her eleman 1/9)\n";

    // Doğrulama (sadece kontrol amaçlı): OpenCV'nin blur sonucu ile karşılaştır
    cv::Mat ref;
    cv::blur(noisy, ref, cv::Size(3, 3));
    std::cout << "Dogrulama: elle yazilan konvolusyon ile cv::blur arasindaki en buyuk fark = "
              << cv::norm(filtered, ref, cv::NORM_INF) << " (0 ya da 1 olmasi normal, yuvarlama farki)\n";

    // Gürültü bastırma başarısının sayısal ölçümü (temiz referans verildiyse)
    if (argc >= 4) {
        cv::Mat clean = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
        if (clean.empty() || clean.size() != noisy.size()) {
            std::cout << "\nUyari: temiz referans okunamadi ya da boyutu farkli, MSE/PSNR hesaplanmadi.\n";
        } else {
            const double mseNoisy = meanSquaredError(noisy, clean);
            const double mseFilt = meanSquaredError(filtered, clean);
            std::cout << "\nTemiz referansa gore hata olcumu (MSE dusuk, PSNR yuksek = daha iyi):\n";
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "  Gurultulu goruntu : MSE = " << mseNoisy << ", PSNR = " << psnr(mseNoisy) << " dB\n";
            std::cout << "  3x3 ortalama sonra: MSE = " << mseFilt << ", PSNR = " << psnr(mseFilt) << " dB\n";
            std::cout << "  PSNR farki        : " << (psnr(mseFilt) - psnr(mseNoisy)) << " dB\n";
        }
    }

    // Görseller: tam görüntü yan yana ve ortadan 200x200'lük bölgenin 3 kat büyütülmüş yakın planı
    auto small = [](const cv::Mat& m) {
        cv::Mat r;
        const double s = 480.0 / m.cols;
        cv::resize(m, r, cv::Size(), s, s, cv::INTER_AREA);
        return r;
    };
    cv::Mat montage;
    cv::hconcat(std::vector<cv::Mat>{small(noisy), small(filtered)}, montage);

    const int cs = std::min(200, std::min(noisy.cols, noisy.rows));
    const cv::Rect roi((noisy.cols - cs) / 2, (noisy.rows - cs) / 2, cs, cs);
    cv::Mat zoomNoisy, zoomFilt, zoom;
    cv::resize(noisy(roi), zoomNoisy, cv::Size(), 3.0, 3.0, cv::INTER_NEAREST);
    cv::resize(filtered(roi), zoomFilt, cv::Size(), 3.0, 3.0, cv::INTER_NEAREST);
    cv::hconcat(std::vector<cv::Mat>{zoomNoisy, zoomFilt}, zoom);

    cv::imwrite((outDir / "1_gurultulu.png").string(), noisy);
    cv::imwrite((outDir / "2_ortalama_3x3.png").string(), filtered);
    cv::imwrite((outDir / "3_yan_yana.png").string(), montage);
    cv::imwrite((outDir / "4_yakin_plan_x3.png").string(), zoom);

    std::cout << "\nKaydedilenler (" << outDir.string() << "):\n"
              << "  1_gurultulu.png      : girdi\n"
              << "  2_ortalama_3x3.png   : 3x3 ortalama filtre sonucu (cikti resim)\n"
              << "  3_yan_yana.png       : soldan saga gurultulu | filtreli\n"
              << "  4_yakin_plan_x3.png  : ortadan 200x200 bolgenin 3 kat buyutulmus hali (gurultulu | filtreli)\n";
    return 0;
}

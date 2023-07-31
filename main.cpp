#include <iostream>
#include <fmt/core.h>
#include <fstream>
#include <vector>

#include <omp.h>

std::vector<int> read_image_PGM() {
    std::ifstream file("../baboon.ascii.pgm", std::ios::binary);
    if (!file) {
        fmt::println("No se pudo abrir el archivo");
        return {};
    }

    std::string line;
    std::getline(file, line);

    if (line != "P2") {
        fmt::println("Formato de archivo no soportado");
        return {};
    }

    int width, height, max_gray;
    file >> width >> height >> max_gray;

    std::vector<int> image(width * height, 0);

    for (int i = 0; i < width * height; ++i) {
        int pixel;
        file >> pixel;
        image[i] = pixel;
    }
    file.close();
    return image;
}

//--01.Serial
std::vector<int> histograma_serial(std::vector<int> &pixels) {
    std::vector<int> histograma(16, 0);
    for (int i = 0; i < pixels.size(); i++) {
        histograma[pixels[i] % histograma.size()]++;
    }
    return histograma;
}

//--02.OpenMp
std::vector<int> histograma_omp(std::vector<int> &pixels) {
    std::vector<int> histograma(16, 0);
    #pragma omp parallel default(none) shared(pixels, histograma)
    {
        int thread_id = omp_get_thread_num();
        int thread_count = omp_get_num_threads();

        std::vector<int> histograma_local(16, 0);

        for (int i = thread_id; i < pixels.size(); i += thread_count) {
            histograma_local[pixels[i] % histograma.size()]++;
        }

        #pragma omp critical
        {
            for (int i = 0; i < histograma.size(); ++i) {
                histograma[i] += histograma_local[i];
            }
        }
    }
    return histograma;
}

void print_histograma(std::vector<int> &histograma) {
    for( int i=0;i < histograma.size() ;i++) {
        fmt::println( "{:3} - {:3} {:10}", i * histograma.size(), i * histograma.size() + histograma.size() - 1, histograma[i]);
    }
}

int main(int argc, char** argv) {
    std::vector<int> imagen = read_image_PGM();

    std::vector<int> histograma = histograma_omp(imagen);

    print_histograma(histograma);

    return 0;
}


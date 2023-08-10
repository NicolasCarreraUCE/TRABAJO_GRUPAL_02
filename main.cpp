#include <iostream>
#include <fmt/core.h>
#include <fstream>
#include <vector>
#include <chrono>

#include <png.h>

#include <omp.h>
#include <mpi.h>

namespace ch=std::chrono;

std::vector<int> read_image_PGM() {
    std::ifstream file("baboon.ascii.pgm", std::ios::binary);
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

std::vector<int> read_image_PNG() {
    FILE *fp = fopen("baboon.png", "rb");

    if(!fp) throw std::runtime_error("Error al abrir el archivo.");

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png) throw std::runtime_error("Error durante la creación de la estructura de lectura PNG");

    png_infop info = png_create_info_struct(png);
    if(!info) throw std::runtime_error("Error durante la creación de la estructura de información PNG");

    if(setjmp(png_jmpbuf(png))) throw std::runtime_error("Error durante init_io de PNG");

    png_init_io(png, fp);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);

    png_read_update_info(png, info);

    png_bytep* row_pointers = new png_bytep[height];
    for(int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
    }

    png_read_image(png, row_pointers);
    fclose(fp);

    std::vector<int> pixels;
    for(int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for(int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 4]);
            pixels.push_back(px[0]);
        }
        free(row_pointers[y]);
    }
    delete[] row_pointers;
    png_destroy_read_struct(&png, &info, NULL);

    return pixels;
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

//--03.MPI
std::vector<int> histograma_mpi(std::vector<int> &pixels, int &argc, char** &argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::vector<int> histograma_local(16, 0);

    int block_size = pixels.size() / nprocs;
    // Divide la imagen en partes iguales para cada proceso
    int start = rank * block_size;
    int end = (rank == nprocs - 1) ? pixels.size() : start + block_size;

    // Cada proceso calcula el histograma para su parte de la imagen
    for (int i = start; i < end; ++i) {
        histograma_local[pixels[i] % histograma_local.size()]++;
    }

    std::vector<int> histograma(16, 0);

    // Suma los histogramas calculados por cada proceso
    MPI_Reduce(histograma_local.data(), histograma.data(), 16, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return histograma;
}

void print_histograma(std::vector<int> &histograma) {
    fmt::println("{:<10}| {:<10}", "Rango", "Frecuencia");
    fmt::println("{:-<{}}+{:-<{}}", "", 10, "", 10);
    for( int i=0;i < histograma.size() ;i++) {
        fmt::println( "{:3} - {:3} | {:10}", i * histograma.size(), i * histograma.size() + histograma.size() - 1, histograma[i]);
    }
}

/**
 * INSTALAR LIBRERIAS
 * ======================
 * sudo apt-get update
 * sudo apt-get install libopenmpi-dev -y
 * sudo apt-get install libpng-dev
 *
 * EJECUTAR
 * ======================
 * mpiexec -n 8 ./TRABAJO_GRUPAL_02
 */
int main(int argc, char** argv) {
    // Cargar Imagen
    std::vector<int> imagen_pgm = read_image_PGM();
    std::vector<int> imagen_png = read_image_PNG();

    // Histograma Serial
    // Imagen PGM
    {
        auto start = ch::high_resolution_clock::now();
        std::vector<int> histograma = histograma_serial(imagen_pgm);
        auto end = ch::high_resolution_clock::now();
        ch::duration<double, std::milli> duration = end-start;

        fmt::println("Histograma Serial (PGM), tiempo: {}ms", duration.count());
        print_histograma(histograma);
    }
    // Imagen PNG
    {
        auto start = ch::high_resolution_clock::now();
        std::vector<int> histograma = histograma_serial(imagen_png);
        auto end = ch::high_resolution_clock::now();
        ch::duration<double, std::milli> duration = end-start;

        fmt::println("Histograma Serial (PNG), tiempo: {}ms", duration.count());
        print_histograma(histograma);
    }
    fmt::println("");

    // Histograma OpenMp
    // Imagen PGM
    {
        auto start = ch::high_resolution_clock::now();
        std::vector<int> histograma = histograma_omp(imagen_pgm);
        auto end = ch::high_resolution_clock::now();
        ch::duration<double, std::milli> duration = end-start;

        fmt::println("Histograma OpenMp (PGM), tiempo: {}ms", duration.count());
        print_histograma(histograma);
    }
    // Imagen PNG
    {
        auto start = ch::high_resolution_clock::now();
        std::vector<int> histograma = histograma_omp(imagen_png);
        auto end = ch::high_resolution_clock::now();
        ch::duration<double, std::milli> duration = end-start;

        fmt::println("Histograma OpenMp (PNG), tiempo: {}ms", duration.count());
        print_histograma(histograma);
    }
    fmt::println("");

    // Histograma MPI
    // Imagen PGM
    {
        auto start = ch::high_resolution_clock::now();
        std::vector<int> histograma = histograma_mpi(imagen_pgm, argc, argv);
        auto end = ch::high_resolution_clock::now();
        ch::duration<double, std::milli> duration = end-start;

        fmt::println("Histograma MPI (PNG), tiempo: {}ms", duration.count());
        print_histograma(histograma);
    }
    // Imagen PNG
    {
        auto start = ch::high_resolution_clock::now();
        std::vector<int> histograma = histograma_mpi(imagen_png, argc, argv);
        auto end = ch::high_resolution_clock::now();
        ch::duration<double, std::milli> duration = end-start;

        fmt::println("Histograma MPI (PNG), tiempo: {}ms", duration.count());
        print_histograma(histograma);
    }

    return 0;
}


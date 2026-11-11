// gol_parallel.c — OpenMP Game of Life with Cache-Line Optimization
//
// Build: gcc gol_parallel.c -o gol_parallel -lraylib -lm -fopenmp
//
// Benchmark usage:
//   ./gol_parallel --bench 1000 1000 100 --threads 4 --repeat 5
//
// Environment variables (overridden by CLI flags):
//   COLS, ROWS, GENS, THREADS, REPEAT

#include "raylib.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// aligned memory allocation
#define ALIGNED_ALLOC(a, n) _aligned_malloc((n), (a))
#define ALIGNED_FREE(p)     _aligned_free(p)
#define RESTRICT            __restrict


#define ALIGN 64 // 64 bytes = standard CPU cache line size

static unsigned char *g_cells; // Current generation (padded, aligned)
static unsigned char *g_next; // Next generation (padded, aligned)
static int g_cols = 200; // Interior width
static int g_rows = 150; // Interior height
static int g_stride = 0; // Row length including halos AND cache padding
static unsigned int g_rand_counter = 0;

// ---------------------------------------------------------------------------
// Halo update — mirrors edges so the inner loop needs no modulo.
// This replaces expensive `%` math with simple memory copies.
// ---------------------------------------------------------------------------
static void UpdateHalo(void) {
    const int W = g_cols;
    const int H = g_rows;
    const int S = g_stride;

    // Copy top/bottom rows
    memcpy(&g_cells[0 * S + 1], &g_cells[H * S + 1], W);
    memcpy(&g_cells[(H + 1) * S + 1], &g_cells[1 * S + 1], W);

    // Copy left/right columns
    for (int y = 1; y <= H; y++) {
        g_cells[y * S + 0] = g_cells[y * S + W];
        g_cells[y * S + W + 1] = g_cells[y * S + 1];
    }

    // Copy the 4 corners
    g_cells[0 * S + 0] = g_cells[H * S + W];
    g_cells[0 * S + W + 1] = g_cells[H * S + 1];
    g_cells[(H + 1) * S + 0] = g_cells[1 * S + W];
    g_cells[(H + 1) * S + W + 1] = g_cells[1 * S + 1];
}

// ---------------------------------------------------------------------------
// Simulation core — parallelized over rows.
//
// Key Optimizations:
// 1. `schedule(static)`: Divides rows into equal chunks. Best for uniform workloads.
// 2. `RESTRICT`: Promises the compiler that `cur` and `nxt` don't overlap,
//    allowing it to auto-vectorize (SIMD) the inner loop.
// 3. No race conditions: Threads only read from `cur` and write to distinct
//    rows in `nxt`. The buffer swap happens safely outside the parallel block.
// ---------------------------------------------------------------------------
static void StepGrid(void) {
    UpdateHalo();

    unsigned char * RESTRICT cur = g_cells;
    unsigned char * RESTRICT nxt = g_next;
    const int S = g_stride;
    const int H = g_rows;
    const int W = g_cols;

#pragma omp parallel for schedule(static)
    for (int y = 1; y <= H; y++) {
        const int ym = (y - 1) * S;
        const int yc = y * S;
        const int yp = (y + 1) * S;

        for (int x = 1; x <= W; x++) {
            // Count 8 neighbors using pure pointer arithmetic
            const int n = cur[ym + x - 1] + cur[ym + x] + cur[ym + x + 1]
                          + cur[yc + x - 1] + cur[yc + x + 1]
                          + cur[yp + x - 1] + cur[yp + x] + cur[yp + x + 1];

            // Apply Conway's rules (alive: 2 or 3 neighbors; dead: 3 neighbors)
            const unsigned char a = cur[yc + x];
            nxt[yc + x] = (unsigned char) (n == 3 || (n == 2 && a));
        }
    }

    // Swap buffers safely
    g_cells = nxt;
    g_next = cur;
}

static void ClearGrid(void) {
    memset(g_cells, 0, (size_t) g_stride * (g_rows + 2));
    memset(g_next, 0, (size_t) g_stride * (g_rows + 2));
}

// ---------------------------------------------------------------------------
// Deterministic xorshift RNG
// Matches the serial version exactly, ensuring checksums can be compared.
// Parallelized so initialization doesn't become a bottleneck at huge sizes.
// ---------------------------------------------------------------------------
static void RandomizeGrid(void) {
    const unsigned int call = g_rand_counter++;
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int y = 1; y <= g_rows; y++) {
            // Seed each row uniquely based on its Y coordinate and the call count
            unsigned int seed = y * 2654435761u + call * 0x9E3779B9u + 1u;
            const int row = y * g_stride;
            for (int x = 1; x <= g_cols; x++) {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                g_cells[row + x] = ((seed & 3) == 0); // ~25% alive
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Headless benchmark
// ---------------------------------------------------------------------------
static int RunBenchmark(int generations, int repeats) {
    if (repeats < 1) repeats = 1;
    if (repeats > 32) repeats = 32;

    double times[32];

    RandomizeGrid();
    for (int i = 0; i < 10; i++) StepGrid(); // Warm up CPU caches and OpenMP threads

    for (int r = 0; r < repeats; r++) {
        if (r > 0) RandomizeGrid(); // Fresh grid for each timed run

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < generations; i++) StepGrid();
        clock_gettime(CLOCK_MONOTONIC, &t1);

        times[r] = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    }

    // Calculate statistics
    double sum = 0.0, mn = times[0], mx = times[0];
    for (int r = 0; r < repeats; r++) {
        sum += times[r];
        if (times[r] < mn) mn = times[r];
        if (times[r] > mx) mx = times[r];
    }
    const double avg = sum / repeats;
    double var = 0.0;
    for (int r = 0; r < repeats; r++) {
        const double d = times[r] - avg;
        var += d * d;
    }
    const double sd = sqrt(var / repeats);
    const double cells = (double) g_cols * (double) g_rows;

    printf("%dx%d %dgens, %d threads\n", g_cols, g_rows, generations, omp_get_max_threads());
    printf("Time:        ");
    for (int r = 0; r < repeats; r++) printf("%.4f s  ", times[r]);
    printf("\n");

    if (repeats > 1) {
        printf("Average:     %.4f s\n", avg);
        printf("Min/Max:     %.4f s  /  %.4f s\n", mn, mx);
        printf("Std dev:     %.4f s\n", sd);
    }
    printf("Gen/sec:     %.2f%s\n", generations / avg, repeats > 1 ? " (avg)" : "");
    printf("Cell upd/s:  %.2f M%s\n", (generations * cells / avg) / 1e6, repeats > 1 ? " (avg)" : "");

    // Checksum calculation verifies correctness.
    // Must use logical indices (y, x) instead of raw memory indices because
    // the parallel stride has padding, while the serial stride does not.
    unsigned long long checksum = 0;
    for (int y = 1; y <= g_rows; y++) {
        const int row = y * g_stride;
        for (int x = 1; x <= g_cols; x++) {
            const size_t idx = (size_t) (y - 1) * g_cols + (x - 1);
            checksum += g_cells[row + x] * idx;
        }
    }
    printf("Checksum:    %llu\n", checksum);
    return 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    int benchmark = 0;
    int bench_gens = 100;
    int bulk_reps = 1;
    int thread_arg = 0;
    int positional[3] = {0, 0, 0};
    int npos = 0;

    // 1. Read environment variables
    const char *env_cols = getenv("COLS");
    const char *env_rows = getenv("ROWS");
    const char *env_gens = getenv("GENS");
    const char *env_threads = getenv("THREADS");
    const char *env_repeat = getenv("REPEAT");

    if (env_cols) g_cols = atoi(env_cols);
    if (env_rows) g_rows = atoi(env_rows);
    if (env_gens) bench_gens = atoi(env_gens);
    if (env_threads) thread_arg = atoi(env_threads);
    if (env_repeat) bulk_reps = atoi(env_repeat);

    // 2. Parse command-line arguments (override env vars)
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) {
            benchmark = 1;
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            bulk_reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            thread_arg = atoi(argv[++i]);
        } else if (npos < 3) {
            positional[npos++] = atoi(argv[i]);
        }
    }
    if (bulk_reps < 1) bulk_reps = 1;
    if (bulk_reps > 32) bulk_reps = 32;

    if (thread_arg > 0) omp_set_num_threads(thread_arg);
    if (npos > 0) g_cols = positional[0];
    if (npos > 1) g_rows = positional[1];
    if (benchmark && npos > 2) bench_gens = positional[2];

    if (g_cols < 1 || g_rows < 1) {
        fprintf(stderr, "usage: %s [--bench] [cols rows [gens]] [--threads N] [--repeat N]\n", argv[0]);
        return 1;
    }

    // Pad stride to 64 bytes to prevent False Sharing between threads
    g_stride = ((g_cols + 2 + ALIGN - 1) / ALIGN) * ALIGN;

    size_t bytes = (size_t) g_stride * (g_rows + 2);
    bytes = ((bytes + ALIGN - 1) / ALIGN) * ALIGN;

    g_cells = (unsigned char *) ALIGNED_ALLOC(ALIGN, bytes);
    g_next = (unsigned char *) ALIGNED_ALLOC(ALIGN, bytes);
    if (!g_cells || !g_next) {
        fprintf(stderr, "out of memory (%zu bytes x2)\n", bytes);
        return 1;
    }
    memset(g_cells, 0, bytes);
    memset(g_next, 0, bytes);

    if (benchmark) {
        int rc = RunBenchmark(bench_gens, bulk_reps);
        ALIGNED_FREE(g_cells);
        ALIGNED_FREE(g_next);
        return rc;
    }

    // ---------------- Visual mode ----------------
    g_rand_counter = (unsigned int) time(NULL);
    RandomizeGrid();

    InitWindow(1200, 900, TextFormat("Conway's Game of Life (OpenMP, %d threads) — %dx%d", omp_get_max_threads(),
                                     g_cols, g_rows));
    SetTargetFPS(60);

    Camera2D cam = {0};
    cam.offset = (Vector2){GetScreenWidth() / 2.0f, GetScreenHeight() / 2.0f};
    cam.target = (Vector2){g_cols / 2.0f, g_rows / 2.0f};
    const float fitZoom = fminf((float) GetScreenWidth() / g_cols, (float) GetScreenHeight() / g_rows);
    cam.zoom = fitZoom * 0.95f;

    bool running = false;
    bool showGrid = true;
    float interval = 1.00f;
    float acc = 0.0f;
    int generation = 0;

    while (!WindowShouldClose()) {
        // --- Input Handling ---
        if (IsKeyPressed(KEY_SPACE)) running = !running;
        if (IsKeyPressed(KEY_R)) {
            RandomizeGrid();
            generation = 0;
        }
        if (IsKeyPressed(KEY_C)) {
            ClearGrid();
            generation = 0;
        }
        if (IsKeyPressed(KEY_G)) showGrid = !showGrid;
        if (IsKeyPressed(KEY_F)) {
            cam.target = (Vector2){g_cols / 2.0f, g_rows / 2.0f};
            cam.zoom = fitZoom * 0.95f;
        }
        if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_EQUAL)) interval = fmaxf(interval * 0.8f, 0.01f);
        if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_MINUS)) interval = fminf(interval * 1.25f, 2.0f);
        bool singleStep = IsKeyPressed(KEY_N) || (IsKeyPressed(KEY_RIGHT) && !running);

        // Camera Zoom (towards mouse cursor)
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            Vector2 mw = GetScreenToWorld2D(GetMousePosition(), cam);
            float newZoom = cam.zoom * powf(1.2f, wheel);
            newZoom = fmaxf(0.02f, fminf(newZoom, 100.0f));
            Vector2 ms = GetMousePosition();
            cam.target.x = mw.x - (ms.x - cam.offset.x) / newZoom;
            cam.target.y = mw.y - (ms.y - cam.offset.y) / newZoom;
            cam.zoom = newZoom;
        }

        // Camera Pan
        if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
            Vector2 d = GetMouseDelta();
            cam.target.x -= d.x / cam.zoom;
            cam.target.y -= d.y / cam.zoom;
        }

        // Mouse Painting
        Vector2 mw = GetScreenToWorld2D(GetMousePosition(), cam);
        int mc = (int) floorf(mw.x);
        int mr = (int) floorf(mw.y);
        if (mc >= 0 && mc < g_cols && mr >= 0 && mr < g_rows) {
            int i = (mr + 1) * g_stride + (mc + 1); // Map to padded index
            if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) g_cells[i] = 1;
            if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) g_cells[i] = 0;
        }

        // --- Logic Update ---
        if (singleStep) {
            StepGrid();
            generation++;
        } else if (running) {
            acc += GetFrameTime();
            while (acc >= interval) {
                StepGrid();
                generation++;
                acc -= interval;
            }
        } else {
            acc = 0.0f;
        }

        // --- Rendering ---
        BeginDrawing();
        ClearBackground(BLACK);
        BeginMode2D(cam);

        // Calculate visible bounds to avoid drawing millions of off-screen cells
        Vector2 tl = GetScreenToWorld2D((Vector2){0, 0}, cam);
        Vector2 br = GetScreenToWorld2D((Vector2){(float) GetScreenWidth(), (float) GetScreenHeight()}, cam);
        int x0 = (int) floorf(fmaxf(0.0f, tl.x));
        int y0 = (int) floorf(fmaxf(0.0f, tl.y));
        int x1 = (int) ceilf(fminf((float) g_cols, br.x));
        int y1 = (int) ceilf(fminf((float) g_rows, br.y));

        for (int y = y0; y < y1; y++) {
            const int row = (y + 1) * g_stride;
            for (int x = x0; x < x1; x++) {
                if (g_cells[row + (x + 1)]) DrawRectangle(x, y, 1, 1, GREEN);
            }
        }

        // Only draw grid lines if zoomed in enough to see them
        if (showGrid && cam.zoom > 6.0f) {
            Color gc = Fade(DARKGRAY, 0.35f);
            for (int x = x0; x <= x1; x++) DrawLine(x, y0, x, y1, gc);
            for (int y = y0; y <= y1; y++) DrawLine(x0, y, x1, y, gc);
        }
        EndMode2D();

        // Draw HUD
        DrawRectangle(0, 0, GetScreenWidth(), 28, Fade(BLACK, 0.7f));
        DrawText(TextFormat("%s | Gen %d | %dx%d | Zoom %.2fx | %.0f gen/s | %d thr",
                            running ? "RUNNING" : "PAUSED", generation, g_cols, g_rows,
                            cam.zoom, 1.0f / interval, omp_get_max_threads()),
                 10, 6, 18, RAYWHITE);

        DrawRectangle(0, GetScreenHeight() - 26, GetScreenWidth(), 26, Fade(BLACK, 0.7f));
        DrawText(
            "SPACE pause | N step | R random | C clear | F fit | G grid | Wheel zoom | MMB pan | LMB draw / RMB erase",
            10, GetScreenHeight() - 21, 16, LIGHTGRAY);

        EndDrawing();
    }

    ALIGNED_FREE(g_cells);
    ALIGNED_FREE(g_next);
    CloseWindow();
    return 0;
}

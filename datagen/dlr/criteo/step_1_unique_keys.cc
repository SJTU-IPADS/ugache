/* 
 * Generate unique keys of raw dataset for each slot
 * Should be run before step_2_replace_keys.out
 */

#include "iostream"
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h> 
#include <math.h>
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <time.h>
#include <string.h>
#include <unistd.h>

#define FILE_NUM 24
#define SLOT_BASE 14
#define SLOT_NUM 26
#define SLOT_MASK(s) (1 << (s - SLOT_BASE))

using namespace std;

uint64_t total_count[SLOT_NUM];

bool read_label(char * &ptr, size_t &remain_bytes) {
    char value = *ptr;
    if ((value != '1' && value != '0') || ptr[1] != '\t') assert(false);

    remain_bytes -= 2;
    ptr += 2;
    if (value == '1') return true;
    else return false;
}

uint32_t read_int(char * &ptr, size_t &remain_bytes) {
    // fill missing
    if (!remain_bytes) return 0;
    if (*ptr == '\t') {
        remain_bytes --;
        ptr ++;
        return 0;
    }

    // read hex
    char *tail = NULL;
    long res = strtol(ptr, &tail, 10);
    assert((tail - ptr) > 0);

    remain_bytes -= (1 + tail - ptr);
    ptr = (1 + tail);
    // printf("\tread int %u\n", static_cast<uint32_t>(res));
    return static_cast<uint32_t>(res);

    // get integar str size
    // size_t sz = 0;
    // while (sz < remain_bytes && ptr[sz] != '\t') sz++;
    // remain_bytes -= (sz + 1);

    // if (sz == 0) return 0; // fill missing
    // else return atoi(ptr);
}

uint32_t read_hex(char * &ptr, size_t &remain_bytes) {
     // fill missing
    if (!remain_bytes) return 0;
    if (*ptr == '\t' || *ptr == '\n') {
        remain_bytes --;
        ptr ++;
        return 0;
    }

    // read hex
    char *tail = NULL;
    long res = strtol(ptr, &tail, 16);
    assert((tail - ptr) == 8);

    remain_bytes -= 9;
    ptr = (1 + tail);
    // printf("\tread hex %u\n", static_cast<uint32_t>(res));
    return static_cast<uint32_t>(res);
}

void read_file(const char * fname, uint32_t *slot_records) {
    clock_t start = clock();

    // get file size
    struct stat statbuf;  
    stat(fname, &statbuf);  
    size_t size = statbuf.st_size;  

    // open and mmap file
    int fd = open(fname, O_RDONLY);
    assert(fd != -1);
    char *data = (char *) mmap( NULL, size ,PROT_READ, MAP_PRIVATE, fd, 0);

    // read and parse file
    char *read_ptr = data;
    size_t remain_bytes = size, line_no = 0;
    uint32_t tmp;
    while (remain_bytes > 2) {
        tmp = read_label(read_ptr, remain_bytes);
        for (int i = 1; i < 14; i++) tmp = read_int(read_ptr, remain_bytes);
        for (int i = 14; i < 40; i++) {
            tmp = read_hex(read_ptr, remain_bytes);
            slot_records[tmp] |= SLOT_MASK(i);
        }

        line_no++;
        if ((line_no % 10000000) == 0) 
        // if ((line_no % 10) == 0) 
            printf("\t %ld samples in file %s use %g s\n", line_no, fname, (double(clock() - start) / CLOCKS_PER_SEC));
    }

    // munmap and close file
    munmap(data, size);
    close(fd);
}

void statitic_count(int file_num, string* input_fnames, char* output_dir) {
    // init slot records
    size_t slot_records_sz = sizeof(int) * static_cast<size_t>(pow(2, 32));
    uint32_t *slot_records = (uint32_t *)malloc(slot_records_sz);
    memset(slot_records, 0, slot_records_sz);
    
    // read each file and record the keys inside
    for (int i = 0; i < file_num; i++) {
        printf("Read file %s...\n", input_fnames[i].c_str());
        read_file(input_fnames[i].c_str(), slot_records);
    }

    // statistic count and port results into output dir
    ofstream *ofiles[SLOT_NUM];
    for (int slot_no = 0; slot_no < SLOT_NUM; ++slot_no) {
        string fname = string(output_dir) + "slot_" + to_string(slot_no);
        ofiles[slot_no] = new ofstream(fname, ofstream::out | ofstream::trunc);
    }
    for (uint64_t i = 0; i < static_cast<size_t>(pow(2, 32)); ++i) {
        uint32_t key = static_cast<uint32_t>(i);
        uint32_t record = slot_records[key];
        if (key && key % 100000000 == 0) printf("process key %u.\n", key);
        for (int slot_no = 0; slot_no < SLOT_NUM; ++slot_no) {
            if (record & SLOT_MASK(slot_no + SLOT_BASE)) {
                total_count[slot_no]++;
                ofiles[slot_no]->write((char *)&key, sizeof(uint32_t));
            }
        }
    }
    for (int slot_no = 0; slot_no < SLOT_NUM; ++slot_no)
        ofiles[slot_no]->close();

    // free slot memory
    free(slot_records);
}

int main(int argc, char** argv) {
    // input arguments
    assert(argc > 3);
    char *input_dir = argv[1];          // the input directory, e.g. /disk1/criteo-TB/raw/
    char *output_dir = argv[2];         // the output directory, e.g. /disk1/criteo-TB/unique_keys/
    int file_num = argc - 3;            // the rest are input raw data file names
    string *input_fnames = new string[file_num];
    for (int i = 0; i < file_num; i++) input_fnames[i] = string(input_dir) + string(argv[i + 3]);
 
    // initialize global count var
    for (int i = 0; i < SLOT_NUM; i++) total_count[i] = 0;

    // process and count each file
    clock_t start = clock();
    statitic_count(1, input_fnames, output_dir);

    // print total count results
    uint64_t total_cat_num = 0;
    printf("Total category COUNT for each slot: ");
    for (int i = 0; i < SLOT_NUM; i++) {
        if (i % 10 == 0) printf("\n\t");
        printf("%ld\t", total_count[i]);
        total_cat_num += total_count[i];
    }
    printf("\nTotal category COUNT sum: %ld\n", total_cat_num);
    printf("[Step 1] Process unique key in %d files using %g s.\n", file_num, (double(clock() - start) / CLOCKS_PER_SEC));
}

#pragma once

#include <stdint.h>
#include <string>
#define ETHASH_ACCESSES			64

#define PROGPOW_LANES			32
#define PROGPOW_REGS			16
#define PROGPOW_CACHE_BYTES		(16*1024)
#define PROGPOW_CNT_MEM			ETHASH_ACCESSES
#define PROGPOW_CNT_CACHE		8
#define PROGPOW_CNT_MATH		8


class ProgPow
{
public:
	typedef enum {
		KERNEL_CUDA,
		KERNEL_CL
	} kernel_t;

	static std::string getKern(uint64_t seed, kernel_t kern);
private:
    static std::string math(std::string d, std::string a, std::string b, uint32_t r);
    static std::string merge(std::string a, std::string b, uint32_t r);

    static uint32_t fnv1a(uint32_t &h, uint32_t d);
    // KISS99 is simple, fast, and passes the TestU01 suite
    // https://en.wikipedia.org/wiki/KISS_(algorithm)
    // http://www.cse.yorku.ca/~oz/marsaglia-rng.html
    typedef struct {
        uint32_t z, w, jsr, jcong;
    } kiss99_t;
    static uint32_t kiss99(kiss99_t &st);
};

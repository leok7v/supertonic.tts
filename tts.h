#pragma once
#include <stdint.h>

struct tts;

struct tts * tts_load(const char * pack);
float * tts_speak(struct tts * tts, const char * text, const char * voice,
                  const char * lang, uint32_t seed, int32_t steps,
                  float speed, int32_t * count);
int32_t tts_rate(void);
void tts_free(float * samples);
void tts_close(struct tts * tts);

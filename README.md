# supertonic.tts

Supertonic 3, Supertone's on-device text to speech model (31 languages,
about 99M parameters, 44.1 kHz), as one file of C or one file of Swift.

    tts.c       the whole engine and its command line
    tts.h       the C API of tts.c, for an app
    tts.swift   the same engine, function for function

Each is complete on its own and needs libc only. Accelerate is an opt in
build, about ten times faster. Both files write the same WAV, byte for
byte.

The voices can be heard in
[ChatOKF](https://leok7v.github.io/ChatOKF/), a chat assistant that
runs on your own Mac or iPhone and reads its replies aloud.

## Get the model

One file, 112 MB, from
[huggingface.co/leok7v/supertonic](https://huggingface.co/leok7v/supertonic):

    HF=https://huggingface.co/leok7v/supertonic/resolve/main
    curl -L -O $HF/supertonic-q8.safetensors

It holds the weights at 8 bits, seventeen voices and the tables of the
text frontend. Two more files there hold the same model:

- `supertonic-fp32.safetensors`, 399 MB, at full precision. By ear it
  cannot be told from the 8 bit file.
- `supertonic-q4.safetensors`, 79 MB, most weights at 4 bits. It takes
  32 MB less memory and about 70 ms more per chunk of text, and it is
  the furthest of the three from full precision. Take the 8 bit file
  unless the memory matters.

Any of them goes after `--pack`.

## Build and run, C

    clang -std=c2x -O3 -ffp-contract=off -DACCELERATE -o tts tts.c \
          -framework Accelerate
    ./tts speak --pack supertonic-q8.safetensors --out hello.wav \
          --text "Hello world."
    afplay hello.wav

Without Accelerate, plain loops and libc:

    clang -std=c2x -O3 -ffp-contract=off -o tts tts.c

On Linux with glibc, clang or gcc:

    clang -std=c2x -O3 -ffp-contract=off -o tts tts.c -lm

## Build and run, Swift

    swiftc -O -D ACCELERATE -Xcc -DACCELERATE_NEW_LAPACK -o tts-swift \
           tts.swift
    ./tts-swift speak --pack supertonic-q8.safetensors --out hello.wav \
           --text "Hello world."

Without Accelerate:

    swiftc -O -o tts-swift tts.swift

## Options

    --pack FILE    the model file, always needed here
    --out FILE     the WAV to write, 44.1 kHz, 16 bit, mono
    --text T       what to say
    --voice M1     a code, F1 to F8 or M1 to M9, or a name, see below
    --lang na      the language tag, see below
    --seed 0       another seed is another reading of the same text
    --steps 8      flow steps; time is linear in them
    --speed 1.05   higher is faster speech
    --profile      print the time per kernel and shape
    --stream       speak through the streaming call, see below; the
                   line printed then says first=, the seconds to the
                   first audio

## Voices

Seventeen, in every model file. `--voice` takes the code or the name.

    F1 Sarah    F2 Lily     F3 Jessica  F4 Olivia   F5 Emily
    F6 Luna     F7 Rosie    F8 Kiki
    M1 Alex     M2 James    M3 Robert   M4 Sam      M5 Daniel
    M6 Jasper   M7 Bruno    M8 Hugo     M9 Leo

F1 to F5 and M1 to M5 are Supertone's presets, named as in their demo.
The other seven are synthetic voices carried over from
[KittenTTS](https://huggingface.co/KittenML/kitten-tts-nano-0.1)
(Apache 2.0), whose speakers are model outputs, not recordings: each
style was recovered by optimizing Supertonic's two style tensors
against clips of that speaker until a speaker verification model
agreed. No person was recorded for any of the seventeen; `Leo` is
KittenTTS's `expr-voice-5-m`. The seven are also here as
`voices/F6.json` to `M9.json`, in the format of Supertone's own
`voice_styles`, so the Python SDK can load them beside its presets.
The engine reads voices from the model file only.

## Text

Text of any length is split at sentence ends into chunks of at most 300
characters, 120 for Korean, with 0.3 s of silence between them. A
character the model has no entry for stops the run with its code point.

Supertonic 3's expression tags `<laugh>` `<breath>` `<surprise>` `<sigh>`
`<scream>` `<throatclear>` `<sad>` `<angry>` `<cough>` `<yawn>` are
chunked apart from the words around them: the model performs a tag only
as an utterance of its own and reads a tag inside a sentence aloud as a
word. A run of tags is one chunk, with or without spaces between them:
`<laugh> <laugh> <laugh>` is one longer laugh, and `<cough><cough><cough>`
sounds like someone coughing. `<laugh>` and `<cough>` are clearly audible; `<sigh>` and
`<breath>` are a breath; the rest do little. Text without tags renders
byte for byte as before.

Text is normalized to NFKD first, as the upstream SDK does, because the
model's character table knows decomposed forms only: an accented letter
is its base letter and a combining mark, a Hangul syllable is its jamo.
The tables for that are tensors in the model file (`nfkd.offsets`,
`nfkd.data`, `nfkd.class`), Unicode 15.0 for the Basic Multilingual
Plane, so the engine carries no Unicode data and needs no ICU. Above
U+FFFF nothing is decomposed: most emoji are dropped, and any other
such character stops the run.

The model reads text in the language it is written in; the tag does
not choose it. `na` is Supertonic's tag for text of unknown language
and the default here: Russian in, Russian out, no guess needed, and a
text may mix languages. The 31 codes are `ar` `bg` `cs` `da` `de` `el`
`en` `es` `et` `fi` `fr` `hi` `hr` `hu` `id` `it` `ja` `ko` `lt` `lv`
`nl` `pl` `pt` `ro` `ru` `sk` `sl` `sv` `tr` `uk` `vi`. Of them `ko`
also shortens the chunks to 120 characters. In our listening a code
changed nothing audible against `na`; the renders differ, so pick one
and keep it.

## Speed and memory

Apple M3, one thread, 8 flow steps, `tts.c`:

| build | file | text | audio | wall | RTF (1) | peak (2) | resident (3) |
|-------|------|------|------:|-----:|--------:|---------:|-------------:|
| Accelerate | q8 | "Hello world." | 1.35 s | 0.21 s | 6.4 | 9 MB | 112 MB |
| Accelerate | q8 | 284 characters | 16.8 s | 1.50 s | 11.2 | 28 MB | 131 MB |
| Accelerate | q4 | "Hello world." | 1.35 s | 0.28 s | 4.8 | 9 MB | 80 MB |
| Accelerate | q4 | 284 characters | 16.8 s | 1.56 s | 10.7 | 28 MB | 99 MB |
| plain | q8 | "Hello world." | 1.35 s | 1.96 s | 0.7 | 8 MB | 107 MB |
| plain | q8 | 284 characters | 16.8 s | 17.1 s | 1.0 | 27 MB | 127 MB |
| plain | q4 | "Hello world." | 1.35 s | 2.02 s | 0.7 | 8 MB | 75 MB |
| plain | q4 | 284 characters | 16.8 s | 17.2 s | 1.0 | 27 MB | 94 MB |

`tts.swift` is within 0.05 s of these. The 4 bit file costs its 70 ms
per chunk of text whatever the length, because narrow weights are
decoded each time they are used.

(1) RTF, the real time factor, is seconds of audio made per second of
wall clock, so higher is faster and 1.0 is exactly real time. At 11.2 a
minute of speech takes 5.4 s. Some papers print the inverse, wall clock
per second of audio, where lower is faster. A short text has a lower
RTF because the cost that does not depend on length is a larger share
of it.

(2) Peak private memory: what the process allocates, mostly the
activations of one chunk of text. It follows the chunk, not the whole
text.

(3) Resident size: (2) plus the pages of the model file that were
touched. The file is mapped read only, so the system can drop those
pages and read them again.

## In an app

`tts.swift` is meant to be lifted into an iOS or macOS target.

- Compile it with `-D NO_MAIN` to leave out the command line. The API
  is the C API in Swift:

      let tts = ttsLoad("supertonic-q8.safetensors")
      let samples: [Float] = ttsSpeak(tts, "Who are you?", "Kiki", "na",
                                      0, 8, 1.05)
      let total = ttsSpeakStream(tts, "One. Two.", "Kiki", "na", 0, 8,
                                 1.05, { part in player.enqueue(part) })
      ttsClose(tts)

  `part` is an `UnsafeBufferPointer<Float>` at `ttsRate()` per second,
  valid during the call only. `ttsSpeakStream` returns the total sample
  count, or -1 for a bad voice, language, step count or speed.
- Compile the file with `-O` in Debug too. Unoptimized it is hundreds
  of times slower.
- Its functions are top level (`fill`, `stack`, `flow`). Wrap the file
  in a namespace if the module has such names of its own.
- Keep one `TTS` for the life of the app. The model is mapped read
  only, so its pages are not private memory.

## C API

`tts.h` declares the API of `tts.c`. Compile `tts.c` with `-DNO_MAIN`
to leave out `main` and the command line; without it the same file is
the `tts` program, with the same output.

    struct tts;
    typedef void (*tts_stream)(void * that, const float * samples,
                               int32_t count);
    struct tts * tts_load(const char * pack);
    float * tts_speak(struct tts * tts, const char * text,
                      const char * voice, const char * lang,
                      uint32_t seed, int32_t steps, float speed,
                      int32_t * count);
    int32_t tts_speak_stream(struct tts * tts, const char * text,
                             const char * voice, const char * lang,
                             uint32_t seed, int32_t steps, float speed,
                             tts_stream stream, void * that);
    int32_t tts_rate(void);
    void tts_free(float * samples);
    void tts_close(struct tts * tts);

`tts_speak` returns `count` mono float samples at `tts_rate()`, 44100
per second. `voice`, `lang`, `seed`, `steps` and `speed` are the
options above, `"M1"`, `"na"`, 0, 8 and 1.05 by default, and the same
arguments give the same samples as `./tts speak`. An error ends the
process with a message, as in the command line. One `struct tts`
serves one thread at a time.

    #include "tts.h"

    struct tts * tts = tts_load("supertonic-q8.safetensors");
    int32_t count = 0;
    float * samples = tts_speak(tts, "Who are you?", "Kiki", "na", 0, 8,
                                1.05f, &count);
    /* count samples at tts_rate() per second */
    tts_free(samples);
    tts_close(tts);

    clang -std=c2x -O3 -ffp-contract=off -DACCELERATE -DNO_MAIN \
          -c tts.c -o tts.o

### Streaming, to start talking earlier

`tts_speak` returns when the whole text is spoken. `tts_speak_stream`
speaks the same text one sentence at a time and calls `stream` after
each one, so a player can start after the first sentence. Between two
sentences it calls `stream` once more with 0.3 s of silence. The calls
in order are the whole utterance; the return value is its sample count,
or -1 for a bad voice, language, step count or speed, where `tts_speak`
ends the process.

`that` is handed back to `stream` unchanged on every call. A C function
pointer carries no state of its own, and a Swift closure that captures
state cannot be passed as a C function pointer, so the state travels
in `that`: a buffer, a player, or `Unmanaged.passUnretained(self)`.
`samples` points into the engine's scratch memory and is valid only
during the call: copy it or play it, and do not call the same
`struct tts` from inside `stream`.

The model generates a chunk of text whole (the duration, then eight
flow steps over every frame, then the vocoder), so a sentence is the
smallest unit that can come out early. Each sentence is its own chunk,
with the same voice, seed, steps and speed, so the stream is not the
same samples as `tts_speak`, which puts up to 300 characters in one
chunk; it says the same words, with a pause between every two
sentences. Apple M3, q8, Accelerate, three sentences, 8.3 s of speech:
the first audio after 0.30 s against 0.65 s for all of it from
`tts_speak`, and 0.88 s for the whole stream, because the cost per
chunk comes three times. Without Accelerate the engine runs at about
real time, and the first sentence arrives after 3.6 s against 7.9 s.

    static void play(void * that, const float * samples, int32_t count) {
        /* copy count samples at tts_rate() into the player behind that */
    }

    int32_t total = tts_speak_stream(tts, "One. Two. Three.", "M1", "na",
                                     0, 8, 1.05f, play, player);

## Where it comes from

Built and tested on macOS on Apple silicon. `tts.c` also builds on
Linux with glibc, with clang or gcc and no extra flag, and writes the
same WAV there as before; `tts.swift` is Apple only.

The engine was written against the upstream ONNX SDK and is checked
stage by stage against it: 497 named taps per text agree to float32
noise, for six texts in five languages.

## License

The code in this repository is under the MIT license, see
[LICENSE](LICENSE).

The model is not. It is Supertone's, released under the BigScience Open
RAIL-M license, and the repacked files keep that license; its text is
next to them on Hugging Face. It restricts some uses: read its
Attachment A before you ship a voice. This repository holds none of the
model. The seven voice files under `voices/` were made with the model
and carry its license too, with KittenTTS (Apache 2.0) as the origin of
the speakers.

- Model: [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3)
- Repacked files: [leok7v/supertonic](https://huggingface.co/leok7v/supertonic)

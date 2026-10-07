# supertonic.tts

Supertonic 3, Supertone's on-device text to speech model (31 languages,
about 99M parameters, 44.1 kHz), as one file of C or one file of Swift.

    tts.c       the whole engine and its command line
    tts.swift   the same engine, function for function

Each is complete on its own and needs libc only. Accelerate is an opt in
build, about ten times faster. Both files write the same WAV, byte for
byte.

The voices can be heard in
[ChatOKF](https://leok7v.github.io/ChatOKF/), a chat assistant that
runs on your own Mac or iPhone and reads its replies aloud.

## Get the model

One file, 111 MB, from
[huggingface.co/leok7v/supertonic](https://huggingface.co/leok7v/supertonic):

    HF=https://huggingface.co/leok7v/supertonic/resolve/main
    curl -L -O $HF/supertonic-q8.safetensors

It holds the weights at 8 bits, the ten voices and the tables of the
text frontend. Two more files there hold the same model:

- `supertonic-fp32.safetensors`, 399 MB, at full precision. By ear it
  cannot be told from the 8 bit file.
- `supertonic-q4.safetensors`, 79 MB, most weights at 4 bits. It takes
  32 MB less memory and about 70 ms more per chunk of text, and it is
  the furthest of the three from full precision. Take the 8 bit file
  unless the memory matters.

Any of them goes after `--pack`.

    shasum -a 256 supertonic-q8.safetensors supertonic-q4.safetensors
    6a9da44654efba508e007ebb028cf6e43b606dc9b53efd8f40523b81a4b25b5a
    492c5ef03b7cde2eb20adbe002c3674b300bba8ca790afe8e535ff3309b2bb5d

## Build and run, C

    clang -std=c2x -O3 -ffp-contract=off -DACCELERATE -o tts tts.c \
          -framework Accelerate
    ./tts speak --pack supertonic-q8.safetensors --out hello.wav \
          --text "Hello world."
    afplay hello.wav

Without Accelerate, plain loops and libc:

    clang -std=c2x -O3 -ffp-contract=off -o tts tts.c

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
    --voice M1     F1 to F5, M1 to M5
    --lang en      the language of the text, see below
    --seed 0       another seed is another reading of the same text
    --steps 8      flow steps; time is linear in them
    --speed 1.05   higher is faster speech
    --profile      print the time per kernel and shape

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

Languages: `ar` `bg` `cs` `da` `de` `el` `en` `es` `et` `fi` `fr` `hi`
`hr` `hu` `id` `it` `ja` `ko` `lt` `lv` `nl` `pl` `pt` `ro` `ru` `sk`
`sl` `sv` `tr` `uk` `vi`.

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

- Delete its last line, `exit(run(CommandLine.arguments))`. `ttsOpen`
  maps the model once, `voiceOpen` picks a voice, and `narrate` turns
  text into float samples.
- Compile the file with `-O` in Debug too. Unoptimized it is hundreds
  of times slower.
- Its functions are top level (`fill`, `stack`, `flow`). Wrap the file
  in a namespace if the module has such names of its own.
- Keep one `TTS` for the life of the app. The model is mapped read
  only, so its pages are not private memory.

## Where it comes from

Built and tested on macOS on Apple silicon. Another libc needs its own
`sincosf` in place of Apple's `__sincosf`.

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
model.

- Model: [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3)
- Repacked files: [leok7v/supertonic](https://huggingface.co/leok7v/supertonic)

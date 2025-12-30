# Compare Models Report: distil-large-v3 vs nvidia/parakeet-tdt-0.6b-v2

Date: 2025-12-30

## Method
- Input files: 15 WAV files from `data/recordings` (approx 4.2-8.9 MB each).
- Server: running locally at `http://localhost:8000`.
- Requests: `/v1/audio/transcriptions` with Bearer token from `SIREN_API_KEY`.
- Models: `distil-large-v3` and `nvidia/parakeet-tdt-0.6b-v2`.
- Whisper language: `en` (omitted for Parakeet).
- Timing: wall time per request in milliseconds.
- Similarity: word-level SequenceMatcher ratio over lowercased word tokens.

## Findings
- Word-level similarity was high: avg 0.967 (min 0.946, max 0.987).
- Word count ratio (parakeet/distil) averaged 0.996.
- Parakeet was faster after warm-up: median 518 ms vs 2,090 ms for distil.
- The first Parakeet call included model load time (6,424 ms max).
- No empty outputs from either model.

## Per-file Metrics
```
file	words_distil_large_v3	words_nvidia_parakeet_tdt_0.6b_v2	word_sim	ms_distil_large_v3	ms_nvidia_parakeet_tdt_0.6b_v2
06012025-205037	331	330	0.983	1862	382
06012025-205533	429	426	0.966	2090	612
06102025-005600	613	627	0.961	2812	751
06162025-214428	445	438	0.967	2172	580
06172025-183552	383	379	0.948	1838	424
06232025-211550	809	795	0.963	3335	1039
06252025-003239	460	443	0.966	2285	518
06252025-172444	562	560	0.975	2905	823
06252025-175044	288	292	0.966	1533	350
06262025-212252	364	362	0.975	1632	366
06272025-031505	290	287	0.977	1640	357
07212025-211555	342	339	0.987	1644	354
07312025-124521	457	482	0.946	3235	6424
07312025-130012	823	797	0.951	3314	1065
09232025-215144	356	355	0.979	1920	414
```

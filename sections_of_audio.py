import numpy as np

#walla_fp = 'crowd-talking2.wav'
walla_fp = 'data/CRWDWalla_Outside talks 4 (ID 3094)_BSB.wav'
walla_fps = [(walla_fp, x) for x in np.arange(6,60,5)]

birdsong_fp = '/home/louis/vb3/datasets/powdermill/soundscape_data/Recording_1_Segment_29.mp3'
birdsong_starts = [10.0, 15.8, 39.0, 54.2, 65.0, 113.0, 160.0, 233.8]
birdsong_neg_starts = [29.0, 41.0, 60.1, 101.0, 102.0, 103.0, 226.9, 230.0, 268.0, 274.0]
birdsong_fps_starts = [(birdsong_fp,x) for x in birdsong_starts]
birdsong_neg_fps_starts = [(birdsong_fp,x) for x in birdsong_neg_starts]

orcavocs_fp1 = 'data/killer_whale.mp3'
orcavocs_starts1 = [3.8, 8.0, 9.0, 10.0, 11.8, 16.0]
#orcavocs_neg_starts1 = [5.5, 13.1, 17.5, 18.0, 19.0, 20.0, 21.0]
orcavocs_neg_starts1 = [5.5, 17.5, 18.0, 19.0, 20.0, 21.0, 22.0]
orcavocs_fps_starts1 = [(orcavocs_fp1,x) for x in orcavocs_starts1]
orcavocs_neg_fps_starts1 = [(orcavocs_fp1,x) for x in orcavocs_neg_starts1]
orcavocs_fp2 = 'data/killer_whale_2.mp3'
orcavocs_starts2 = [3.2, 10.0, 19.0, 26.0, 33.5]
orcavocs_neg_starts2 = [8.5, 17.0, 64.0, 65.0, 66.0]
orcavocs_fps_starts2 = [(orcavocs_fp2,x) for x in orcavocs_starts2]
orcavocs_neg_fps_starts2 = [(orcavocs_fp2,x) for x in orcavocs_neg_starts2]
orcavocs_fps_starts = orcavocs_fps_starts1 + orcavocs_fps_starts2
orcavocs_neg_fps_starts = orcavocs_neg_fps_starts1 + orcavocs_neg_fps_starts2

irish_s1_audio_fp1 = 'data/cv-corpus-17.0-2024-03-15/ga-IE/clips/common_voice_ga-IE_17358884.mp3'
irish_s1_audio_fp2 = 'data/cv-corpus-17.0-2024-03-15/ga-IE/clips/common_voice_ga-IE_17358885.mp3'
irish_s1_audio_fp3 = 'data/cv-corpus-17.0-2024-03-15/ga-IE/clips/common_voice_ga-IE_17358886.mp3'
irish_s1_audio_fp4 = 'data/cv-corpus-17.0-2024-03-15/ga-IE/clips/common_voice_ga-IE_17358896.mp3'
irish_s1_audio_fp5 = 'data/cv-corpus-17.0-2024-03-15/ga-IE/clips/common_voice_ga-IE_17358898.mp3'
irish_s1_fps_starts = [
(irish_s1_audio_fp1, 2.5),
(irish_s1_audio_fp1, 3.5),
(irish_s1_audio_fp2, 2.0),
(irish_s1_audio_fp2, 3.0),
(irish_s1_audio_fp3, 0.8),
(irish_s1_audio_fp4, 0.9),
(irish_s1_audio_fp4, 1.9),
(irish_s1_audio_fp4, 2.9),
(irish_s1_audio_fp5, 1.0),
(irish_s1_audio_fp5, 2.0),
(irish_s1_audio_fp5, 3.0),
]

irish_s2_audio_fp1 = 'data/cv-corpus-17.0-2024-03-15/ga-IE/clips/common_voice_ga-IE_17832707.mp3'
irish_s2_audio_fp2 = 'data/cv-corpus-17.0-2024-03-15/ga-IE/clips/common_voice_ga-IE_17832708.mp3'
irish_s2_audio_fp3 = 'data/cv-corpus-17.0-2024-03-15/ga-IE/clips/common_voice_ga-IE_17832709.mp3'
irish_s2_audio_fp4 = 'data/cv-corpus-17.0-2024-03-15/ga-IE/clips/common_voice_ga-IE_17832710.mp3'
irish_s2_audio_fp5 = 'data/cv-corpus-17.0-2024-03-15/ga-IE/clips/common_voice_ga-IE_17771012.mp3'
irish_s2_audio_fp6 = 'data/cv-corpus-17.0-2024-03-15/ga-IE/clips/common_voice_ga-IE_17771013.mp3'
irish_s2_audio_fp7 = 'data/cv-corpus-17.0-2024-03-15/ga-IE/clips/common_voice_ga-IE_17771014.mp3'
irish_s2_audio_fp8 = 'data/cv-corpus-17.0-2024-03-15/ga-IE/clips/common_voice_ga-IE_17771015.mp3'
irish_s2_fps_starts = [
(irish_s2_audio_fp1, 1.3),
(irish_s2_audio_fp2, 0.7),
(irish_s2_audio_fp3, 1.8),
(irish_s2_audio_fp3, 2.8),
(irish_s2_audio_fp4, 0.3),
(irish_s2_audio_fp5, 0.8),
(irish_s2_audio_fp6, 0.8),
(irish_s2_audio_fp7, 0.8),
(irish_s2_audio_fp8, 0.8),
]

german_s1_audio_fp1 = 'data/cv-corpus-17.0-delta-2024-03-15/de/clips/common_voice_de_39587346.mp3'
german_s1_audio_fp2 = 'data/cv-corpus-17.0-delta-2024-03-15/de/clips/common_voice_de_39587347.mp3'
german_s1_audio_fp3 = 'data/cv-corpus-17.0-delta-2024-03-15/de/clips/common_voice_de_39587348.mp3'
german_s1_audio_fp4 = 'data/cv-corpus-17.0-delta-2024-03-15/de/clips/common_voice_de_39587349.mp3'
german_s1_fps_starts = [
(german_s1_audio_fp1, 1.1),
(german_s1_audio_fp1, 2.1),
(german_s1_audio_fp2, 1.0),
(german_s1_audio_fp2, 2.0),
(german_s1_audio_fp2, 3.0),
(german_s1_audio_fp3, 1.0),
(german_s1_audio_fp3, 3.0),
(german_s1_audio_fp3, 5.0),
(german_s1_audio_fp4, 4.5),
]

german_s2_audio_fp1 = 'data/cv-corpus-17.0-delta-2024-03-15/de/clips/common_voice_de_39622005.mp3'
german_s2_audio_fp2 = 'data/cv-corpus-17.0-delta-2024-03-15/de/clips/common_voice_de_39622006.mp3'
german_s2_audio_fp3 = 'data/cv-corpus-17.0-delta-2024-03-15/de/clips/common_voice_de_39622007.mp3'
german_s2_audio_fp4 = 'data/cv-corpus-17.0-delta-2024-03-15/de/clips/common_voice_de_39622008.mp3'
german_s2_fps_starts = [
(german_s2_audio_fp1, 1.0),
(german_s2_audio_fp1, 2.0),
(german_s2_audio_fp1, 3.0),
(german_s2_audio_fp2, 1.1),
(german_s2_audio_fp2, 2.1),
(german_s2_audio_fp3, 0.8),
(german_s2_audio_fp3, 1.8),
(german_s2_audio_fp3, 2.8),
(german_s2_audio_fp4, 1.5),
(german_s2_audio_fp4, 2.5),
]


english_s1_audio_fp1 = 'data/cv-corpus-17.0-delta-2024-03-15/en/clips/common_voice_en_39586336.mp3'
english_s1_audio_fp2 = 'data/cv-corpus-17.0-delta-2024-03-15/en/clips/common_voice_en_39586337.mp3'
english_s1_audio_fp3 = 'data/cv-corpus-17.0-delta-2024-03-15/en/clips/common_voice_en_39586338.mp3'
english_s1_audio_fp4 = 'data/cv-corpus-17.0-delta-2024-03-15/en/clips/common_voice_en_39586339.mp3'
english_s1_fps_starts = [
(english_s1_audio_fp1, 0.8),
(english_s1_audio_fp1, 1.8),
(english_s1_audio_fp2, 1.0),
(english_s1_audio_fp2, 3.0),
(english_s1_audio_fp3, 1.0),
(english_s1_audio_fp3, 2.0),
(english_s1_audio_fp4, 0.8),
(english_s1_audio_fp4, 1.8),
(english_s1_audio_fp4, 2.8),
]

english_s2_audio_fp1 = 'data/cv-corpus-17.0-delta-2024-03-15/en/clips/common_voice_en_39596002.mp3'
english_s2_audio_fp2 = 'data/cv-corpus-17.0-delta-2024-03-15/en/clips/common_voice_en_39596003.mp3'
english_s2_audio_fp3 = 'data/cv-corpus-17.0-delta-2024-03-15/en/clips/common_voice_en_39596004.mp3'
english_s2_audio_fp4 = 'data/cv-corpus-17.0-delta-2024-03-15/en/clips/common_voice_en_39596005.mp3'
english_s2_fps_starts = [
(english_s2_audio_fp1, 0.9),
(english_s2_audio_fp1, 1.9),
(english_s2_audio_fp1, 2.9),
(english_s2_audio_fp2, 0.8),
(english_s2_audio_fp2, 1.8),
(english_s2_audio_fp2, 2.8),
(english_s2_audio_fp3, 0.9),
(english_s2_audio_fp3, 1.9),
(english_s2_audio_fp4, 0.4),
(english_s2_audio_fp4, 1.4),
]

rain_fp = 'data/rain-01-10s.wav'
rain_fps_starts = [(rain_fp, x) for x in np.arange(3,10,5)]

bell_fp = 'data/015828_school-bell-56309.mp3'
bell_fps_starts = [(bell_fp, 0.8+x) for x in range(1,6)]

gong_fp1 = 'data/gong-bell-129820.mp3'
gong_fp2 = 'data/gong-106628.mp3'
english_s2_audio_fp2 = 'data/cv-corpus-17.0-delta-2024-03-15/en/clips/common_voice_en_39596003.mp3'
english_s2_audio_fp3 = 'data/cv-corpus-17.0-delta-2024-03-15/en/clips/common_voice_en_39596004.mp3'
english_s2_audio_fp4 = 'data/cv-corpus-17.0-delta-2024-03-15/en/clips/common_voice_en_39596005.mp3'
gong_fps_starts = [
(gong_fp1, 0.0),
(gong_fp1, 1.0),
(gong_fp1, 2.0),
(gong_fp2, 0.5),
(gong_fp2, 1.5),
(gong_fp2, 2.5),
]

tuning_forks_fps = [
            ('data/tuning_fork_sounds/NjEzNjYzMjk1NjEzNzI3_fwfe77T5R4A.mp3', 1),
            ('data/tuning_fork_sounds/NjEzNjYzMjk1NjEzNzI3_fwfe77T5R4A.mp3', 1.5),
            ('data/tuning_fork_sounds/NjI5MzYzMjk1NjI5Mzgy_Bcv6HSBwGfU.mp3', 1),
            ('data/tuning_fork_sounds/NjI5MzYzMjk1NjI5Mzgy_Bcv6HSBwGfU.mp3', 2),
            ('data/tuning_fork_sounds/NTAwNzYzMjk1NTAwODIx_jWUQ0s6_2fGlg.mp3', 1),
            ('data/tuning_fork_sounds/NTAwNzYzMjk1NTAwODIx_jWUQ0s6_2fGlg.mp3', 2),
            ('data/tuning_fork_sounds/NTMyMjYzMjk1NTMyMjcx_i_2fZzkEewI4s.mp3', 1),
            ('data/tuning_fork_sounds/NTMyMjYzMjk1NTMyMjcx_i_2fZzkEewI4s.mp3', 2),
            ('data/tuning_fork_sounds/NTYzNjYzMjk1NTYzNjUx_u2G3GEi4DBE.mp3', 1),
            ('data/tuning_fork_sounds/NTYzNjYzMjk1NTYzNjUx_u2G3GEi4DBE.mp3', 1.5),
            ]

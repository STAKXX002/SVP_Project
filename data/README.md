# Data

Media files and trained models are not tracked in git.

Expected layout:

    data/
      reference/    reference pronunciations, one wav per word
      learner/      learner recordings and calibration takes
      verses/       training recordings for the HMM, one folder per verse
        01/         take_1.wav, take_2.wav, ...
        02/
      models/       trained verse models (written by speech.hmm_train)
      video/        the 30-minute source video

Use mono, 16 kHz wav for speech. Other sample rates are resampled on load.

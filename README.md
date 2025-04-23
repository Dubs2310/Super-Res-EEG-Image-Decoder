# EEG-Image Decoding with a Consumer-grade EEG

## THINGS-EEG [1]
This dataset contains EEG recordings from 50 subjects visually observing 1854 object concepts (22,248 images) pulled from the THINGS, a database for studying human vision.
The authors mention that within the context of visual decoding paradigms for BCIs, there is a lack of available subject data (i.e., neural response to visual stimuli) to work with.
This is especially due to the large image datasets requiring comically complex experimental setups and hours-long experimental durations.
For example, collecting data from the THINGs database while presenting a single image per second is infeasible. 
Recording the neural response of a single subject to all the images would take up to 7 hours!

Instead, what the authors do is utilize their knowledge of Rapid Serial Visual Presentation to reduce the total duration of an experiment to just within 1 hour!
The authors reference their studies [12, 13, 14, 15] where it is observed that rapid displaying of images to participants is enough for the brain to decode the vague object concepts (minus the details) present in the visual stimuli:
- They detail one of their studies [12] where participants viewed over 16000 visual object presentations at around 5-20 images/sec all in under 40 mins. This study validated their results by employing "multivariate pattern classification" and "representation similarity analysis" which revealed that the brain processed the detailed temporal dynamics of an object similar to studies that used slower presentation speeds (of about 1 img/sec).
- **NOTE: GET REFERENCES FROM THIS PAPER OF STUDIES THAT DISPLAYS IMAGES 1 IMAGE A SECOND**

The authors break their data collection down into 2 phases:
- Main: Each image is repeated only once in a trial.
- Secondary: 200 validation images are repeated 12 times each to assess data quality and to compare it to datasets acquired in the future from other modalities.

### BENEFITS AND COSTS OF THEIR EXPERIMENTAL DESIGN
The authors also ask potential utilizers of this dataset to make note of certain benefits and costs to a subject's EEG recordings, where 25000 trials take place in under an hour:
- Images had to be displayed in rapid succession at about 10 Hz, which meant that new information was presented while previous trials were still being processed by the brain.
- Images were also forward and backward masked (i.e,. time windows of brain responses were overlapped with the grayscale resting state and subsequent image presentations), which ties in with the previous point, that the data doesn't capture the full response to each image, therefore, no form of memories or emotions are given time to emerge for an image.
- **NOTE: SEE IF THIS MIGHT HAVE AN EFFECT ON RESULTS**
- An additional limitation is that since each image is only presented once, it will make image-specific analysis difficult. Users of this dataset would be unable to conduct downstream tasks in EEG classification, such as "image retrieval".
- In turn, there is the benefit that the raw EEG data from subjects maintains focus more so on object concepts rather than the detailed minor images.
- **NOTE: WE CAN CONSIDER EEG-TEXT SEMANTIC SIMILARITY & EMBEDDING ALIGNMENT BY EMPLOYING CLIP AND BLIP2 FOR TEXT LABELLING, AND CONTRASTIVE LEARNING**

### DETAILED SUBJECT SUMMARY
- Number of Subjects: 50
- Gender Distribution: 36F, 14M
- Age: 17-30 (range), 20.44 (mean), 2.72 (sd)
- Language Profiles: 26 (English native), 24 (English non-native) | 24 monolinguals, 25 bilinguals
- Handedness: Unrecorded
- Vision: Reported normal and corrected-to-normal
- Neurological or Psychiatric Disorders: None
- Number of Subjects with Bad Data: 4 (listed in participants.tsv)

### EEG HEADSET CONFIGURATION
- Device: BrainVision ActiChamp
- Number of electrodes: 64 (placed following the international 10-10 system)
- Sampling Rate: Digitized at 1000 Hz
- Resolution: 0.0488281 µV
- Reference Channel: Cz

### REFERENCES
- [1] Grootswagers, T., Zhou, I., Robinson, A.K. et al. Human EEG recordings for 1,854 concepts presented in rapid serial visual presentation streams. Sci Data 9, 3 (2022). https://doi.org/10.1038/s41597-021-01102-7
- [12] Grootswagers, T., Robinson, A. K. & Carlson, T. A. The representational dynamics of visual objects in rapid serial visual processing streams. NeuroImage 188, 668–679 (2019).
- [13] Robinson, A. K., Grootswagers, T. & Carlson, T. A. The influence of image masking on object representations during rapid serial visual presentation. NeuroImage 197, 224–231 (2019).
- [14] Grootswagers, T., Robinson, A. K., Shatek, S. M. & Carlson, T. A. Untangling featural and conceptual object representations. NeuroImage 202, 116083 (2019).
- [15] Grootswagers, T., Robinson, A. K., Shatek, S. M. & Carlson, T. A. The neural dynamics underlying prioritisation of task-relevant information. Neurons Behav. Data Anal. Theory 5, 1–17 (2021).

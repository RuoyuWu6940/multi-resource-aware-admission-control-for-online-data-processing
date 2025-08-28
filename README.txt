This is the code for CIKM'25 accepted paper Multi-resource-aware Admission Control for Online Data Processing.

Set1-Run contains files that run numerical experiments for competitive ratio validation.
Set1-1 -- Set1-4 correspond to Figures 1-(a) -- 1-(d), respectively.
Set1-Draw then use the results obtained from Set1-Run to draw the figures.

Set2-Run contains files that run trace-driven experiments comparing OMMA against benchmarks, of which the outputs contain both performance results and runtime results.
CIKMRd4-Bino+Uniform.py runs the experiment for Figures 2-(a) and 2-(d); CIKMRd4-log.py runs the experiment for Figure 2-(b); CIKMRd4-poisson.py runs the experiment for Figure 2-(c).
The performance results from Set2-Run are drawn as Figure 2 by Set2-RewardDraw.py in the folder Set2-Draw, and then presented in the paper.
The runtime results from Set2-Run are drawn as figures by Set2-RuntimeDraw.py in the folder Set2-Draw, but only referred to and discussed, not directly presented in the paper.
The runtime figures are provided in the folder Set2-RuntimeDraw.py, where Original Dist 4 ST Runtime.pdf corresponds to the experiment of Figure 2-(a), Log Norm Dist 4 ST Runtime.pdf corresponds to that of Figure 2-(b); Poisson Dist 4 ST Runtime.pdf corresponds to that of Figure 2-(c); and Uniform Dist 4 ST Runtime.pdf corresponds to that of Figure 2-(d).

File alg.py contains all the code for implementing algorithm OMMA. File benchmark.py contains all the benchmarks used in the race-driven experiments.

Hope this work could help you with yours :)
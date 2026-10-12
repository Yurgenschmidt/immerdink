The work path is main<-dev<-feature<-working branch.
When no feature is needed, you may pr directly from working branch into dev. Merging into main is not the objective of an issue unless specified. The default end goal is getting merges into dev.
Working branches should be cleaned up when closing issues. Features only get cleaned once the feature is done. Features should be related to tracking issues. When creating a feature branch, an associated draft PR into dev should be started. When creating a working branch, an associated draft PR into the feature branch should be started.
Agents cannot create issues unless specifically prompted to by users. They should provide the issues to the user for verification before creating the issue.
HPC runs are hard to acquire, so work generally should be done using formal methods and well-structured approaches.
When we do HPC runs, a single large HPC run is generally preferrred to multiple smaller ones. In order to avoid needing HPC runs, try using existing knowledge from papers, developer info, and software to develop more efficiently. Track full citations+DOI+url for reference works.
When doing local testing, tests should still try to minimize the amount of time that hardware is used. Hardware should also be shared to ensure that other performance work is not impacted, but work that is only testing capabilities does not need to concern itself with this (still respect other ongoing work if that work is performance-based).

All human-facing documentation (overviews, guides, READMEs, design explanations written for people to
read) is aimed at newcomers to immerdink who have some general scientific-computing experience: define a
project term the first time it appears, say what and why before how, and do not assume the reader
knows the codebase.

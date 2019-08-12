================================================================
                    MADAR SHARED TASK 2019
            Arabic Fine-Grained Dialect Identification

   The Fourth Workshop for Arabic Natural Language Processing
                         (WANLP 2019)

                     Test Phase - Subtask 2
                         6 May 2019

=================================================================
MADAR-Shared-Task-Subtask-2
=================================================================

Description of Data:
---------------------
The goal of Subtask 2 is to predict countries of Twitter users by
using information about tweets posted by the Twitter users. The
provided development data contain two files:

(1) MADAR-Twitter-Subtask-2.TEST.user-label.tsv
(2) MADAR-Twitter-Subtask-2.TEST.user-tweets-features.tsv

The participants have to predict labels of countries for users
in MADAR-Twitter-Subtask-2.TEST.user-label.tsv

### The User-Label file

The *.user-label.tsv file contains Twitter users and corresponding
labels of their countries. The country labels are: Algeria, 
Bahrain, Djibouti, Egypt, Iraq, Jordan, Kuwait, Lebanon, Libya, 
Mauritania, Morocco, Oman, Palestine, Qatar, Saudi_Arabia, Somalia,
Sudan, Syria, Tunisia, United_Arab_Emirates, Yemen.

### The User-Tweets-Features file

The *.user-tweets-features.tsv file contains information about 
tweets posted by users in the *.user-label.tsv file. Due to 
restrictions by Twitter, we do not provide the tweets but provide
tweet IDs and information about tweets. 

In the *.user-tweets-features.tsv file, we provide the following 
information:

- The first column (#1 User) contains username of Twitter user
from the *.user-label.tsv file.

- The second column (#2 Tweet ID) contains ID of tweet posted by
the Twitter user.

- The third column (#3 Automatically Detected Language by Twitter)
contains language of the tweet as detected by Twitter.

- The fourth column (#4 Features ALE,ALG,ALX,AMM,ASW,BAG,BAS,BEI,
BEN,CAI,DAM,DOH,FES,JED,JER,KHA,MOS,MSA,MUS,RAB,RIY,SAL,SAN,
SFX,TRI,TUN) contains comma-separated probability scores of 25
city dialects and MSA for each tweet. The probability scores
are obtained by running the best model described in Salameh
et al. (2018).  The tweets were preprocessed to exclude all
non-Arabic characters. If a tweet becomes empty after excluding
non-Arabic characters, the cell contains the value <NIL>.


### File Statistics

Below, we present more details about the contents of the 
development files.

(1) MADAR-Twitter-Subtask-2.TEST.user-label.tsv contains 500
users.

(2) MADAR-Twitter-Subtask-2.TEST.user-tweet-features.tsv
contains information about up to 100 tweets for each of the 500
profiles in the MADAR-Twitter-Subtask-2.DEV.user-label.tsv
file (a total of 49,962 tweets).

Submission Instructions:
-------------------------------------
* The data provided on Codalab for MADAR Shared Task Subtask-2 Twitter 
User Profiles and information about Tweets posted by the Twitter Users. 
The task is to predict countries of the Twitter Users.

* Each team is allowed to submit up to three (3) runs for development 
and three (3) runs for test phase. In other words, a team can test 
several methods or parameter settings and submit the three they prefer.

* Please structure your test results as follows:
    One file per submission, named <team><N>.subtask<i>.dev, 
    where:
    # <team> stands for your team name (please use only ASCII 
    # letters, digits and “-” or “_”)
    # <N> (1, 2 or 3) is the run number
    # <i> is the number of the subtask (1 or 2)

* The file content and format should be the same as the gold 
standard files provided with the sample and training data and contain 
only the dialect class that the systems predict.

* Create an archive with the submission file (<team><N>.subtask<i>.dev.zip)

Shared Task Metrics and Restrictions:
--------------------------------------
The performance of submitted systems will be evaluated on
predictions of country labels for Twitter users in
MADAR-Twitter-Subtask-2.TEST.user-label.tsv.

IMPORTANT: Participants are NOT allowed to use
MADAR-Twitter-Subtask-2.DEV.user-label.tsv and
MADAR-Twitter-Subtask-2.DEV.user-tweet-features.tsv
for training purposes. Participants must report the performance
of their best system on MADAR-Twitter-Subtask-2.DEV.user-label.tsv
in their Shared Task system description paper.

IMPORTANT: Participants can only use the text of the tweets
obtained through (MADAR-Obtain-Tweets-v.0.1.py) and the specific
information about the tweets provided in
MADAR-Twitter-Subtask-2.TRAIN.user-tweet-features.tsv.
Participants are NOT allowed to use additional tweets or outside
information any of the tweets provided in
MADAR-Twitter-Subtask-2.TRAIN.user-tweet-features.tsv to build
their systems.  Specifically -- participants should not use meta
data from Twitter about the users or the tweets, e.g., 
geo-location data.


MADAR-Twitter-Corpus-License.txt contains the license for using
this data.

=================================================================
References
=================================================================

[1] Salameh, Mohammad, Houda Bouamor, and Nizar Habash. "Fine-
Grained Arabic Dialect Identification." Proceedings of the
27th International Conference on Computational Linguistics.
Santa Fe, New Mexico, USA, 2018.
http://aclweb.org/anthology/C18-1113

[2] Bouamor, Houda, Nizar Habash, Mohammad Salameh, Wajdi
Zaghouani, Owen Rambow, Dana Abdulrahim, Ossama Obeid, Salam
Khalifa, Fadhl Eryani, Alexander Erdmann and Kemal Oflazer.
"The MADAR Arabic Dialect Corpus and Lexicon."  Proceedings
of the Eleventh International Conference on Language Resources
and Evaluation (LREC 2018). Miyazaki, Japan, 2018.
http://www.lrec-conf.org/proceedings/lrec2018/pdf/351.pdf

================================================================
Copyright (c) 2019 Carnegie Mellon University Qatar
and New York University Abu Dhabi. All rights reserved.
================================================================

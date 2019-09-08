# In Depth NBA Shots Log Analysis 2014-2015 Season
NBA shots log 2014-2015 is a publicly available data set from Kaggle.

# Overview
With massive amounts of readily available data, professional sports teams, like most businesses, rely heavily on data analytics to guide their evolution.  The NBA is no different.  The game looks very different than it did 30, 20 or even 5 years ago.  In the past the game was dominated by “Big Men” who played close to the basket.  The offense was centered around getting the ball first to the center near the basket.  The center would either take the shot or pass to an open man if double teamed.  The 3-point shot was considered a risky low percentage shot. Rather than a first option, it was considered a second or third option.
	Analytics has changed the mindset and offensive strategy of many, if not all teams.  Teams began to realize the risk was worth the reward.  The idea that 3 is always greater than 2 began to take hold.  Shooting 10 3-point shots at 40% is still better than shooting 10 2-point shots at 50%.  Now teams like the Houston Rockets base their offense on a two-shot outcome.  Shoot a 3-pointer or take the highest percentage shot within 5 feet from the basket.  Teams like the Rockets have reversed the older paradigm.  They spread the floor with four shooters beyond the 3-point arc and have one “big man” under the basket.  They look to get the ball to an open 3-point shooter first to take the shot.  If the defender under the basket comes out to help defend the 3-point shooters the ball is then passed to the one person under the basket for a dunk or a lay-up.  The mid-range shot is not completely dead.  There are still players like Chris Paul and Dirk Nowitzki who are good enough shooters form anywhere on the court that they always have the green light when they are open.  Here we will look at some of the best close-range, mid-range and long-range shooters. As well as defenders.  Good defense never goes out of style.  We will also use machine learning to classify or “predict” shots made or missed and who won the game.  

# Project Structure
- The project analysis is contained in three Jupyter notebooks
- Notebook 1: Capstone.ipynb
	- Plotly analysis of 2 and 3-point shots made and missed
	- Plotly analysis of 2 and 3-point shooting percentages by game period
	- Plotly analysis of how shot distance is affected by closest defender
	- Machine learning classification of game winner or loser
- Notebook 2: Capstone Notebook 2.ipynb
	- Overall look at scorers and defenders in the NBA using Tableau dashboard
	- In depth statistical analysis of best defenders, best close, mid and long-range shooters
	- This analysis will be visualized using Tableau
- Notebook 3: Capstone Notebook 3.ipynb
  - In depth machine learning analysis to find the best model for predicting shots made and     missed
# Number of Made and Missed 2 and 3-pointers
Images in this README will be rendered as static images. Many are interactive Plotly or Tableau visualizations within the notebooks

<img src="https://lh3.googleusercontent.com/1fYAVbCOxqjTPxZwRsCu1BMu-DCdG6I-K4QjwR1Uv2XOtaDBfm0VUZRtHGWOWZyT1eGd4f0fzAD4a3laj1JxxQ0wi5MGL2nBfXS0epBIHmtJlmCIRa71TejQ8YwFm6Ey4O_9lwwdRK1kPTYsmtFuVPtZ3gUD3uLcUEdE8w6DDd9XVO-BUp36ARG20T7aCVGj0ORBUl9h_ci8WXuSZbpnsjvoJJJPphkNqF0oFtD9gN0oyzApxoH7Juz0fF3qcy5vSV4cLgRv3nJ4YWrpnqXdVKfUiNVYcMvVcgQzsoKJM_jwrK-0epr1sDxXljJwof5Ll_yjF2eKSYfZK9StzbqM1zMWJj2Fnm6LYAJniWJx-Ncymuaez2PN63pAQLG8IlhgPz79dbmZ3nLBX0X32UenJmDq3BaWfM39GkhEqBJWdF-CNGIVqi-LI3BMhTpux5XgEnnSrPon8gZLJgbcwh--f0UE-WA2gljc-7EnlxRs0_wWudLhHHghMgS_Qb4SSc4ucTuFQHUWjvURzKP_EdKHR_2g75dDfVKNZMCVpU4XPQ2ONNMd_fdCY0Di3iBsiqlwfogldGN9O2SHuR24OIB8CqJNPjNGeopzR22J1APq8navelIpjdZ5P4Bv7GOcgcxgWgIO-gRImt2E7seXo-AkNDnhkyOaaPogX2fTrs6AcYxKZqjLib4fiq8=w700-h450-no" width="850">
- Overall Season 2-Point shooting Percentage = 49%
- Overall Season 3-Point shooting Percentage = 35%
- Overall 2014-2015 NBA shooting percentage = 45%
# 2- and 3-Point Shooting Percentages by Quarter or OT Period
<img src="https://lh3.googleusercontent.com/qZMcoeOcQRd3U5i9dmQ0P9arVd5Rr8-7fi5P9z2rZTvmDQdUhonjdfLShFov97bqno22fYIii6cyMG0IVHnRRDW5lIqcEt08y9G41rnoy7yLa7HgPN0pENIloc-VPe1qTSxN2T_FL5Hzm21RkoP6ELSfOLRya5Cd6OMEutNzMTe4yZaa9rorSJ0CEk_0duVaF3fpGkS6NHZBQQM1h3rd2Iu2rL306d70DPVl1Uakr1nyK5xXW715MTOluXLXQbDTJIjdj9D0LDM6Lp5nGVsxnenCgjvdcpBmwkSdRttskffaw8uPu0sjKayu2cbT1QjY1F4b5JCrUAj_9hf7A5d5uenk3PDvr6GJiiXlx1HOTuebrmc2X25fQpJ99hpjkx0ApXchOTJHF92EGvgoBHpedo0iMmcD03fW7bQWJMIBEtaOZrBeBHL-T2FzEJx5sSPphExqPOEw0S6BNLQ13kktLqkRmvPY7-A2Fm8zepkZLn7TwUGbZtS57rN3l-Lu3ktBGP0e-ZXZOxvbG1gbgw-auREAKm-PzIImBpQHgINM2VwOmKSo6us3zcK83pR8qzSfPiTeZIy0EHjoLwd91CyT9qBo2XbVMmUmrkWDZeQgPh7OdsPYW2uYrynJ5leZk0lQPXzGD8pStisLXxkLYzullii9fOuhnoXzfum60vNr_ntFBevb4HDMOOE=w700-h450-no" width="850">

# How Shot Distance is Affected by Closest Defender
This plot did give some insight on how the closest defender affects shot distance.  However, more interesting to me was to see the distribution of shot distances.  The three-point line is at 23.75 feet in the NBA.  You see the largest aggregation of shots close to the basket and near the three-point line.  This lends credence to my theory that these two shots ranges have become the most prevalent in the NBA.
<img src="https://lh3.googleusercontent.com/wn8z0EIXiTnXFDqXqZZTLPl7Wyw_QZ_mPp8akrP7EzNCoUA9GAQuWPVMfqUDfE5J8gjfAO65eLT_36A85lt8A83a9Fnd4oo_u20ORwW7zaJ03gt-PAtcmHacLK0rPhojz5ct44U99Ty3_Kb1GC5eB3AYOxQfNX1xQ30YqzjMwKIXhKJcJhXH9_oh5weGP5cVEr5J3hZZmVjhiNsz-tRXiv-SzO47R8c30LK1_Sgt8ygzAG8Q07axISpae8vI8KgwD1o4xlb5uOtM_rm_jkkqFtEu587YOUC7eSWuWmR3LxOF4F8jzIsZfU4sbdldsWYwy3CRqVcxtLG0PTVqgFUZz1AExCAg4NAtgs-UJ3F_OS4fRafSVc5ELVNuybEVTkl0Gz7jcCH4epVsaC1laOqJqDp1irpVqGsXC11AOH_a3BadjP_oJcsVtCoPUSCfVY8D6hhSQoHQkxGw-E8EUfOX7OmVKP3xnsLh9Bx7vu9gVMGXpr5H3XdpZVqmubu0_KYQRm0jFR_3HZQSbgVzC020bx0u4xBoofjlceSMZfsxNpxYRXM4C7c34znKSZNjB7NBnbALkLZdkmnaV6GEcEAz39JdSvm1cPO86b_MhLNH3kHqrwVhaxPxkRRdW2xfmVT1QCNU_GSL4oNLnWcKKrxwaAHdiTxe7SqpSmVSO9Kem3bF2w8l1iMl8Lo=w700-h450-no" width="850">

# Gradient Boosting Classifier to Predict Game Won
- This was more for a bit of fun.  The data within this data set contains statistics related to shooting.
- It does not contain stats that are strongly skewed toward the winning or losing team such as rebound margin, turnover margin, number of steals or number of free throws taken.
- I wanted to see if we could produce a model with better than coin-flip accuracy at classifying the winner of the game.
- The overall accuracy was close to 60%.  Not bad!  But I would guess Vegas has better predictive value than my model!
- I changed margin of victory to all positive numbers so the answer would not be given away.  Not surprisingly the two most predictive features were Home team and margin of victory.

<img src="https://lh3.googleusercontent.com/DrhLf9orB93CpCJmQLi1Ic8wnilFLiX6F336k8nuHryrzK-INYO4bBaYq6G1_7QaQUK_w_MsYgF8BdMEjorptvxBUtXryceEfKKZYlJTGtPGfvutF29oHzNnKptMv-FLU1KA52Q1eaX1Q2bD5P-gXzojoykzZl2hDhBupGFeUUiqAsk_KmqiQxTKnD0fdc9v6fv1FirAVhX7QnVofY-XGPSQFqLkrpZlUvSOtumRUNWo8MMimIg2KtUG9K8Wb-903v9LXgs9WuJJC0GhScJpwgORn1wev-N8x3BPOn4DvLwNJHOdwZAkjnqGN14JmLgsmlfh4CmZK_wUeI97kNx4e7yUUTZS2IDudIfK9Yzqk6YBCbg2vs0CuU51xmBoXALt1nIt0dYO6I5kobvRmRXiBRMD5roRw2AE053IORjhavK9LIAsExQwFsejtBrryrdHhROhx1IjYLIOryPWBqm4GDNZAIgSnN9U2Uivh4wfIRhj3klOl7Hxsswu9kFbgnn4daw3nCk82rd58mtMcia4wmJUzFMK7kruWzgsfgT3m_D4heDe3vPw6kEQIHssTlbNa2yv-ijlkx6ADqat2L8EoGQZ0-3EQQM1m9wFULldMe5oMsxmAVAd-xms2Zg7zZWwwU-ZjDlyRvWTNXAkNN0BpaNEL991GigSPeGKyAuFwZjzceMRGWTbEZE=w824-h630-no" width="850">

# Best Close-Range Shooters 

<img src="https://lh3.googleusercontent.com/4Mu-0JU1QVuksaBnrpyBHC5fQfDQciKlPpjJjnyDbqmkj9KrZMbEJwEPiwiBXbz1CiTDXH9P89lxHBFNEbwI5_uBxstXBECYlfMWxudCJjl1y-v3mWZbLGrkIM5T_cGmbiCs6p-pnwMshduldDBmeqyKrEmQyst3UIXt8gCuiT2Rw1uUkXWRE1WM-qfzv2EhPfdCzeW1fnhzDldsh7gc1G8UKkVnTB8AqbY-C4RBdo463HbvzaBfjdzP4XAeoepfhRHSg3YE7jAOxqwi_dH_4pZ5Yq5DJEUbPJy7f0PZzRH7dLbBbGBdh2h6BSosRlmZG1gwv9xyrMT2vGs0gkNKYPNnoe4zQNkR6WroEuDW3-Z1v4RZC0lvD-cw_vKIjseKLOs5r2aDBlhwjUft6A5uCnTjU57kZyMgw0zqeeAgw11nXB1QHs7uUe0CITwJCb8zOMJQqrrzJoeXOkmys9AQQgoKQWAyPINaA24TyY4pEfzU64Fl_RV-tsNDTRRSZF5XROa_t11VxWtMIFb1VsWAzEQrk80kJzLtxyg8ptVbwF3Zo-ozOu_AQMQHNeyRqQpe1QXlA0N9ITar0Ndq_AGyN9EIq6lO_VPbB57hVgBDctMhnXeb7q9huqrPAqNQFnEz8_IG913UKAqOrI1SLgyNIMrcG3ytt6lj2ZE0LiE8e6w9XhjoNPPj0C4=w1362-h849-no" width="850">

# Best Long-Range Shooters
<img src="https://lh3.googleusercontent.com/QWyuw3TWuQ3Td5Av8kJ2-rDoISI9Ct1R3-eEmXbKQvWpzbbFUJH4_mEF11eoaiL7Vr2hKPMy-IAPqBcH63iQRrTORCxPvbbmwrQXIlpwEWVgPhLhNF1Hq4oNdhTAPD1zYeAX-f0u61NzWAaENuiOTiHM-ifl9Lzfut0AD_iet_Jur15ZCym2XLRjJbKvl044iwaxUOmo7pBp7K0vBpVttQHrKF1GNY3qIRQAZQWfBTlXp5Ybfe9C55pVyFJR44pTMrAhNUydZmm0SIeZDVbz-oAiJjUN6NXzmGG77iRq2ggq4O_7Ll2N9eKYsxjqK_1KqRjzdan1qYc39nFlkj8CrBsF4C3c_HO0HmRJngaJpMeTE67HtGvFrfEhk-OZFirQoIjyXYqOKm3Grgm-3QtG64MxNzceycn718ksU-vxzjpE7J4q7-_VO3VZu6lsaEUsUWBmkQPtVvoG-jB5Gv-btwm4v4Mk8Wg9iahascB9oCPw0v66Jn7Kfz6RkINHHbhErWQkpZvjFV0a_9Hht6cQgXgtHw6PSH-CaqKv9ZnP-psfxrtUQoH6SfFCqYd3BcqR-dhEQ0EyFD-xIfa_qhYYgo43kClS0YcZIX9IwEXLdsm1Y3eKg8Fhp1qwd6kUssrnRYPhMXAfBW6w0pvcKCDy_PDXy5JJ4BpN8wfBEoQAkxza0cdX7CxPy0Q=w1367-h847-no" width="850">

# Best Mid-Range Shooters
<img src="https://lh3.googleusercontent.com/F9HsXcZRvW74KoIttbjAD0YthVE8-red-2Zm_F0UaJb5mlXUzl-dHI_m6tIsBYp2-r3txJpeUgueaC9pX27VTQaW-ikh3G5nkBm0Dk9j14eDyc_sUQStHta5bcWNo7SoJRhwllOG1-nKFhQa6Exox_7d_EYMioeoan65iS4akI6bfBBMEew_E_1wAlZQnd0zwxBr-Edb39XQXqbbyfpX00BMkH1Ncb_WDOXDuMAOPEAeqPmmXr-6OOBGSs0IDaQ6tvwDipKkokLoOmIXhboI6_aATc8NDd062dbfKWETiH2tK8Ub4jsldt4i_6NcNP1CygbdtG7Mcz_Tft3IP3eKHfsOf8EmdXwjzXMoan7eCK3zBW8Ymc0IimcBc3pfInSZRN7DdzF98pUJiXxTVraTq-UWAn_EFCE6xrg5T9R1t7xbLfO6KNZNTYaTLzjIukA0CFwcVD8Ht88k4lWpoWy4IxReOQww8KDy_XJ8i6vwoMsaatxvIzR4ub6d7GFd54PhLGHzYElcmQW4Z5NPmColBwqvR7_w0_6yCZ81ANqEIU4vbA-u9WLq8jOTSlZLPonS6QCt3D8xcwgjYctjm2ZjXOVMsJ8IAk5qab0nxtjN-txeGqwc8anz99vfF-Ce3NYEqfBs70XB4T3fqpaN2sAU85tSKZ6E6MjphNApN9R0FNqJJpRkdbgIu2s=w1378-h822-no" width="850">

# Best Defenders
Measured by percentage of shots missed while defender is guarding

<img src="https://lh3.googleusercontent.com/GES8dZJGwzfI7izVuXLbdZWw5LSKddhlUwR8TIPZ2LEE1gJE3VFh8ADHhbJiTDWHxerRvVmgPRxr8XuseQlU-dsW4Vwi1W_SqsHUVObWqqMpIxxJwPRmvCGnm2CGVCPy0_YmIDLnzPJdd5wH98EmP4E7pDtaKppJnLXgb93h4zt63NGwvItpSHZq5Xy6nqzQos0nqdcmZY8eIXDxm3DbnojksrHxs01mQ3M11g10rcjucaaR9_CRI-ACnPltDZQv5uWyQGgKI_01tXmfeSpwDl5CwIA3vYuPKALckfADA5iAHnfIwwJv3nnxEtIuE3vu6fjM3rBeVN1TxIzCDYY-hUA09d6K9RinJH_fkBbnZXaM9vcYu0p2QH91Y7yLiuMVQycxo8456BHkhkpp2wsr8MCABqeWCXjsjt0tUAfpVVL2B-4u4g8NMvUqy28I8baoKFJPl1T7cS_aTniUqUidw_S05N2yaoDkgyMNWIHDlCKB1wdzFpwPBB58diZs2ll1PWnwoIYvCsbrZkZVjnrb-QEd1hk9SEG-jlTmOmiOrmA2x4oj-5unISlGwbv6xK7RhZLTWKzSe0oiKB-ZfNRQZbJUtgNV0GW1XhmyuNr4H6Diom-7UWSfXKHrhuzUi4_bPgl6N9Z9RZWmP4Wr3TaUrk3ZDXdvsEanC7cWz-xWzpxwUyrjivLA3IE=w1408-h844-no" width="850">

# Modeling to Classify Shots Made and Shots Missed
- The correlation between target (field goals made) and the numerical features of the data set were analyzed.  Engineered features were created by combining the most positively correlated features together to form two columns (or features).  The same was done for negatively correlated features. Thus, the data set with the engineered features contained four features and the target FGM.
- Random Forest, AdaBoost and Gradient Boosting Classifier from scikit learn were used to model the data.  A model using all features and one for the engineered features were built for the Random Forest, AdaBoost Models and the Gradient Boosting Models.  A grid search to find the optimized parameters was performed.  A model using 8 features and the optimized parameters was built for the Gradient Boosting Model.
- Models using the engineered features did not fare as well as models including all features. Overfitting was not an issue for the two best models.  The two best performing models were the AdaBoost model and the Gradient Boosting Model with all features included.
- Classification report, confusion matrix and feature importance shown for top two models.
- Graph of 10-fold cross validated Accuracy, Precision and F1-Score shown for all models built

# AdaBoost Model Classification Report and Confusion Matrix

<img src="https://lh3.googleusercontent.com/jhD1c-ow8mZUJXvdfdyfFXymGX6kFJmohhgBwcuzusFQbIN0ASJD79_aVy8DFFRk5-BoGFaF75Lgy_TbXZ897VN8tcTA6nLiIw42o_oUceDHb0iiKqz6eVwS1lpllFWFi7FWg5eXQ76MAwLH9jB3e63x06LpliacNmOMqaLoMT9rFI0OnF19rgkEJGv6U-X704Xaq0ZBk_ENeX0k_Uf-860QjGyAtXV20TGes-uJP_Rb09-8TdhvIGk45u6CYcWVs29XwJ8DOd9J8U_zDUxuL89m_KZX_pf8GMZhrCuPygYWVoMxs6JHrDgttx5Jey_G2HZ9SoGuLwUZDN2uxhPu-vNH5mhSsd5zIdXB9M1cJw1PhRUGMVJcyp4Bq0QmAlSWX4O05qvUsD0wjVaaXgsBRgRs0wV8JSMFiUWMLR3v78o3yXyW3Ri21VnFFahEBp-RGREx-M29zKFd8ayZwyV2fCVJn4p5CkhH_4fABkVkj7PVe0mZ71jrRzi_nWuhy8XSc-IpSlrP2xFtnQvKxAXFQxgYda2pMnF3CPjs_x2RlbaSmJ3ivjnsfn_y0aBohvs23Yhpt84YGhqVDhRcqS9tn7aM1nU09DceZcSRABxirXAkcGSrRIzv7MfExO0n4foMct08bwHm-bJngA1xW4ED9QAQt6Sdj34aKdQamb3aveMl36aKrSqtoco=w950-h653-no" width="850">

# Feature Importance for AdaBoost Model

<img src="https://lh3.googleusercontent.com/SnqoLrgen-LNByJ0Mhh9CHr8yf9TTq9Mh2WC5aP3cP2co9HGRFJcLoEMUshC4RzEDPI4GGqKllDcSzNBnejJovAMushxP-srmUC_De8TtRzPr7xSFHUa5r54ShXbnb4nT1WAfS5jpWUXbG68bGiLKtq36lxfZcg6upC4P3Ua4PuLG-SirxSWAPPOikK3V1eMTNywfQwyFnYSgoLnecilroTbWG54-fSCbXxDxLCjMi4VdtIVB3iK1XpItbs74hO87Vz_nkMEdHnYug3EYyto09-8ZhZ9K07HGTltr1BL4ia6ZLifv1KJT2ZAdavUc0_PyVnWUOdWqa7HJFVHFjpwUL-1onOpSW8QJrLDuMzbsEkp44t6RCKOpnVxxIWR8QeJlFzZlxpSKtzchAtuGYiPdS506lKtN5gW9YXiC2y7BtkYB1179ZVQvN-_gHw6-fOf6VYaGR6C5purJupGyoOHiKU0ABDQ2ZuWD26c7zL-xlsEaMhTTP1oaMlSqPbT_4BpKbNvuqPFM1b0Pn7sgS5uXC5LiAiFWrgCJgsZkKQknUQvlYHhuaSH-RHJpmtXtwkNiLdw27Vma4fp4P04TFHdmuGLDKLmEeNibrf9umWgQX6DjNcJvDKYFFnb1wuvE1nu8F1qkoh9V5dxpJSMJehUwmtnniO66mq_7Bb62b4gSAxPqOYjCH1h0O4=w992-h371-no" width="850">

# Gradient Boosting Classifier Classification Report and Confusion Matrix

<img src="https://lh3.googleusercontent.com/DrAwaw8ycQXu-nOmr7tFgvbQW3yC3bFZ5-nCJVRjjwGdKy32mvCn3DPYnM8LT62Q2UhCv6I0JsTlEmRwecv30i44J3uH3UCtqzGPsn90FJGhWaTOgRy_14Bs-xOOdRUroj1HFkgpn2dsHE9CbLBb_5BMKcHQKuuGAtCHpvyuZvlVZ39CxCXtFQrM7NFLL0XkTQOgZhCL-0OZlgBoom6kzJjNF4HJa_nwSo6bF3hou9nqorpxEfyoeU4EmY5vJeljsHt-RUjWNXHWFNpHpfEQNKoONrY4aaPso18vFgGaq0VnW6UjIVuX1grCTHODXEh-WTOh2zE8TrUDjdojk50TDhGz-Mv8st5TlBLCyEYc3UuWDZfUfv5uKxeWUiKWCkLP7HVmusAFfyklZ4X0qdNVwhMhI93p6peny3zDj9lLpXh4qZx-6C_d7CAysIkdVCj_fEKa3DwGB4Z2TxnDJ163U7f7RRx13zsxM8fEIfUhz5ZPw2WXhoN0qSZVeUQY57CFmkpxQozzk1boN1z1XaPsgyZudDxt-rjATtHPbZKO5oiU7Kuz9tkXJ0lkObloEqhmuDoAaXqFPjLjo7eJNee6Jk9v0QVfLlz_r29smuXxC5oHdZWUgFc4bNEYJxL8VqFpQY59nqrFjO1Fg9J-FbK5HcFHCjQnkLTM7Btoc_pxM_AqJR0CLyJ4zwE=w879-h661-no" width="850">

# Gradient Boosting Classifier Feature Importance

<img src="https://lh3.googleusercontent.com/Dzk1zT6Finl50Udz1WuHVLyGD9oIB0LFVz80pOrp2eYqTL0wPJgKoK1S071nqHUibGhwucaiwxwAgBnOhOK5xAa2tUgjh92ovkqE9fDuTUt6P4ewZI7XO1elZzeQtYbNEprC3acmTsAafQFSy2iufIojQzaNfX8pentlf7dN8nehwgURbqRXYyWbz8khwHyNRwWvjg3ztw5YN-MFd06-kjWLCc-y8nwG5V9zvJKRbaqClrO1vv8hnQ-t01l-qZbqzZPWQE5p1UCpXIL6PIBurcK52ZwYk-b7tM6VHHp-egBY2V481mZR3bV63xr7Z1w29aWg8zcbJHvaFIfkZH9HKTA-Dh1le2ZEUt0q4c92Vo9BOtKjeQkaz1boYCwchh2QQaDnn1bjrPKf81CTj4LuLnbQ5Sx1jo8OSvCS87esKJT926fYHbsYrgmpahkXuKLxucrXDbN2GCSZFUSUKeWfyS0eUOgPOv6v0Ksg1bV9f8NdmYWAq8HSB7GjKfdIP10Kj8lTE5Z0FjyDIM0H0P5g0L5uWCYeNaUabTehsEzEMGCuOp9TRebQsjo0CmWAFS-hBf6WVHC84WIXyOSyYkzO64hyWGh_Bl5k2WpZqcKsbg4gqlseQUQcS1_SGAIJZccvg02mH0Cgae6NsabA6Z7IxS6cKUV9cL-AdBwrp0gTz-kitXwfF1P_5kw=w933-h364-no" width="850">

# Graph of 10-Fold Cross Validated Accuracy All Models

<img src="https://lh3.googleusercontent.com/HBwbd2VJcV2IkPGzMswIVu4DFukL-lkMSMvAEv09SRJ4_uUs5a1m9_e5Fvl8glP8s7wMUwRK5Z7T2VwLDbe_MTqyU2_KvsNanSsLeiEwgbcu_hSIgjlXoK_Cuicrg0s1EcKP2vlrXFg2dq3YvqJoVpRqQ-BKV1fx4AyF5VyYCnyzT-exx0fjyBn-9XbCG7_f7UiXBaZWVSnSdmtmWv5Ko3Xcgr0zPCQS8cD6FrR1kFszI1IwGf0znDRNekOxHtudnBgX3qdA-wa9thVEoNr4HigOFkzpSQtc_B-krvWNhikJ-olBY0ziZACylD3msBiUyappCroZwsiB_2peLGqGDwQ1khIVU6qsTQjbEgYpufmJygjq8GI7U3ZUGaKfd2x6gsCZPxHWyQ-UJ1uQ6oHUDXMhhvZu7RSejdCMRjy4yUO7yDQpd3MLnukia1sw84KtWuMEdeOsQEZyPNp7s4ECWohoj4-VHQTiEEDPn7ww9QLyUepAew16goQhqANu-eEGmcabxtXAjuCesJd2vWaizt1TqDkE6E-bregEQcpEKRLPguLYvr1PuornsV1yu7uuUO1nNE3vYnX75kFL8_BE9hz1gSgy2tzzifAgM3wg4QME9jCErUGP3tyQeBZ4ggbNYGnMsmcFCw2JZQI5VT86hWqcUOLuJvlSVYVetUYoOour-376rIVqH8E=w988-h357-no" width="850">

# Graph of 10-Fold Cross Validated Precision All Models

<img src="https://lh3.googleusercontent.com/BVNyC2TjWjEpjPmj3wFlysv1TzBQTx-qtZZe_AHb0_i-DbJI-_CZ_sF3Y_CkWIDJZg29T6ydHF11BJY0hu6m-xhqixrq6zKhCTHrQs03Umb-qduyqHOdO3DqpLH-Sf4hH5pobps1iRMtYHSETiEDivNe1h3H2rLtSqnmWFF638QAv3yvJ9H5Exo8dYxFM3Je3m8zMbVrnZ3Xj6_vzc22dWLjz8YinaszmWxjLAxqxn8J-qimXlvS-5PZlMjIvdXqPXDlP8OoPRv-kx8YpO9XyHshIcZ9FPFiMJMtVEpxlUsBCPYmuq_JFz7Pvx1-5cdy2SxncnbzZjnnEX2szcS-62hKPvL0eE7mJ1VEDradUvdz8Je-YKQwcVWTqb0w1CTZdAT78X22PcIqubic48bBI-YvQ_tKUhJAhZYWliHbGNJuVTC8CC89-SXzHoAJ2f-K6FrC0h2ysDlVTKEyPap2EAbs9ROQbiJleyLAFek0FnzGpD-Z1LRVKZEVhgI7aif3eZYgCPAzIOepPzDYSBOlq-lGLrKyGwNZibWq955ITnSG-1NJ5W8rSoBViBImPq0Bx2paWCsByGWiSt7_BKleIqxbWkWMMFY3Lh7R1jWCzwPyVCuaNy3u9ecPWs0iTIV1cBFGfPFVlEZH28_WP9ttlVefKLx8lWy4dwZggsPQB92GCK6rByN2uBU=w990-h370-no" width="850">

# Graph of 10-Fold Cross Validated F1-Score All Models

<img src="https://lh3.googleusercontent.com/up0Acbkc5LNY65vNoNl-3dkNua1qrtTAbXiZZn9fTJ_VnxYlstlCur6TpjkOi0fffPZTf5SEtqvV_KEf94PXxKEWW_xAPy3fx6MiTzkq3Z-NEUl473XOzuyrIezUEvehzGDbQOUVArDrtg9J5xSyTIwrHNiqQA9--FDB5VoI7LK2LCIP_u_qC68ZxJrrWzT7F6-4UrftaChyC1YaVJicSkkqWP5iEbDqx2SNZot6KolYrGz3rsVaz3DqmDJ05bOBOzW25MQPaScn8wLmxG3V_O2IPikwyFLBZB6XgnryWkN-_98m7PWX0dRGqt8KJ_U6IfFCsE6e9s1ftcJcyj4iyCZL-A8DvklW2ih5afMd7DTb1WlRPlf0PMyLG-bPhl6LepYLBBFKraU3NDIPlHWTs5YWy8Z4H2B4d7ts7VCJOtV9eB8L7d965mQo0KxPJMBxsgpWMJ0qR25jCXX5fLC2ZLnaoGCyzI0mZfhSJLP1TjfetXru1ca4EPWNPnrhzKU1mVsmeNdT4fRhTjcpYTLg6O7f6vEzAMPvnm_pWroxeORf1rSjRy3wUpH4EdmDTEfcHGh5QqN6DFK8vzDptafq5X_sf-zDPK-k1MTeLP2aVMj_uIU1Lp7iLd938oErSII-2lGGbo0vDL70ui55TjNTxiIya5m6ISQPnktHYv3m3MQeQWCqnL8ijtQ=w991-h359-no" width="850">

# Conclusions: What is Our Best Model?
- To answer this question let’s start with what we know from our previous analysis.
- The overall shooting percentage in the NBA for the 2014-2015 season was 45%.
- This gives us the ground truth values that we would like to improve upon with our models.
- 55% of all shots taken were missed (class 0 in our models).
- 45% of all shots taken were made (class 1 in our models).
- Looking at the graphs of the 10-fold validation metrics it is clear the gradient boosting model performs the best.  Right?
- Yes and No.  The gradient boosting model classifies or “predicts” missed shots correctly 81% of the time. A 26% increase above the actual value.  55% of all shots were missed during this season.  This gives us valuable and useful information.  With this model, and the feature importances associated with the model, we have insight into what constitutes a “bad” shot or a shot that is likely to be missed. For instance, shot distance and closest defender were important features in this model.  Passing the ball to a player who is closer to the basket and less closely defended will likely improve the chances of making the shot.
- However, the gradient boosting model only classified made shots correctly at 48%. Just 3% above the actual value of 45% of all shots made during this season.
- It was clear from initial analysis that made shots would be the more difficult class to classify.
- I intentionally included the AdaBoost model with a decision tree base estimator.  This model focuses more on the harder to classify examples.
- With the AdaBoost model we were able to correctly classify missed shots 71% of the time.  However, we improved the classification of made shots to 54%.  And feature importances shifted.  Home game and the period the shot was taken became the most important features.  Shot distance and closest defender were still important.  Having these two added dimensions gives a better idea of what constitutes a “good” shot with a better chance of going in.
- This is a case where one model outperforms the other in classifying a particular class.  We don’t have to choose.  Each model provides important information and should be used in combination.

# Future Directions
- We only have one season of data.  Data from multiple seasons would be helpful for statistical analysis as well as building more robust models.
- To improve our model for classifying game outcome we should add more statistics that are important in determining the outcome of the game.  Rebounds, free throws, steals, turnovers and points off turnovers are all stats that are generally skewed in the winner or losers favor.  A Team that has more rebounds (especially offensive rebounds), more steals, less turnovers and more free throws will be much more likely to win the game.  This information was not available in our data set.
- In this data set we were only given shot distance.  It would be helpful to have the shot coordinates.  Where the shot was taken on the floor.  A corner 3-pointer is only 20 feet from the basket and is therefore a higher percentage shot.  A 3-point shot in front of the basket is 23.75 feet and not surprisingly a lower percentage shot.  It would have been interesting to see the shot groupings by actual position on the court.






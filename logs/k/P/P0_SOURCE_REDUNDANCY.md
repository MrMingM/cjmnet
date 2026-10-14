# P0-A Source Redundancy Audit

## 一句话结论

beneficial groups 的跨 source 等价最佳占比：cls 20.28%，reg 26.09%；仅用于 N2 先验背景，不判定 K1。

每个 source ID 为当前帧内 CAV 索引；query:i 仍使用全部来源，只以 i 为 query。相同最佳 discrete6 不保证框身份逐位相同或物理机制相同。

| Scope | Beneficial | unique source | ≥2 sources | best actions 1/2/≥3 | best sources 1/2/≥3 | action mean/median/p90 | single/query/both | same-source both | cross-family only-one-source | scenes/max share |
|---|---:|---:|---:|---|---|---|---|---:|---:|---|
| train+validation/adverse pooled/cls/associated | 1189 | 72.75% | 27.25% | 762/280/147 | 865/251/73 | 1.61/1/3.00 | 905/40/244 | 229 | 103 | 41/13.37% |
| train+validation/adverse pooled/cls/background | 3711 | 67.48% | 32.52% | 2046/960/705 | 2504/858/349 | 1.89/1/4.00 | 2469/93/1149 | 1107 | 458 | 41/9.08% |
| train+validation/adverse pooled/reg/associated | 337 | 77.45% | 22.55% | 206/85/46 | 261/59/17 | 1.60/1/3.00 | 201/38/98 | 93 | 55 | 38/7.12% |
| train+validation/adverse pooled/reg/background | 586 | 79.86% | 20.14% | 373/135/78 | 468/93/25 | 1.56/1.00/3.00 | 355/60/171 | 164 | 95 | 39/5.80% |
| train+validation/clean/cls/associated | 252 | 73.41% | 26.59% | 175/58/19 | 185/60/7 | 1.40/1.00/2.00 | 212/13/27 | 22 | 10 | 31/21.03% |
| train+validation/clean/cls/background | 987 | 68.90% | 31.10% | 574/244/169 | 680/213/94 | 1.87/1/3.00 | 702/20/265 | 255 | 106 | 39/10.84% |
| train+validation/clean/reg/associated | 63 | 82.54% | 17.46% | 32/27/4 | 52/10/1 | 1.60/1/2.00 | 30/9/24 | 24 | 20 | 23/14.29% |
| train+validation/clean/reg/background | 137 | 86.86% | 13.14% | 77/48/12 | 119/14/4 | 1.58/1/2.00 | 73/10/54 | 54 | 42 | 24/7.30% |
| train+validation/fog/cls/associated | 281 | 71.53% | 28.47% | 183/72/26 | 201/66/14 | 1.49/1/2.00 | 234/11/36 | 34 | 18 | 33/17.08% |
| train+validation/fog/cls/background | 1062 | 66.57% | 33.43% | 597/295/170 | 707/279/76 | 1.81/1.00/3.00 | 750/30/282 | 270 | 110 | 41/10.08% |
| train+validation/fog/reg/associated | 73 | 80.82% | 19.18% | 49/16/8 | 59/11/3 | 1.49/1/2.80 | 47/10/16 | 16 | 10 | 22/9.59% |
| train+validation/fog/reg/background | 149 | 83.89% | 16.11% | 108/27/14 | 125/20/4 | 1.42/1/2.00 | 105/16/28 | 28 | 17 | 24/9.40% |
| train+validation/rain/cls/associated | 249 | 74.30% | 25.70% | 169/59/21 | 185/53/11 | 1.49/1/2.00 | 200/12/37 | 32 | 16 | 33/20.88% |
| train+validation/rain/cls/background | 1028 | 70.72% | 29.28% | 594/266/168 | 727/213/88 | 1.84/1.00/3.00 | 701/26/301 | 286 | 133 | 39/12.45% |
| train+validation/rain/reg/associated | 67 | 82.09% | 17.91% | 36/22/9 | 55/11/1 | 1.61/1/3.00 | 29/10/28 | 27 | 19 | 28/17.91% |
| train+validation/rain/reg/background | 145 | 77.24% | 22.76% | 75/46/24 | 112/29/4 | 1.65/1/3.00 | 67/18/60 | 57 | 37 | 29/9.66% |
| train+validation/snow/cls/associated | 659 | 72.69% | 27.31% | 410/149/100 | 479/132/48 | 1.71/1/3.00 | 471/17/171 | 163 | 69 | 41/10.02% |
| train+validation/snow/cls/background | 1621 | 66.01% | 33.99% | 855/399/367 | 1070/366/185 | 1.99/1/4.00 | 1018/37/566 | 551 | 215 | 41/6.29% |
| train+validation/snow/reg/associated | 197 | 74.62% | 25.38% | 121/47/29 | 147/37/13 | 1.64/1/3.00 | 125/18/54 | 50 | 26 | 38/6.09% |
| train+validation/snow/reg/background | 292 | 79.11% | 20.89% | 190/62/40 | 231/44/17 | 1.60/1.00/3.00 | 183/26/83 | 79 | 41 | 39/6.51% |
| train/adverse pooled/cls/associated | 760 | 68.82% | 31.18% | 463/195/102 | 523/194/43 | 1.64/1.00/3.00 | 570/30/160 | 149 | 60 | 32/19.47% |
| train/adverse pooled/cls/background | 2418 | 64.39% | 35.61% | 1247/686/485 | 1557/647/214 | 1.91/1.00/4.00 | 1554/68/796 | 765 | 310 | 32/8.40% |
| train/adverse pooled/reg/associated | 222 | 79.28% | 20.72% | 142/52/28 | 176/37/9 | 1.57/1.00/3.00 | 138/21/63 | 60 | 34 | 30/10.81% |
| train/adverse pooled/reg/background | 435 | 79.54% | 20.46% | 278/94/63 | 346/71/18 | 1.58/1/3.00 | 264/40/131 | 127 | 68 | 31/7.59% |
| train/clean/cls/associated | 149 | 61.07% | 38.93% | 84/49/16 | 91/53/5 | 1.58/1/3.00 | 118/10/21 | 18 | 7 | 23/35.57% |
| train/clean/cls/background | 630 | 64.92% | 35.08% | 334/193/103 | 409/174/47 | 1.81/1.00/3.00 | 445/15/170 | 164 | 75 | 30/9.37% |
| train/clean/reg/associated | 39 | 76.92% | 23.08% | 18/18/3 | 30/8/1 | 1.67/2/2.00 | 20/4/15 | 15 | 12 | 17/17.95% |
| train/clean/reg/background | 102 | 83.33% | 16.67% | 51/39/12 | 85/13/4 | 1.70/1.50/3.00 | 55/1/46 | 46 | 34 | 18/9.80% |
| train/fog/cls/associated | 166 | 66.27% | 33.73% | 100/52/14 | 110/50/6 | 1.52/1.00/2.00 | 136/8/22 | 20 | 10 | 26/24.10% |
| train/fog/cls/background | 652 | 61.66% | 38.34% | 334/208/110 | 402/208/42 | 1.85/1.00/3.00 | 444/22/186 | 176 | 68 | 32/9.05% |
| train/fog/reg/associated | 44 | 84.09% | 15.91% | 32/9/3 | 37/7/0 | 1.39/1.00/2.00 | 31/5/8 | 8 | 5 | 15/13.64% |
| train/fog/reg/background | 105 | 86.67% | 13.33% | 81/17/7 | 91/14/0 | 1.33/1/2.00 | 77/11/17 | 17 | 10 | 17/12.38% |
| train/rain/cls/associated | 138 | 65.22% | 34.78% | 85/37/16 | 90/40/8 | 1.61/1.00/3.00 | 111/8/19 | 17 | 5 | 26/30.43% |
| train/rain/cls/background | 619 | 68.82% | 31.18% | 346/178/95 | 426/153/40 | 1.76/1/3.00 | 424/20/175 | 168 | 80 | 30/11.15% |
| train/rain/reg/associated | 38 | 76.32% | 23.68% | 21/10/7 | 29/8/1 | 1.66/1.00/3.00 | 18/5/15 | 14 | 8 | 22/15.79% |
| train/rain/reg/background | 107 | 71.96% | 28.04% | 56/30/21 | 77/26/4 | 1.67/1/3.00 | 55/11/41 | 38 | 21 | 23/13.08% |
| train/snow/cls/associated | 456 | 70.83% | 29.17% | 278/106/72 | 323/104/29 | 1.70/1.00/3.00 | 323/14/119 | 112 | 45 | 32/14.47% |
| train/snow/cls/background | 1147 | 63.56% | 36.44% | 567/300/280 | 729/286/132 | 2.03/2/4.00 | 686/26/435 | 421 | 162 | 32/6.54% |
| train/snow/reg/associated | 140 | 78.57% | 21.43% | 89/33/18 | 110/22/8 | 1.61/1.00/3.00 | 89/11/40 | 38 | 21 | 30/8.57% |
| train/snow/reg/background | 223 | 79.82% | 20.18% | 141/47/35 | 178/31/14 | 1.65/1/3.00 | 132/18/73 | 72 | 37 | 31/8.07% |
| validation/adverse pooled/cls/associated | 429 | 79.72% | 20.28% | 299/85/45 | 342/57/30 | 1.55/1/3.00 | 335/10/84 | 80 | 43 | 9/37.06% |
| validation/adverse pooled/cls/background | 1293 | 73.24% | 26.76% | 799/274/220 | 947/211/135 | 1.86/1/3.00 | 915/25/353 | 342 | 148 | 9/26.06% |
| validation/adverse pooled/reg/associated | 115 | 73.91% | 26.09% | 64/33/18 | 85/22/8 | 1.66/1/3.00 | 63/17/35 | 33 | 21 | 8/20.87% |
| validation/adverse pooled/reg/background | 151 | 80.79% | 19.21% | 95/41/15 | 122/22/7 | 1.52/1/2.00 | 91/20/40 | 37 | 27 | 8/22.52% |
| validation/clean/cls/associated | 103 | 91.26% | 8.74% | 91/9/3 | 94/7/2 | 1.16/1/2.00 | 94/3/6 | 4 | 3 | 8/44.66% |
| validation/clean/cls/background | 357 | 75.91% | 24.09% | 240/51/66 | 271/39/47 | 1.96/1/5.00 | 257/5/95 | 91 | 31 | 9/29.97% |
| validation/clean/reg/associated | 24 | 91.67% | 8.33% | 14/9/1 | 22/2/0 | 1.50/1.00/2.00 | 10/5/9 | 9 | 8 | 6/37.50% |
| validation/clean/reg/background | 35 | 97.14% | 2.86% | 26/9/0 | 34/1/0 | 1.26/1/2.00 | 18/9/8 | 8 | 8 | 6/28.57% |
| validation/fog/cls/associated | 115 | 79.13% | 20.87% | 83/20/12 | 91/16/8 | 1.44/1/2.60 | 98/3/14 | 14 | 8 | 7/41.74% |
| validation/fog/cls/background | 410 | 74.39% | 25.61% | 263/87/60 | 305/71/34 | 1.73/1.00/3.00 | 306/8/96 | 94 | 42 | 9/26.10% |
| validation/fog/reg/associated | 29 | 75.86% | 24.14% | 17/7/5 | 22/4/3 | 1.66/1/3.00 | 16/5/8 | 8 | 5 | 7/24.14% |
| validation/fog/reg/background | 44 | 77.27% | 22.73% | 27/10/7 | 34/6/4 | 1.61/1.00/3.00 | 28/5/11 | 11 | 7 | 7/31.82% |
| validation/rain/cls/associated | 111 | 85.59% | 14.41% | 84/22/5 | 95/13/3 | 1.33/1/2.00 | 89/4/18 | 15 | 11 | 7/46.85% |
| validation/rain/cls/background | 409 | 73.59% | 26.41% | 248/88/73 | 301/60/48 | 1.96/1/4.00 | 277/6/126 | 118 | 53 | 9/31.30% |
| validation/rain/reg/associated | 29 | 89.66% | 10.34% | 15/12/2 | 26/3/0 | 1.55/1/2.00 | 11/5/13 | 13 | 11 | 6/41.38% |
| validation/rain/reg/background | 38 | 92.11% | 7.89% | 19/16/3 | 35/3/0 | 1.58/1.50/2.00 | 12/7/19 | 19 | 16 | 6/34.21% |
| validation/snow/cls/associated | 203 | 76.85% | 23.15% | 132/43/28 | 156/28/19 | 1.73/1/3.00 | 148/3/52 | 51 | 24 | 9/29.06% |
| validation/snow/cls/background | 474 | 71.94% | 28.06% | 288/99/87 | 341/80/53 | 1.88/1.00/3.00 | 332/11/131 | 130 | 53 | 9/21.52% |
| validation/snow/reg/associated | 57 | 64.91% | 35.09% | 32/14/11 | 37/15/5 | 1.72/1/3.00 | 36/7/14 | 12 | 5 | 8/21.05% |
| validation/snow/reg/background | 69 | 76.81% | 23.19% | 49/15/5 | 53/13/3 | 1.43/1/2.00 | 51/8/10 | 7 | 4 | 8/27.54% |

## adverse weather equal-weight

| Split | Task | unique-source fraction | multi-source fraction |
|---|---|---:|---:|
| train | cls | 67.44% | 32.56% |
| train | reg | 79.66% | 20.34% |
| validation | cls | 80.52% | 19.48% |
| validation | reg | 76.81% | 23.19% |
| train+validation | cls | 72.84% | 27.16% |
| train+validation | reg | 79.18% | 20.82% |

## N2 先验与 family 冗余

- cls：唯一 source 最佳占多数，N2 PASS 的先验相对增强；仍可能被随机扰动解释。 train/validation 的 multi-source fraction 为 31.18%/20.28%；只有同一 CAV 的 single+query 都最佳的 beneficial group 占比为 10.02%。
- reg：唯一 source 最佳占多数，N2 PASS 的先验相对增强；仍可能被随机扰动解释。 train/validation 的 multi-source fraction 为 20.72%/26.09%；只有同一 CAV 的 single+query 都最佳的 beneficial group 占比为 18.26%。

每 scene 计数、所有类别 scene 覆盖与集中度、action/source 完整分布见 JSON。family 的 single/query/both 为互斥三类；same-source cross-family 可与 cross-source 同时存在。`MODIFY_STABLE_SOURCE_AMBIGUOUS` 在本报告只表示缓存状态下离散优于 KEEP 且最佳集跨 CAV，尚未执行扰动稳定性检验。它的数量等于 multi-source beneficial groups。

对 N2 的先验：跨 source 等价最佳越常见，唯一正确 CAV 的主张越弱，N2 更容易获得类似结果；单 source 最佳较多时可增加 N2 PASS 的先验倾向，但随机混合仍须实测，不能提前判断 K1。

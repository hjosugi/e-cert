
# 目次

- [目次](#目次)
- [ResNet (転移学習)](#resnet-転移学習)
  - [半教師あり学習と自己教師あり学習概論](#半教師あり学習と自己教師あり学習概論)
    - [半教師あり学習](#半教師あり学習)
  - [CPU・GPU・TPUの演算方法](#cpugputpuの演算方法)
    - [SISD](#sisd)
    - [SIMD](#simd)
    - [MIMD](#mimd)
    - [CPU・GPU・TPUのメモリアーキテクチャ](#cpugputpuのメモリアーキテクチャ)
      - [CPUのメモリアーキテクチャ](#cpuのメモリアーキテクチャ)
      - [GPUのメモリアーキテクチャ](#gpuのメモリアーキテクチャ)
      - [TPUのメモリアーキテクチャ](#tpuのメモリアーキテクチャ)
- [EfficientNet](#efficientnet)
  - [EfficientNetとCNNモデルのスケールアップ\*\*](#efficientnetとcnnモデルのスケールアップ)
  - [EfficientNetの性能\*\*](#efficientnetの性能)
  - [EfficientNet の性能](#efficientnet-の性能)
  - [Compound Scaling Method（複合スケーリング手法）の詳細](#compound-scaling-method複合スケーリング手法の詳細)
      - [Depth (d)：](#depth-d)
      - [Width (w)：](#width-w)
      - [Resolution (r)：](#resolution-r)
  - [参考文献からの補足](#参考文献からの補足)
  - [AutoMLによるベースアーキテクチャ設計](#automlによるベースアーキテクチャ設計)
  - [ImageNetでの性能比較](#imagenetでの性能比較)
    - [転移学習でも高性能](#転移学習でも高性能)
    - [今後への影響と利点](#今後への影響と利点)
    - [関連分野](#関連分野)
- [Vision Transformer](#vision-transformer)
    - [画像特徴量の入力方法･･･ 画像の”トークン”系列化](#画像特徴量の入力方法-画像のトークン系列化)
    - [ViTのアーキテクチャ ･･･ Transformer Encoderの使用](#vitのアーキテクチャ--transformer-encoderの使用)
    - [事前学習とファインチューニング](#事前学習とファインチューニング)
  - [画像特徴量の入力方法](#画像特徴量の入力方法)
  - [ViTの計算過程](#vitの計算過程)
  - [事前学習とファインチューニング](#事前学習とファインチューニング-1)
    - [Vision Transformerの事前学習のデータ量と精度](#vision-transformerの事前学習のデータ量と精度)
    - [Vision Transformerのファインチューニング性能](#vision-transformerのファインチューニング性能)
  - [まとめ](#まとめ)
    - [Vision Transformerのデータ表現と入力](#vision-transformerのデータ表現と入力)
  - [Vision Transformerのアーキテクチャ](#vision-transformerのアーキテクチャ)
    - [Vision Transformerにおける事前学習(Pre-training)とファインチューニング(Fine-tuning)](#vision-transformerにおける事前学習pre-trainingとファインチューニングfine-tuning)
    - [性能とその評価](#性能とその評価)
- [物体検知とSS分解](#物体検知とss分解)
  - [鳥瞰図：広義の物体認識タスク](#鳥瞰図広義の物体認識タスク)
  - [種類](#種類)
  - [参考文献からの補足](#参考文献からの補足-1)
  - [代表的なデータセット](#代表的なデータセット)
  - [補足](#補足)
  - [IoU（Intersection over Union）](#iouintersection-over-union)
  - [補足](#補足-1)
  - [精度評価の指標](#精度評価の指標)
  - [補足](#補足-2)
  - [深層学習以降の物体検知](#深層学習以降の物体検知)
  - [SSD（Single Shot Detector）](#ssdsingle-shot-detector)
  - [特徴マップからの出力](#特徴マップからの出力)
  - [補足： SSDは2016年に提案された1段階検出器の代表例で、異なるスケールの特徴マップを使用してマルチスケール検出を実現した。その後、RetinaNet、YOLOv3/v4/v5などの改良版が登場し、精度と速度の両方で大幅な改善が見られている。](#補足-ssdは2016年に提案された1段階検出器の代表例で異なるスケールの特徴マップを使用してマルチスケール検出を実現したその後retinanetyolov3v4v5などの改良版が登場し精度と速度の両方で大幅な改善が見られている)
  - [Semantic Segmentation](#semantic-segmentation)
    - [解決手法：](#解決手法)
  - [問題](#問題)
  - [物体検出タスクにおける大規模データセットの重要性](#物体検出タスクにおける大規模データセットの重要性)
  - [解答：\[テーマ\] 代表的データセットの特徴](#解答テーマ-代表的データセットの特徴)
  - [問題](#問題-1)
  - [解答：\[テーマ\] IoUの計算](#解答テーマ-iouの計算)
  - [問題](#問題-2)
  - [解答：\[テーマ\] 評価指標計算の前提](#解答テーマ-評価指標計算の前提)
  - [問題](#問題-3)
  - [解答：\[テーマ\] SSDの弱点](#解答テーマ-ssdの弱点)
  - [問題](#問題-4)
  - [問題](#問題-5)
  - [解答：\[テーマ\] Unpoolingの基本](#解答テーマ-unpoolingの基本)
- [Mask R-CNN](#mask-r-cnn)
    - [物体認識タスクの分類](#物体認識タスクの分類)
  - [物体検知の評価指標](#物体検知の評価指標)
    - [IoU（Intersection over Union）](#iouintersection-over-union-1)
    - [Precision-Recall曲線](#precision-recall曲線)
  - [物体検出の代表的フレームワーク](#物体検出の代表的フレームワーク)
    - [2段階検出器（例：RCNN）](#2段階検出器例rcnn)
    - [1段階検出器（例：YOLO）](#1段階検出器例yolo)
  - [SSD（Single Shot Detector）](#ssdsingle-shot-detector-1)
    - [出力形式](#出力形式)
    - [対処手法](#対処手法)
  - [Semantic Segmentation](#semantic-segmentation-1)
    - [技術要素](#技術要素)
    - [その他](#その他)
- [Instance Segmentation](#instance-segmentation)
  - [概要](#概要)
  - [代表的な手法](#代表的な手法)
    - [YOLACT（You Only Look At CoefficienTs）](#yolactyou-only-look-at-coefficients)
    - [Mask R-CNN](#mask-r-cnn-1)
  - [R-CNNファミリーの進化](#r-cnnファミリーの進化)
    - [R-CNN](#r-cnn)
    - [Fast R-CNN](#fast-r-cnn)
    - [Faster R-CNN](#faster-r-cnn)
  - [YOLO（You Only Look Once）](#yoloyou-only-look-once)
  - [FCOS（Fully Convolutional One-Stage Object Detection）](#fcosfully-convolutional-one-stage-object-detection)
  - [Feature Pyramid Networks（FPN）](#feature-pyramid-networksfpn)
  - [Mask R-CNN の構造](#mask-r-cnn-の構造)
  - [RoI Pooling vs RoI Align](#roi-pooling-vs-roi-align)
- [GAN（Generative Adversarial Networks）とその拡張](#gangenerative-adversarial-networksとその拡張)
  - [Discriminatorの更新戦略](#discriminatorの更新戦略)
- [Faster-RCNN, YOLO](#faster-rcnn-yolo)
  - [セマンティックセグメンテーション](#セマンティックセグメンテーション)
  - [R-CNN（Regional CNN）](#r-cnnregional-cnn)
    - [処理の流れ](#処理の流れ)
  - [高速R-CNN（Fast R-CNN）](#高速r-cnnfast-r-cnn)
  - [Faster R-CNN](#faster-r-cnn-1)
  - [その他の発展版](#その他の発展版)
    - [メリット](#メリット)
- [FCOS](#fcos)
  - [R-CNNの課題](#r-cnnの課題)
  - [Faster-RCNN](#faster-rcnn)
    - [RPN（Region Proposal Network）の仕組み](#rpnregion-proposal-networkの仕組み)
  - [YOLO (V1)](#yolo-v1)
  - [利点と欠点](#利点と欠点)
    - [ネットワークの出力](#ネットワークの出力)
  - [Grid cellに物体が存在する場合の予測・真値Bounding Boxの信頼度スコアの2乗誤差](#grid-cellに物体が存在する場合の予測真値bounding-boxの信頼度スコアの2乗誤差)
    - [主な違い](#主な違い)
  - [まとめ](#まとめ-1)
- [Transformer](#transformer)
  - [Seq2seqとは？](#seq2seqとは)
    - [Encoder-Decoderモデルとも呼ばれる](#encoder-decoderモデルとも呼ばれる)
  - [Transformer](#transformer-1)
    - [Attention](#attention)
    - [Transformer](#transformer-2)
    - [Attension例](#attension例)
    - [計算機構](#計算機構)
- [BERT](#bert)
  - [事前学習のアプローチ](#事前学習のアプローチ)
  - [BERTのアプローチ](#bertのアプローチ)
- [GPT](#gpt)
  - [GPT-3原論文： 「Language Models are Few-Shot Learners」](#gpt-3原論文-language-models-are-few-shot-learners)
    - [参考文献](#参考文献)
  - [GPTの仕組み](#gptの仕組み)
  - [学習プロセス](#学習プロセス)
  - [GPT-3について報告されている問題点](#gpt-3について報告されている問題点)
    - [学習や運用のコストの制約](#学習や運用のコストの制約)
    - [機能の限界（人間社会の慣習や常識を認識できないことに起因）](#機能の限界人間社会の慣習や常識を認識できないことに起因)
  - [GPT-3のモデルサイズ・アーキテクチャー・計算量](#gpt-3のモデルサイズアーキテクチャー計算量)
  - [GPTの事前学習](#gptの事前学習)
  - [GPTとベースTransformerの比較](#gptとベースtransformerの比較)
- [音声認識](#音声認識)
  - [機械学習のための音声データの扱い](#機械学習のための音声データの扱い)
    - [音声データとAI (1/2)](#音声データとai-12)
    - [音声データとAI (2/2)](#音声データとai-22)
    - [そもそも、音声データとは?](#そもそも音声データとは)
      - [音が聞こえる仕組み](#音が聞こえる仕組み)
      - [音波](#音波)
      - [周波数と角周波数](#周波数と角周波数)
    - [フーリエ変換の概算(補足)](#フーリエ変換の概算補足)
  - [フーリエ変換の公式](#フーリエ変換の公式)
  - [スペクトログラムの特徴](#スペクトログラムの特徴)
- [CTC](#ctc)
  - [音声認識の概要](#音声認識の概要)
  - [基本的な音声認識処理の流れ](#基本的な音声認識処理の流れ)
    - [特徴量抽出： 音声信号を認識しやすい表現に変換する](#特徴量抽出-音声信号を認識しやすい表現に変換する)
    - [従来の音声認識モデルの構造](#従来の音声認識モデルの構造)
    - [3つの構成要素](#3つの構成要素)
      - [音響モデル](#音響モデル)
    - [発音辞書](#発音辞書)
    - [言語モデル](#言語モデル)
    - [従来手法の問題点](#従来手法の問題点)
    - [End-to-Endモデルの登場](#end-to-endモデルの登場)
    - [CTC (Connectionist Temporal Classification)](#ctc-connectionist-temporal-classification)
    - [CTCにおける重要な発明](#ctcにおける重要な発明)
    - [前提](#前提)
    - [ブランクの導入](#ブランクの導入)
    - [縮約（Contraction）の手順](#縮約contractionの手順)
    - [損失関数](#損失関数)
    - [確率計算の詳細](#確率計算の詳細)
    - [効率的な計算：前向き・後ろ向きアルゴリズム](#効率的な計算前向き後ろ向きアルゴリズム)
    - [パスの分類](#パスの分類)
    - [確率の和の法則により：](#確率の和の法則により)
- [DCGAN](#dcgan)
    - [GANについて](#ganについて)
    - [応用技術の紹介](#応用技術の紹介)
  - [2プレイヤーのミニマックスゲームとは？](#2プレイヤーのミニマックスゲームとは)
  - [GANの価値関数はバイナリークロスエントロピー](#ganの価値関数はバイナリークロスエントロピー)
  - [最適化方法](#最適化方法)
  - [なぜGeneratorは本物のようなデータを生成するのか？](#なぜgeneratorは本物のようなデータを生成するのか)
  - [ステップ1: 価値関数を最大化する$D(x)$の値は？](#ステップ1-価値関数を最大化するdxの値は)
  - [ステップ1: 価値関数を最大化する$D(x)$の値は？](#ステップ1-価値関数を最大化するdxの値は-1)
  - [ステップ2: 価値関数はいつ最小化するか？](#ステップ2-価値関数はいつ最小化するか)
  - [ステップ2: 価値関数はいつ最小化するか？](#ステップ2-価値関数はいつ最小化するか-1)
  - [GANの問題点](#ganの問題点)
  - [Wasserstein GAN (WGAN)](#wasserstein-gan-wgan)
  - [Wasserstein GAN (WGAN)](#wasserstein-gan-wgan-1)
  - [CycleGAN](#cyclegan)
  - [CycleGAN](#cyclegan-1)
- [Conditional GAN](#conditional-gan)
  - [問題1](#問題1)
  - [問題2](#問題2)
  - [*Generator*: G(Z)G(Z)](#generator-gzgz)
  - [CGANのネットワーク](#cganのネットワーク)
  - [Generator: G(Z∣Y)G(Z|Y)](#generator-gzygzy)
  - [重要なポイント](#重要なポイント)
    - [具体例（MNIST）](#具体例mnist)
- [Pix2Pix](#pix2pix)
  - [Pix2pix の概要](#pix2pix-の概要)
    - [簡単なおさらい：GAN（1/2）](#簡単なおさらいgan12)
    - [簡単なおさらい：GAN（2/2）](#簡単なおさらいgan22)
    - [Pix2pix：概要](#pix2pix概要)
    - [Pix2pix：学習データ](#pix2pix学習データ)
    - [Pix2pix：ネットワーク](#pix2pixネットワーク)
    - [Pix2pix：工夫](#pix2pix工夫)
- [A3C](#a3c)
  - [A3Cとは](#a3cとは)
    - [A3Cによる非同期（Asynchronous）学習の詳細](#a3cによる非同期asynchronous学習の詳細)
    - [並列分散エージェントで学習を行うA3Cのメリット](#並列分散エージェントで学習を行うa3cのメリット)
      - [安定化について](#安定化について)
    - [A3Cの難しいところとA2Cについて](#a3cの難しいところとa2cについて)
    - [A3Cのロス関数](#a3cのロス関数)
    - [分岐型Actor-Criticネットワーク](#分岐型actor-criticネットワーク)
    - [方策勾配法の基本](#方策勾配法の基本)
    - [θを勾配法で最適化する](#θを勾配法で最適化する)
  - [A3Cの特徴](#a3cの特徴)
    - [A3Cのアルゴリズムの特徴としては、勾配を推定する際に：](#a3cのアルゴリズムの特徴としては勾配を推定する際に)
    - [A3CのAtari 2600における性能検証](#a3cのatari-2600における性能検証)
  - [距離学習 (Metric learning)](#距離学習-metric-learning)
    - [距離学習とは](#距離学習とは)
  - [Deep Metric Learning](#deep-metric-learning)
  - [Siamese Network](#siamese-network)
  - [Triplet Network](#triplet-network)
- [MAML(メタ学習)](#mamlメタ学習)
  - [MAMLが解決したい課題](#mamlが解決したい課題)
    - [深層学習モデル開発に必要なデータ量](#深層学習モデル開発に必要なデータ量)
  - [MAMLのコンセプト](#mamlのコンセプト)
  - [MAMLの学習手順](#mamlの学習手順)
  - [MAMLの効果](#mamlの効果)
  - [MAMLの課題と対処](#mamlの課題と対処)
- [グラフ畳み込み(GCN)](#グラフ畳み込みgcn)
    - [教科書の数式とは違うように見えるのは？](#教科書の数式とは違うように見えるのは)
  - [Spatialな場合](#spatialな場合)
    - [Spatialな場合（どんな手順で？）](#spatialな場合どんな手順で)
- [Grad-CAM, LIME, SHAP](#grad-cam-lime-shap)
  - [ディープラーニングモデルの解釈性](#ディープラーニングモデルの解釈性)
    - [CAM](#cam)
    - [Grad-CAM](#grad-cam)
    - [LIME](#lime)
    - [参考文献から補足](#参考文献から補足)
    - [SHAP](#shap)
- [Docker](#docker)
  - [コンテナ型仮想化](#コンテナ型仮想化)
  - [Dockerとは](#dockerとは)
  - [Dockerの用途](#dockerの用途)
  - [基本的な Docker コマンド](#基本的な-docker-コマンド)
  - [Dockerfile の作成方法](#dockerfile-の作成方法)
  - [GPU 環境におけるDockerの使用](#gpu-環境におけるdockerの使用)
  - [コンテナオーケストレーション](#コンテナオーケストレーション)
    - [参考文献補足](#参考文献補足)

# ResNet (転移学習)
([目次に戻る](#目次))

画像識別モデルの実利用
- 効率的な学習方法
- 異なるドメインの学習結果を利用する
- ImageNetによる事前学習
- 今回の事前学習で利用するモデル
- ファインチューニング
- ハンズオン

- 効率的な学習方法
  - 教師あり学習において、目的とするタスクでの教師データが少ない場合に、別の目的で学習した学習済みモデルを再利用する転移学習
  - 異なるドメインの学習結果を利用する

- 異なるドメインの学習結果を利用する
  異なるドメインのデータで精度の高い学習済みモデルがあるとした場合・・・
  - そのモデルの構造は似たタスクでも有効ではないか?
  - 学習済みモデルを別タスクでそのまま利用できるのではないか?
  - 事前に学習した情報から始めた方が学習が効率的になるのではないか?

- ImageNetによる事前学習
  ImageNetは1400万件以上の写真のデータセット。様々なAI/MLモデルの評価基準になっており、学習済みモデルも多く公開されている。
  https://cs.stanford.edu/people/karpathy/cnnembed/

- 事前学習で利用するモデル
  ImageNetを1000分類で分類した教師データを利用。ResNetにより学習。以下はサンプル。
  事前学習で利用するモデル

- ImageNet学習済みモデルの概要 (ResNet抜粋)。
  ttps://keras.io/api/applications/

- ResNet: SkipConnection
  中間層部分
  - 深い層の積み重ねでも学習可能に
  - 勾配消失の回避
  - 勾配爆発の回避
  - 中間層の部分出力: $H(x)$
  - 残差ブロック: $H(x) = F(x) + x$
  - 学習部分: $F(x)$

- ResNet: Bottleneck構造
  - Plainアーキテクチャ
  - Bottleneckアーキテクチャ
    - 同一計算コストで1層多い構造
    - 途中の層で3x3の畳込みを行う

- WideResnet: 構造
  -  ResNetにおけるフィルタ数をK倍
  -  畳込みチャンネル数が増加
  -  高速･高精度の学習が可能に
  -  GPUの特性に合った動作
  -  ResNetに比べ層数を浅くした
  -  DropoutをResidualブロックに導入

- 事前学習で利用するモデル
  Wide ResNet。
  -  フィルタ数をk倍したResNet。
  -  パラメータを増やす方法として、層を深くするのではなく、各層を広く(Wide)した。
  -  ハンズオンではResNet-50とResNet-50 x 3(k=3のWide ResNet)の実装例を解説。

- ファインチューニング
  ImageNetでの事前学習モデル (1000分類)

- 対象タスク
  今回はtf_flowers
  分類モデルの作成
  有効モデルの採用
  事前学習結果をそのまま学習
  事前学習結果を初期値として再学習

- ハンズオン
  Google Colaboratoryによるハンズオン。
  -  transfer-learning.ipynb
  -  wide-resnet.ipynb

- 半教師あり学習と自己教師あり学習
 深層学習モデルの学習には大量のラベルデータが付与されたデータが必要です。ただ、ラベルが付与されたデータを大量に収集したり、アノテーションするにはコストが非常にかかってしまいます。
$\Rightarrow$上記理由が、深層学習モデルの普及が進まない一因です。半教師あり学習と自己教師あり学習はラベルが少ない場合やラベルがない場合でも、深層学習モデルを構築することができ、この分野の研究が進むことで、深層学習モデルの普及がより進みます。
- 半教師あり学習
  教師あり学習と教師なし学習の中間的な学習方法です。学習モデルを構築する際にラベル付きデータとラベルなしデータ両方を活用することが特徴です。

- 自己教師あり学習
  人の手によるアノテーション作業を排除した学習方法です。学習モデルを構築する際にラベルデータを必要とせず、機械自身が教師データを作成しながら学習していくことが特徴です。

## 半教師あり学習と自己教師あり学習概論

### 半教師あり学習
- 半教師あり学習は、ラベル付きデータとラベルなしデータを活用して学習モデルを構築
- 学習の流れはまずラベル有のデータを使ってモデルを構築、構築したモデルを使ってラベルなしデータを予測、閾値以上の信頼度であれば予測されたデータを使って再度学習し、この流れを繰り返していく
- ラベル付きデータが少ない場合やラベルが偏ってしまっている場合に活用できます。

- 一致正則化
  一致正則化は同一のデータであれば、拡張されたデータも拡張されていないデータも、モデルの予測が一致するように学習させる手法です。
  参考文献： Regularization With Stochastic Transformations and Perturbations for Deep Semi-Supervised Learning
- 疑似ラベル
  疑似ラベルとは、ラベル有データで作成した学習モデルが、ラベルなしデータで予測したラベルを真のラベルとして扱う手法です。
  参考文献： Pseudo-Label : The Simple and Efficient Semi-Supervised LearningMethod for Deep Neural Networks
- エントロピー最小化
  エントロピー最小化とは、ラベル付けされていないデータのクラス確立を最小化することで、クラス間の低密度分離を促進します。簡単に言うとラベル付けされていないデータのエントロピーを最小化することで、汎化性能を高める手法です。
  参考文献：Pseudo-Label : The Simple and Efficient Semi-Supervised LearningMethod for Deep Neural Networks

- 半教師あり学習における3つの主要な手法

- Self-Training
  - ラベル付きデータとラベルなしデータを使用して学習します。大まかな流れとしては、ラベル付きデータを使ってモデルを構築、構築したモデルを使ってラベルなしデータを予測し疑似ラベルを付与、一定の閾値以上のデータをラベル付きデータとして扱い再度学習。このサイクルを繰り返し学習モデルを構築します。

- Self-TrainingとCo-Training
ラベル付きデータ → モデル作成 → ラベルなしデータ予測 → 信頼度が閾値以上のデータをラベル付きデータへ追加

- Co-Training
  - 複数の異なる分類器を使用して学習します。
  大まかな流れとしては、Self-Trainingと同様、ラベル付きデータを分割しモデルを構築、構築したモデルを使ってラベルなしデータを予測し疑似ラベルを付与、一定の閾値以上のデータをラベル付きデータとして扱い、分類器 1と分類器2のデータを加えて再度学習。このサイクルを繰り返し学習モデルを構築します。


- 自己教師あり学習の特徴は、人の手によるアノテーション作業の排除が挙げられます。機械自身が与えられたラベルなしデータを使って、教師データを作成しながら深層学習モデルを構築していきます。使おうとしているデータにラベルがない場合に有効な手法です。代表的な手法にContrastive Learningがあります。


- Contrastive learning
  - 似ているサンプルデータは近くになるように埋め込み、異なるサンプルデータの場合は埋め込みが遠くなるように学習する手法です。以下の図を具体的に見ると、左側のOriginal imageとAugmented Positive Imageの組み合わせは埋め込みが近くなるように学習され、Original imageとNegative Imageの組み合わせが埋め込みが遠くなるように学習されます。

  - Contrastive learning
    参照元：A SURVEY ON CONTRASTIVE SELF-SUPERVISED LEARNING

手法の選択

| 観点                                       | 選択肢                                             | 備考                                                                    |
| :----------------------------------------- | :------------------------------------------------- | :---------------------------------------------------------------------- |
| データセットにおけるラベル付きデータの割合 | 多い：半教師あり学習<br>少ない：自己教師あり学習   | ー                                                                      |
| ラベル付けにかかるコスト                   | 多い：自己教師あり学習<br>低い：半教師あり学習     | ー                                                                      |
| 学習データの質                             | 良い：半教師あり学習<br>悪い：自己教師あり学習     | 質:学習データ内のノイズや誤ったラベル付けがされたデータなどを指します。 |
| 利用できる計算リソース                     | 大きい：自己教師あり学習<br>小さい：半教師あり学習 | 自己教師あり学習は大規模な計算リソース必要                              |

| データ         | 半教師あり学習                                       | 自己教師あり学習                             |
| :------------- | :--------------------------------------------------- | :------------------------------------------- |
| テキストデータ | 分類、感情分析、文章要約、機械翻訳等                 | 意味表現の学習、トピック抽出、固有表現抽出等 |
| 画像データ     | 画像分類、物体検出、画像セグメンテーション、顔認識等 | 特徴量抽出、異常検知、画像生成等             |
| 音声データ     | 音声認識、音声合成、話者識別等                       | 音声識別、音声クラスタリング等               |

## CPU・GPU・TPUの演算方法


### SISD
- SISD（Single Instruction, Single Data）
- 1つの命令を命令プールから取得
- 1つのデータをデータプールから取得
- 取得した命令で取得したデータに実行
- 古いCPUで一般的に使用される
- **シンプルなアーキテクチャであるため、挙動予測やデバッグが容易**
- 命令をシリアルに実行し、一つの命令が完了するまで次の命令に移らない
- ハードウェアは、シングルスレッドのプログラムの実行に最適化される
- シリアル処理のため、プログラムの挙動が予測しやすく、デバッグが容易

### SIMD
- SIMD（Single Instruction, Multiple Data）
- 1つの命令を命令プールから取得
- 複数のデータをデータプールから取得
- 取得した命令で複数のデータに対し、複数の処理ユニットで同時に実行
- 一部のCPU、GPU、TPUで使用される
- **大量のデータに対する同じ演算を高速に実行可能**
- 画像処理で、全ピクセルに対して同じ演算を適用する場合、並列・高速に実行可能
- Deep learningで、テンソル内の要素に対する演算を一度に行い、行列乗算が高速に実行可能
- パイプライン処理が可能な点や高いスケーラビリティなどの特徴も持つ

### MIMD
  - 複数の命令を命令プールから取得
  - 複数のデータをデータプールから取得
  - 取得した複数の命令で、取得した複数のデータに対し、複数の処理ユニットで実行
  - マルチコアのCPU、スパコン、分散コンピューティングシステムで使用される
  - **大規模なデータ解析など、複雑な計算処理が求められる問題に対応可能**
  - 各処理ユニットが異なるタスクに取り組めるため、非常に高い柔軟性を持つ
  - 新しい処理ユニットを追加することで、全体の計算能力を容易に拡張可能であり、高いスケーラビリティを持つ

### CPU・GPU・TPUのメモリアーキテクチャ

####  CPUのメモリアーキテクチャ
-  システムの主メモリ（RAM）を直接操作し、メモリとの間でデータをやり取りする
-  L1、L2、L3などの階層的なキャッシュメモリを持ち、処理速度を高速化するために使い分けられる
$\rightarrow$ L1：最も高速だが低容量、L2：L1より速度は劣るが容量は大きい、L3：最も低速だが容量は最も大きい
-  低レイテンシに最適化されており、ランダムアクセスパターンや小さなデータセットの処理に適している

#### GPUのメモリアーキテクチャ
-  高いメモリ帯域幅（一定時間内に読み書きできるデータの最大量）を持ち、大量のデータを同時に処理可能
-  個々のGPUコアに近い「ローカルメモリ」と、全てのコアからアクセス可能な「グローバルメモリ」の両方を使用
$\rightarrow$ ローカルメモリは高速だが低容量、グローバルメモリは低速だが大容量
-  多数のコアが同時にメモリにアクセスすることを想定しているため、並列データ処理に最適化される

#### TPUのメモリアーキテクチャ
-  機械学習やDeep learningに特化したアーキテクチャを持つ
-  行列乗算に最適化されたユニットを持ち、深層学習モデルの学習と推論を大幅に高速化可能
$\rightarrow$ テンソル（多次元配列）処理や行列演算に特化したメモリ設計。大規模な行列データをオンチップのメモリから読み込むことが可能。
-  データの移動を最小限に抑えるための大容量のオンチップメモリを持つ
$\rightarrow$ 計算コアに非常に近いため、データアクセスのレイテンシを最小限に抑えることができ、演算速度が向上
-  TensorFlowなどの機械学習フレームワークと密に連携することで、メモリ使用の最適化などをソフトウェアレベルで調整可能

# EfficientNet
([目次に戻る](#目次))
- AlexNet以降、CNNモデルを大規模にスケールアップすることで精度を改善するアプローチが主流
（例：ResNet-18からResNet-200まである）


拡充では、以下の変数（幅、深さ、解像度）を「適切」にスケールアップ
- 幅：レイヤーのサイズ（ニューロンの数）
- 深さ：レイヤーの数
- 解像度：入力画像の大きさ
精度を向上できたものの、モデルが複雑で高コスト

2019年に開発されたEfficientNetモデル群は、効率的なスケールアップの規則を採用することで、開発当時の最高水準の精度を上回り、同時にパラメータ数を大幅に減少

## EfficientNetとCNNモデルのスケールアップ**

* ICML2019の論文で、新たなモデルスケーリングの「法則」が提案された
* 幅、深さ、解像度などを個別に増やすか、複合係数（Compound Coefficient）を導入することで最適化
    * 参考文献「EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks」解説；
    https://arxiv.org/abs/1905.11946

* EfficientNetではCompound Coefficientに基づいて、深さ・広さ・解像度を最適化したことにより、「小さなモデル」がかつ高い精度を達成
* モデルが小さい（パラメータ数が少ない）$\rightarrow$効率化（小型化と動作の高速化）

[図: EfficientNet-B0のアーキテクチャ]
画像引用: Google AI Blog, https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html

## EfficientNetの性能**
* EfficientNetは精度と効率の両側面で優れている
* パラメータの数と計算量は数倍〜1桁減少
* ResNet-50に比べEfficientNet-B4は同程度の処理速度と計算量で精度が$6.3\%$改善

* EfficientNetは、転移学習でも性能を発揮（シンプルかつ簡素な構造、汎用性が高い）

-  AlexNet 以降は、CNN モデルを大規模にスケールアップすることで精度を改善するアプローチが主流となった（例：ResNet-18 から ResNet-200 まである）

-  従来では、以下の変数（幅、深さ、解像度）を「適当」にスケールアップ
    - 幅：1 レイヤーのサイズ (ニューロンの数)
    - 深さ：レイヤーの数
    - 解像度：入力画像の大きさ

-  精度を向上できたものの、モデルが複雑で高コスト


EfficientNet と CNN モデルのスケールアップ

-  EfficientNet では Compound Coefficient に基づいて、深さ・広さ・解像度を最適化したことにより、「小さなモデル」かつ高い精度を達成
-  モデルが小さい（パラメータ数が少ない）→ 効率化（小型化と動作の高速化）
-  ICML2019 の論文で、新たなモデルスケーリングの「法則」が提案された
-  幅、深さ、解像度などを何倍増やすかは、複合係数（Compound Coefficient）を導入することで最適化

参考論文「EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks」解説;
https://arxiv.org/abs/1905.11946

（右図）EfficientNet-B0 のアーキテクチャ
画像引用：Google AI Blog, https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html


## EfficientNet の性能

-  EfficientNet は精度と効率の両側面で優れている
-  パラメータの数と計算量は数倍〜１桁減少
-  ResNet-50 に比べて EfficientNet-B4 は同程度の処理速度と計算量で精度が 6.3% 改善
-  EfficientNet は、転移学習でも性能を発揮（シンプルかつ簡潔な構造、汎用性が高い）

> モデルはオープンソースとして公開
> （参考）EfficientNet の実装：
> https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet

## Compound Scaling Method（複合スケーリング手法）の詳細

#### Depth (d)：
-  ネットワークの層を深くすることで、表現力を高くし、複雑な特徴表現を獲得できる

#### Width (w)：
-  ユニット数を増やすことでより細かい特徴表現を獲得し、学習を高速化できる
-  しかし、深さに対してユニット数が大きすぎると、高レベルな特徴表現を獲得しにくくなる

#### Resolution (r)：
-  高解像度の入力画像を用いると、画像中の詳細なパターンを見出せる
-  時代が進むにつれてより大きいサイズの入力画像が使われるようになってきた
-  畳み込み演算の演算量 FLOPS は $d, w^2, r^2$ に比例
（例：depth が 2 倍になると FLOPS も 2 倍、width と resolution が 2 倍になると FLOPS は 4 倍になる）

-  3 つのパラメータ Depth、width、resolution は、単一の係数 $\phi$ で一様にスケーリング可能
-  $\alpha, \beta, \gamma$ はグリッドサーチで求める定数
-  $\phi$ はユーザー指定パラメータ、モデルのスケーリングに使用できる計算リソースを制御する役割

ディープラーニングの手法

Compound Scaling Method（複合スケーリング手法）の詳細

-  CNN では畳み込み演算が計算コストを占領するので、$w$ と $r$ が 2 乗のオーダーで効いてくる
$\rightarrow$ $d, w^2, r^2$ に比例する FLOPS は $\sim (\alpha \cdot \beta^2 \cdot \gamma^2)^\phi$ 倍にスケールする
※ 原論文では $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ という制約を設けており、FLOPS は $\sim 2^\phi$ で増加すると近似できる

## 参考文献からの補足
> 原文: [Google AI Blog (2019)](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)  
> 著者: Mingxing Tan（Staff Software Engineer）と Quoc V. Le（Principal Scientist）
##  AutoMLによるベースアーキテクチャ設計
- ベースモデル EfficientNet-B0 は AutoML MNAS を用いて自動的に設計
- MobileNetV2 と同様の MBConv（Mobile Inverted Bottleneck Conv）を採用
- FLOPS（演算量）と精度の両方を最適化

## ImageNetでの性能比較

| モデル          | Top-1精度 | パラメータ数 | 推論速度（CPU） | 備考            |
| --------------- | --------- | ------------ | --------------- | --------------- |
| ResNet-50       | 76.3%     | 中           | 中              | ベースライン    |
| EfficientNet-B4 | 82.6%     | 同等         | 高速            | +6.3%精度向上   |
| EfficientNet-B7 | 84.4%     | 8.4倍小さい  | 6.1倍高速       | Gpipeより高性能 |

### 転移学習でも高性能

EfficientNetはImageNet以外のデータセットでも優れた性能を発揮：

- CIFAR-100: 91.7%
- Flowers: 98.8%
- 他にも8つの転移学習データセットで検証し、5つでSOTA（State-of-the-art）達成
- 最大 21倍のパラメータ削減 にもかかわらず精度は維持または向上

### 今後への影響と利点

EfficientNetは以下のユースケースで非常に有望：

- モバイルデバイスやエッジデバイス（低リソース環境）
- 高精度を必要とする画像分類タスク
- 転移学習のベースモデルとしての活用

GitHub: [tensorflow/tpu/models/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

### 関連分野

- 機械学習・深層学習
- コンピュータビジョン
- モデル圧縮・効率化
- AutoML

# Vision Transformer
([目次に戻る](#目次))

### 画像特徴量の入力方法･･･ 画像の”トークン”系列化
1. 画像をパッチに分割し系列化 ･･･ $N$個の画像パッチで構成される系列
2. パッチごとにFlatten化 ･･･ “トークン(単語)”化 → これを入力値に使用

###  ViTのアーキテクチャ ･･･ Transformer Encoderの使用
Transformer Encoderへの入力値の準備
1. 画像データから計算したEmbedding表現(埋込表現)を計算
2. 系列の最初に[CLS] Tokenという特別な系列値を付加
3. パッチの位置関係を示すPosition Embeddingの付加
-  Transformer Encoder ･･･ 言語処理向けのオリジナルと同等の構造
-  MLP Head ･･･ [CLS] Token系列値の出力特徴量から分類結果を出力

###  事前学習とファインチューニング
1. 大規模なデータセットでの事前学習
  実験されたデータセット JFT-300M > ImageNet-21k > ImageNet
2. ファインチューニング ･･･ 事前学習より高解像度な画像を入力
  目的タスクに応じたMLP Headの変更
  Position Embeddingの差し替え

## 画像特徴量の入力方法
画像特徴量の処理
1. 入力画像をパッチに分割する
2. パッチごとにFlatten処理を行い、”トークン”系列得る
3. Embedding表現(埋込表現)に変換する
   Inductive bias: Embedding表現に線形変換を使用
   Hybrid Architecture: Embedding表現にCNNを使用
4. CLS Tokenを系列データの最初に付加する
   Transformer Encoderの出力で、このトークンに対する出力をMLP Headで利用し、分類結果を得る
5. Position Embedding(パッチの位置)を付加する
   この情報はパラメータであり、学習により自動獲得される

ViTでは、画像をパッチに分割し、系列データとして利用できるよう加工し、特徴量として使用します。

## ViTの計算過程

1.  画像 $x \in \mathbb{R}^{H \times W \times C}$ （高さ $H$ 幅 $W$ チャンネル $C$）を、縦横が $P$ のパッチで分割します。
    分割した画像は $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$ （$N = HW/P^2$ はパッチ数）です。
    ※ $N$ がTransformerの入力における系列数になります。

2.  パッチ画像を $D$ 次元の特徴量 $Z_0$ に変換しTransformer Encoderに入力します。
    $$
    Z_0 = [X_{class}; X_p^1 E; X_p^2 E; ...; X_p^N E] + E_{pos} \quad E \in \mathbb{R}^{(P^2 \cdot C) \times D}, E_{pos} \in \mathbb{R}^{(N+1) \times D}
    $$
    $E$ は画像の埋め込み表現（$D$ 次元）への変換
    $X_{class}$ はCLS Tokenで、パッチ系列の先頭に付加される
    $E_{pos}$ はPosition Embeddingsで、パッチ画像の位置関係を学習

3.  Transformer Encoderは$L$回（$L$層）重ね、$\mathcal{l}$層目について、
    Layer Norm を $LN$ Multi-Head self attentionを $MSA$ とし、
    $$
    Z'_{\mathcal{l}} = MSA(LN(Z_{\mathcal{l}-1})) + Z_{\mathcal{l}-1} \quad (\mathcal{l} = 1 ... L)
    $$
    $$
    Z_{\mathcal{l}} = MLP(LN(Z'_{\mathcal{l}})) + Z'_{\mathcal{l}} \quad (\mathcal{l} = 1 ... L)
    $$

4.  Transformer Encoderの最終層では、[CLS]トークンに当たる特徴量を出力
    $$
    y = LN(Z_L^0)
    $$

5.  この$y$をMLP Headに入力し、最終的な分類結果を得る

## 事前学習とファインチューニング
- 事前学習
  - ViTの事前学習は教師ラベル付きの大規模なデータセットで行われる
  - 論文では、ImageNet/ImageNet-21k/JFT-300M
  - 事前学習の手順
  - 事前学習時にはMLP Headを出力層に使用
  - ViTの事前学習は教師ラベル付きの大規模なデータセットで行われる
  - ファインチューニングより低解像度の画像を使用
- ファインチューニング
  - 分類クラスの変更
  - Transformer Encoderは$D$次元の特徴量出力を行う
  - MLP Headで$D$次元を$K$クラスへ変換する
  - ファインチューニング時にはMLP HeadをLinear層に取り換え
  - 入力解像度の変更
  - パッチサイズは変更せず、入力時のPosition Embeddingを付替え
  - 事前学習よりファインチューニングでは高解像度の画像を使用
  - ViTは、大規模なデータセットで事前学習が行われ、ファインチューニングでは出力クラス数･入力解像度を変更することができます

| データセット | クラス数 | 画像枚数  |
| :----------- | :------- | :-------- |
| ImageNet     | 1,000    | 130万枚   |
| ImageNet-21k | 21,000   | 1,400万枚 |
| JFT-300M     | 18,000   | 3億枚     |

論文中の実験において事前学習で用いられたデータセット

|                  | 事前学習 | ファインチューニング |
| :--------------- | :------- | :------------------- |
| バッチサイズ     | 4,096    | 512                  |
| オプティマイザー | Adam     | SGD with momentum    |

論文中の実験において事前学習とファインチューニングの構成の違い

| Model     | Layers | Hidden size | MLP size | Heads | Params |
| :-------- | :----- | :---------- | :------- | :---- | :----- |
| ViT-Base  | 12     | 768         | 3,072    | 12    | 86M    |
| ViT-Large | 24     | 1,024       | 4,096    | 16    | 307M   |
| ViT-Huge  | 32     | 1280        | 5,120    | 16    | 632M   |


### Vision Transformerの事前学習のデータ量と精度
- 事前学習におけるデータセット規模と性能の関係
  - ImageNet < ImageNet-21k < JFT-300Mでの事前学習 (上段左)
  - Weight decay･Dropout･Label smoothingを最適化
- 大規模データセットでViTがBiTより有利
  -  JFT-300Mのサブセット(9M < 30M < 90M < 300M)での事前学習 (上段右)
- 大規模データセットでViTがResNetより有利
- 事前学習における計算量と性能の関係 (下段)
  -  ViTは同計算量においてBiTより高性能
  -  ViTは大計算量域でさらなる性能向上が見られる
  -  Hybrid architectureは小計算量域で高性能 (大計算量では違いがない)

### Vision Transformerのファインチューニング性能
ViTのファインチューニング性能はCNNモデルより高く、学習に必要な計算量も少ない

## まとめ
### Vision Transformerのデータ表現と入力
- 画像をパッチに分割し、パッチごとの特徴量を系列データとしてTransformerに入力する。
- パッチごとの特徴量は、ピクセル値をFlatten処理･Embedding処理(埋め込み)を行い、Position Embedding情報を加えたもの。
- 入力系列の1番目に、分類タスク用に特別な”[CLS]”トークンを連結して入力する。従って、入力系列の長さは、パッチ数 + 1 となる。

## Vision Transformerのアーキテクチャ
- 言語処理におけるTransformerのエンコーダーとほぼ同様の構造である。(系列データを入力)
- エンコーダー部分からの出力における1トークン目の特徴量を、MLP Headに入力、最終的な分類結果を出力
- モデルサイズの違いにより、Base/Large/Hugeの3つが存在する。
### Vision Transformerにおける事前学習(Pre-training)とファインチューニング(Fine-tuning)
- ViTの事前学習は、教師ラベル付きの巨大データセットで行われ、性能がデータセット規模の影響を受ける
- ファインチューニングでは、MLP Head部分を取り換えることで、分類タスクにおけるクラス数の違いに対応する。
- ファインチューニングで、事前学習より高い解像度の画像に、Position Embeddingの変更のみで対応できる。
### 性能とその評価
- 前学習におけるデータセットが大規模な場合において、既存手法より高性能。
- 事前学習のデータセットが小規模な場合、既存手法(CNN)に比べ低性能。
- 同計算量において、既存手法より高性能。大計算量域でさらなる性能向上の余地がある。

# 物体検知とSS分解
([目次に戻る](#目次))

## 鳥瞰図：広義の物体認識タスク

|                  | 出力                                     |
| :--------------- | :--------------------------------------- |
| **分類**         | (画像に対し単一または複数の)クラスラベル |
| **物体検知**     | Bounding Box [bbox/BB]                   |
| **意味領域分割** | (各ピクセルに対し単一の)クラスラベル     |
| **個体領域分割** | (各ピクセルに対し単一の)クラスラベル     |

## 種類
- セマンティックセグメンテーション

目的： 画像内の全てのピクセルをカテゴリに分類する
特徴： 同じカテゴリーの物体でも個々に区別せず、全てのピクセルが何らかのカテゴリーに分類される

- パノプティックセグメンテーション

目的： セマンティックセグメンテーションと物体検出の組み合わせで、画像内の全ての物体を個別に認識し、それぞれのピクセルにカテゴリを割り当てる
特徴： 個々の物体を識別しつつ、画像内の全ピクセルをカバーする

## 参考文献からの補足
パノプティックセグメンテーションは2018年に提案された比較的新しいタスクで、Thing（可算物体）とStuff（背景領域）の両方を統一的に扱う。現在ではMask2FormerやPanoptic-DeepLabなどの手法が提案されている。
出典： Kirillov, A., et al. (2019). Panoptic segmentation. CVPR.

## 代表的なデータセット
いずれも物体検出コンペティションで用いられたデータセット
- VOC： Visual Object Classes
- ILSVRC： ImageNet Scale Visual Recognition Challenge
- COCO： Common Object in Context
- OICOD： Open Images Challenge Object Detection

さまざまなアルゴリズムを精度評価する時にデータセットが利用される
目的に応じたデータセットの選択を行う必要がある

BOX/画像→一枚にどれくらい物体あるか？は大事！
→アイコンみたいの？ORより複雑（自動運転）
※ILSVRCだけはInstance Annotationがない

## 補足
近年では、より大規模で多様なデータセットが登場している。例えば、Objects365（365クラス、200万画像）、Open Images V6（600クラス、900万画像）などがある。また、3D物体検出向けのKITTI、nuScenes、Waymoデータセットも重要である。
出典： Lin, T. Y., et al. (2014). Microsoft COCO: Common objects in context. ECCV.

## IoU（Intersection over Union）
- 物体検出においてはクラスラベルだけでなく、物体位置の予測精度も評価したい！
- IoU = TP / (TP + FP + FN)

- Confusion Matrixの要素を用いて表現
  重なっている部分がTP、他の部分がFPもしくはFN
  > IoUは、Ground-Truth BBに対する占有面積、Predicted BBに対する占有面積ではないことに注意


## 補足
IoUは物体検出における最も基本的な評価指標だが、境界ボックスの形状や位置関係を考慮しない問題がある。近年では、GIoU（Generalized IoU）、DIoU（Distance IoU）、CIoU（Complete IoU）などの改良版が提案されている。
出典： Rezatofighi, H., et al. (2019). Generalized intersection over union: A metric and a loss for bounding box regression. CVPR.

## 精度評価の指標
- AP：Average Precision（PR曲線の下側面積）
- AP = ∫₀¹ P(R)dR
- mAP：mean Average Precision
- クラス数がCの時：
  - mAP = (1/C) ∑ᵢ₌₁ᶜ APᵢ
  - FPS：Frames per Second
  応用上の要請から、検出精度に加え検出速度を指標としたもの

## 補足
COCOデータセットでは、mAP@0.5:0.95（IoU閾値0.5から0.95まで0.05刻みで計算したmAPの平均）が標準的な評価指標として使用される。また、小物体（mAP-S）、中物体（mAP-M）、大物体（mAP-L）での性能も重要な指標となっている。
出典： COCO Detection Challenge: https://cocodataset.org/#detection-eval

## 深層学習以降の物体検知

2012年、AlexNetの登場を皮切りに、時代はSIFTからDCNNへ
2013年以降のアルゴリズムは以下の通り

- 2段階
  候補領域の検出とクラス推定を別々に行う
  相対的に精度が高い傾向
  相対的に計算量が大きく推論も遅い傾向


- 1段階
  候補領域の検出とクラス推定を同時に行う
  相対的に精度が低い傾向
  相対的に計算量が小さく推論も早い傾向


## SSD（Single Shot Detector）
Default BOXを適当な位置に適当な大きさで用意
SSDが学習を進めてDefault BOXを修正していく

## 特徴マップからの出力

- オフセット項としてDefault Boxの中心位置や幅、高さを用意している
- k個のDefault Boxを用意する時、特徴マップのサイズがm×nだとすると、特徴マップごとに用意するDefault Box数はk×m×nとなる

## 補足： SSDは2016年に提案された1段階検出器の代表例で、異なるスケールの特徴マップを使用してマルチスケール検出を実現した。その後、RetinaNet、YOLOv3/v4/v5などの改良版が登場し、精度と速度の両方で大幅な改善が見られている。
出典： Liu, W., et al. (2016). SSD: Single shot multibox detector. ECCV.

## Semantic Segmentation

Convolutionを重ねていった最後に解像度を元に戻す（Up-sampling）
Convolutionにより解像度が落ちていくのが問題

### 解決手法：
- FCN
  低レイヤーPooling層の出力を要素ごとに足し算することでローカルな情報を補完してからUp-sampling
- Unpooling
  プーリングした時の位置情報を保持しておき、プーリングを元に戻すときに保持していた位置情報を利用する
- Dilated Convolution
  Convolutionの段階で受容野を広げる手法
  

## 問題
## 物体検出タスクにおける大規模データセットの重要性

物体検出タスクでは、精度評価を行う上で大規模なデータセットの存在が不可欠です。

以下の選択肢のうち、各データセットに関する説明として最も妥当な主張と考えられるものを1つ選んでください。

**選択肢:**

1.  VOC12ではGround-Truth BBの情報が中心座標と幅、高さで与えられていない。
2.  フリマアプリの出品画像をを入力とする物体検出タスクに取り組みたい。訓練データとしてCOCO18を用いることで精度の向上が期待される。
3.  ILSVRC17はImageNetのサブセットで構成され、Instance Segmentationの学習に必要な情報も与えられている。
4.  OICID18はクラス数が500と比較的大きなデータセットであるが、物体検出タスクにおいては常にクラス数の大きいデータセットで学習を行う方が好ましい。

---

## 解答：[テーマ] 代表的データセットの特徴

物体検出タスクには精度評価をはじめ大規模なデータセットの存在が欠かせません。

以下の選択肢のうち、各データセットに関する説明として最も妥当な主張と考えられるものを1つ選んでください。

**選択肢:**

1.  VOC12ではGround-Truth BBの情報が中心座標と幅、高さで与えられていない。
2.  フリマアプリの出品画像をを入力とする物体検出タスクに取り組みたい。訓練データとしてCOCO18を用いることで精度の向上が期待される。
3.  ILSVRC17はImageNetのサブセットで構成され、Instance Segmentationの学習に必要な情報も与えられている。
4.  OICID18はクラス数が500と比較的大きなデータセットであるが、物体検出タスクにおいては常にクラス数の大きいデータセットで学習を行う方が好ましい。

---

**追加情報:**

VOC12ではGround-Truth BBの情報として、左上隅 ($x_{min}, y_{min}$) と右下隅 ($x_{max}, y_{max}$) の座標が与えられます。

## 問題

正方形で面積の等しいGround-Truth BBとPredicted BBを考える。両者が完全に重なっているときのIoUは1.0である。Predicted BBが完全一致の状態から右方向および上方向にそれぞれ1辺の長さの10%だけ平行移動された場合のIoUとして適切な選択肢を選べ（有効数字3桁）。
必要があれば以下の図を参考に考えてもよい。ただし、図は正確に描かれていないことに注意せよ。


**選択肢:**
① 0.900
② 0.810
③ 0.681
④ 0.669

---

## 解答：[テーマ] IoUの計算

正方形で面積の等しいGround-Truth BBとPredicted BBを考える。両者が完全に重なっているときのIoUは1.0である。Predicted BBが完全一致の状態から右方向および上方向にそれぞれ1辺の長さの10%だけ平行移動された場合のIoUとして適切な選択肢を選べ（有効数字3桁）。
必要があれば以下の図を参考に考えてもよい。ただし、図は正確に描かれていないことに注意せよ。


$$
IoU = \frac{81}{119} \approx 0.6806
$$

この問題では、2つの正方形のバウンディングボックス (BB) があり、それらのIoU (Intersection over Union) を計算します。

1辺の長さを10と仮定すると、Ground-Truth BBとPredicted BBの面積はそれぞれ $10 \times 10 = 100$ となります。

Predicted BBが右方向および上方向にそれぞれ10%（つまり1）だけ移動した場合、重なっている部分 (Intersection) は $9 \times 9 = 81$ となります。

Unionは、$Ground-Truth BBの面積 + Predicted BBの面積 - Intersection$ で計算できます。
Union $= 100 + 100 - 81 = 119$

したがって、IoUは次のように計算されます。
$$
IoU = \frac{Intersection}{Union} = \frac{81}{119} \approx 0.68067...
$$

有効数字3桁で丸めると、$0.681$ となります。

**選択肢:**
① 0.900
② 0.810
③ **0.681**
④ 0.669

**正解は③です。**

## 問題

物体検出モデルの評価にはPrecisionやRecallといった指標が用いられることも多い。ここで、それぞれの指標は次式で定義されます。

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

ここで、TP, FP, FNはそれぞれTrue Positive, False Positive, False Negativeを表します。

ある物体検知アルゴリズムにおける「任意のPredicted BBに検出されないGround-Truth BB」の扱いとして適切な選択肢を選べ。

**選択肢**
① True Positive
② False Positive
③ True Negative
④ False Negative

---

## 解答：[テーマ] 評価指標計算の前提

物体検出モデルの評価にはPrecisionやRecallといった指標が用いられることも多い。ここで、それぞれの指標は次式で定義されます。

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

ここで、TP, FP, FNはそれぞれTrue Positive, False Positive, False Negativeを表します。

ある物体検知アルゴリズムにおける「任意のPredicted BBに検出されないGround-Truth BB」の扱いとして適切な選択肢を選べ。


**選択肢**
① True Positive
② False Positive
③ True Negative
④ **False Negative**

**解説:**

* **True Positive (TP):** 正しく検出された物体（Ground-Truth BBとPredicted BBが一致）。
* **False Positive (FP):** 誤って検出された物体（実際には物体がないのにPredicted BBが存在する、または異なるカテゴリの物体を検出した）。
* **False Negative (FN):** 検出されなかった物体（Ground-Truth BBがあるのにPredicted BBが存在しない）。
* **True Negative (TN):** 物体がない領域を正しく「物体なし」と判断したケース。**物体検知においては基本的に考える必要がない** とされることが多いです。

問題の「任意のPredicted BBに検出されないGround-Truth BB」とは、**実際には物体があるのに、モデルがそれを検出できなかった場合**を指します。これはFalse Negativeの定義に合致します。

**正解は④です。**

## 問題

次の表はSSDの原著論文内で示された実験結果であり、各列の値はmAPを表している。SSD300とSSD512は入力画像の解像度の違いであり、SSDの基本的な構造は変わらない。この結果から読み取れる主張について誤っている選択肢を選べ。ただし、Methodに続く各カラムには実験で用いたテストデータ、訓練データ、IoUの閾値が3行にわたって表示されている。

| Method | VOC2007 `+07+12` | VOC2012 Test `07+12+COCO` | COCO `dev2015` `trainval35k` |
| :----- | :--------------- | :------------------------ | :--------------------------- |
|        | 0.5              | 0.5                       | 0.5                          |
|        | 0.5              | 0.5                       | 0.95 0.5 0.75                |
|        |                  |                           |                              |
| SSD300 | 74.3             | 79.6                      | 72.4                         |
| SSD512 | 76.8             | 81.6                      | 74.9                         |
|        |                  |                           |                              |
|        |                  |                           |                              |
|        |                  |                           | 23.2 41.2 23.4               |
|        |                  |                           | 26.8 46.5 27.8               |

**選択肢**
1.  VOC07とVOC12で扱われているクラス数は変わらない。
2.  COCOではBB位置の精度を重視した指標も用いられている。
3.  入力の解像度は精度に影響すると考えられる。
4.  SSDは小さな物体の検出を苦手とすることが予想される。

---

## 解答：[テーマ] SSDの弱点

次の表はSSDの原著論文内で示された実験結果であり、各列の値はmAPを表している。SSD300とSSD512は入力画像の解像度の違いであり、SSDの基本的な構造は変わらない。この結果から読み取れる主張について誤っている選択肢を選べ。ただし、Methodに続く各カラムには実験で用いたテストデータ、訓練データ、IoUの閾値が3行にわたって表示されている。

| Method | VOC2007 `+07+12` | VOC2012 Test `07+12+COCO` | COCO `dev2015` `trainval35k` |
| :----- | :--------------- | :------------------------ | :--------------------------- |
|        | 0.5              | 0.5                       | 0.5                          |
|        | 0.5              | 0.5                       | 0.95 0.5 0.75                |
|        |                  |                           |                              |
| SSD300 | 74.3             | 79.6                      | 72.4                         |
| SSD512 | 76.8             | 81.6                      | 74.9                         |
|        |                  |                           |                              |
|        |                  |                           | 23.2 41.2 23.4               |
|        |                  |                           | 26.8 46.5 27.8               |

**選択肢**
1.  VOC07とVOC12で扱われているクラス数は変わらない。
2.  COCOではBB位置の精度を重視した指標も用いられている。
3.  入力の解像度は精度に影響すると考えられる。
4.  SSDは小さな物体の検出を苦手とすることが予想される。

**正解: 1**

**解説:**

1.  **VOC07とVOC12で扱われているクラス数は変わらない。**
    これは誤りです。VOC07とVOC12はそれぞれ異なるデータセットであり、扱っているクラス数は異なります。VOC07は20クラス、VOC12は20クラスですが、データセットの規模や画像の種類が異なります。この表から直接クラス数の違いを読み取ることはできませんが、一般的な知識としてVOC07とVOC12のデータセットは独立しており、クラス数は同じでもデータ分布が異なります。

2.  **COCOではBB位置の精度を重視した指標も用いられている。**
    COCO `dev2015` `trainval35k` の行を見ると、IoUの閾値が`0.5`だけでなく、より厳密な`0.75`や、`0.5:0.95` (0.05刻みでの平均) など、複数のIoU閾値でmAPが評価されています。特に`0.75`は、物体位置の精度が高い検出に対してより高い評価を与えるため、BB位置の精度を重視していると言えます。

3.  **入力の解像度は精度に影響すると考えられる。**
    SSD300 (入力サイズ $300 \times 300$) とSSD512 (入力サイズ $512 \times 512$) を比較すると、どのデータセットにおいてもSSD512の方がmAPが高いことがわかります (例: VOC2007のSSD300: 74.3 vs SSD512: 76.8)。これは、入力解像度が高い方が精度が向上することを示しており、解像度が精度に影響を与えると考えられます。

4.  **SSDは小さな物体の検出を苦手とすることが予想される。**
    COCOデータセットのmAP (特に`0.5:0.95`や`0.75`の閾値) を見ると、VOCデータセットに比べてmAPの値が大幅に低くなっています (例: VOC2007のSSD512: 76.8 vs COCOのSSD512: 27.8)。COCOデータセットはVOCデータセットと比較して、より多様なスケールの物体、特に小さな物体が多く含まれていることで知られています。COCOでの性能が低いことから、SSDが小さな物体の検出を苦手としていると予想されます。

以上の理由により、誤っている選択肢は1です。

## 問題

4x4の特徴マップにMax Poolingを適用した後の特徴マップとその際のMax Pooling Indicesが下図のように与えられている。2x2の特徴マップをUnpoolingした際の正しい特徴マップを選択肢から選べ。ただし、Poolingに用いたカーネルサイズは2x2、strideは2とする。


**選択肢**

  
---

## 問題

4x4の特徴マップにMax Poolingを適用した後の特徴マップとその際のMax Pooling Indicesが下図のように与えられている。2x2の特徴マップをUnpoolingした際の正しい特徴マップを選択肢から選べ。ただし、Poolingに用いたカーネルサイズは2x2、strideは2とする。


**選択肢**

  
---

## 解答：[テーマ] Unpoolingの基本

4x4の特徴マップにMax Poolingを適用した後の特徴マップとその際のMax Pooling Indicesが下図のように与えられている。2x2の特徴マップをUnpoolingした際の正しい特徴マップを選択肢から選べ。ただし、Poolingに用いたカーネルサイズは2x2、strideは2とする。


**解説:**

Max Poolingでは、指定されたカーネルサイズ（ここでは2x2）とストライド（ここでは2）で特徴マップを走査し、各領域内の最大値を取り出します。同時に、その最大値が元の特徴マップのどの位置から来たかを示すインデックス（Max Pooling Indices）を記録します。

UnpoolingはMax Poolingの逆操作です。Max Pooling Indicesを使用して、プーリング後の特徴マップの値を元の位置に戻します。

与えられた特徴マップ（2x2）とMax Pooling Indices（4x4）を見ていきます。

**特徴マップ:**
3  7
1  8


**Max Pooling Indices:**
0 0 0 0
1 0 0 1
0 1 0 0
0 0 1 0


1.  **特徴マップの `3`:** Max Pooling Indicesの左上2x2ブロックで `1` がある位置（(1,0) - 2行1列目）に `3` を配置します。
    ```
    0 0 0 0
    3 0 0 0
    0 0 0 0
    0 0 0 0
    ```

2.  **特徴マップの `7`:** Max Pooling Indicesの右上2x2ブロックで `1` がある位置（(1,3) - 2行4列目）に `7` を配置します。
    ```
    0 0 0 0
    0 0 0 7
    0 0 0 0
    0 0 0 0
    ```

3.  **特徴マップの `1`:** Max Pooling Indicesの左下2x2ブロックで `1` がある位置（(2,1) - 3行2列目）に `1` を配置します。
    ```
    0 0 0 0
    0 0 0 0
    0 1 0 0
    0 0 0 0
    ```

4.  **特徴マップの `8`:** Max Pooling Indicesの右下2x2ブロックで `1` がある位置（(3,2) - 4行3列目）に `8` を配置します。
    ```
    0 0 0 0
    0 0 0 0
    0 0 0 0
    0 0 8 0
    ```

これらをすべて組み合わせると、以下のようになります。

0 0 0 0
3 0 0 7
0 1 0 0
0 0 8 0


これは選択肢②と一致します。

**正解は②です。**

# Mask R-CNN
([目次に戻る](#目次))

Mask R-CNN とその関連技術

物体検出は特徴を抽出し、対象物の領域を切り出し、クラスを認識し....とたくさんのことをやっている

* 物体検出の流れ
    1.  物体の同定（identification）：画像の中で物体がどこにあるのか？
    $\downarrow$
    2.  物体認識/分類（classification）：何の物体であるか？

物体検出
* 物体検出はバウンディングボックスを使用
    長方形で関心領域を切り出す
    4座標を予測する回帰問題
    Ross, Girshick, "Fast r-cnn." in ICCV2015

セマンティックセグメンテーションとは

* 物体領域を画素単位で切り出し、各画素にクラスを割り当てる手法
* 工業検査や医療画像解析など、精密な領域分割に応用される
* 重要な学習データセットに、VOC2012とMSCOCOがある
* 主要手法は、完全畳み込みネットワーク（FNC; Fully Convolutional Network）
    * 全ての層が畳み込み層、全結合層を有しない
    * 画素ごとにラベル付した教師データを与えて学習する $\rightarrow$ 出力ノードが多数
    * 未知画像も画素単位でカテゴリを予測する
    * 入力画像のサイズは可変で良い

（左図）バイクと乗車している人間のそれぞれの境界線 / 輪郭線を描くためには、画素単位の高密度な予測をが必要

R-CNN（Regional CNN）

* 物体検出 + 物体認識のアルゴリズムの原形は R-CNN（Regional CNN）
* 物体検出タスクと物体認識タスクを順次に行う
    Ross, Girshick, "Fast r-cnn." in ICCV2015

- 関心領域（ROI;Region of Interest）を切り出す
- 類似する領域をグルーピング
- 候補領域の画像の大きさを揃える
- CNNにより特徴量を求める
- CNNで求めた特徴量をSVMで学習
- 未知画像も特徴量をCNNで求め、学習済みSVMで分類

- R-CNNの発展版

* R-CNNは多数の物体領域に対し複雑な処理を行うため、処理が重くて遅いのが課題
* 改良版の高速R-CNN（Fast R-CNN）では、関心領域ごとに畳み込み層に通すのではなく、画像ごとに一回の畳み込み操作を行う $\rightarrow$ 計算量を大幅に減少
* その後、さらに Faster R-CNN も開発された
    * 関心領域の切り出しもCNNで行う
    * ほぼリアルタイムで動作し、動画認識への応用も
* 他の発展版、YOLO（You Only Look Once）や SSD（Single Shot Detector）も領域の切り出しと物体認識を同時に行う

Ross, Girshick, "Fast r-cnn." in ICCV2015
高速R-CNN（Fast R-CNN）の模式図

### 物体認識タスクの分類

| タスク                                | 出力                                         |
| ------------------------------------- | -------------------------------------------- |
| 分類（Classification）                | クラスラベル                                 |
| 物体検知（Object Detection）          | Bounding Box + クラス + 信頼度（Confidence） |
| 意味領域分割（Semantic Segmentation） | 各ピクセルに対するクラスラベル               |
| 個体領域分割（Instance Segmentation） | 各物体に対して独立したピクセルラベリング     |

## 物体検知の評価指標

### IoU（Intersection over Union）

$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}
$$

- **Overlap**: 予測と正解の共通部分
- **Union**: 予測と正解の和集合
- IoUとConfidenceの両方に閾値を設けて精度を評価

### Precision-Recall曲線
- Confidenceの閾値を変化させてPR曲線を描く
- **AP（Average Precision）**: PR曲線の下側の面積
- **mAP（mean AP）**: クラスごとのAPを平均

## 物体検出の代表的フレームワーク

### 2段階検出器（例：RCNN）
- 候補領域抽出と分類を別々に行う
- **高精度だが計算コスト大**

### 1段階検出器（例：YOLO）
- 候補抽出と分類を同時に行う
- **高速だがやや精度低**

## SSD（Single Shot Detector）

- ベースモデル：VGG16
- **マルチスケール特徴マップ** を利用
  - 大：小さい物体の検出
  - 小：大きい物体の検出

### 出力形式
- 各特徴マップから $k \times (\text{クラス数} + 4)$ を出力
  - 4：Bounding Boxのオフセット（$\Delta x, \Delta y, \Delta w, \Delta h$）

### 対処手法
- **Non-Maximum Suppression (NMS)**: 重複検出排除
- **Hard Negative Mining**: 背景クラスのバランス調整

## Semantic Segmentation

- 各ピクセルに対してクラスラベルを予測
- Upsampling（解像度復元）が必要

### 技術要素

- **Deconvolution / Transposed Convolution**
  - stride を挟んで特徴マップを拡張
- **FCN（Fully Convolutional Network）**
  - 低層の特徴と高層の特徴を合成して輪郭などを補完

### その他

- **Dilated Convolution**
  - フィルター内のピクセル間隔を空けて畳み込み
  - 同じカーネルサイズでも広い受容野を確保可能

# Instance Segmentation

## 概要

- **Instance Segmentation（個体領域分割）** は、Semantic Segmentation に加えて、同一クラス内で個別のオブジェクト（インスタンス）を識別するタスク。
- 各ピクセルに **クラスID + インスタンスID** を付与する。

---

## 代表的な手法

### YOLACT（You Only Look At CoefficienTs）

- **ワンステップ**でインスタンスセグメンテーションを実現。
- 軽量かつリアルタイム処理に向いている。

---

### Mask R-CNN

- **Faster R-CNN** をベースにした拡張モデル。
- バウンディングボックス領域ごとに**ピクセル単位のマスク出力**を追加。

---

## R-CNNファミリーの進化

### R-CNN

- 関心領域 (RoI) を **Selective Search** によって抽出。
- 各RoIをサイズ統一 → CNNで特徴抽出 → SVMでクラス分類。
- 精度は高いが **非常に遅い**。

###  Fast R-CNN

- 入力画像に1度だけCNNを適用し、得られた特徴マップを利用してRoIを抽出。
- **畳み込みの再利用**で高速化。

### Faster R-CNN

- **Region Proposal Network (RPN)** を導入。
- RoI候補の提案もCNNで処理可能に。
- **End-to-End学習**可能で高速・高精度。

---

## YOLO（You Only Look Once）

- **物体検出とクラス分類を一つのネットワークで同時に実行**。
- 画像全体を $S \times S$ グリッドに分割し、各グリッドから $B$ 個のバウンディングボックスを予測。
- **非常に高速**だが、Faster R-CNNに比べて**やや精度が劣る**。

---

## FCOS（Fully Convolutional One-Stage Object Detection）

- **アンカーフリー**の物体検出手法。
- アンカーボックスのハイパーパラメータ設定不要。
- 画像内の **全ピクセル**から物体位置（4次元ベクトル）を直接予測。

## Feature Pyramid Networks（FPN）

- **マルチスケールの特徴マップ**を生成。
  - 高解像度層：小さい物体に強い
  - 低解像度層：意味的特徴に強い
- 重なった物体の処理に強い。


## Mask R-CNN の構造

- Faster R-CNN に **マスク出力の分岐を追加**。
- RoI毎にマスク（ピクセル分類）を出力。
- **RoI Align** により RoI Pooling の精度問題を改善。
  - 特徴マップの補間処理により位置ズレを軽減。

## RoI Pooling vs RoI Align

| 手法        | 特徴                                         |
| ----------- | -------------------------------------------- |
| RoI Pooling | 固定サイズに間引き、**高速だがズレが生じる** |
| RoI Align   | 補間処理でズレを抑える、**精度向上**         |

---

このように、インスタンスセグメンテーションでは **物体の位置・クラス・形状（マスク）** を同時に扱うため、高精度での位置合わせやスケール対応が重要になります。

# GAN（Generative Adversarial Networks）とその拡張

## Discriminatorの更新戦略

- 通常、**Discriminatorの更新を複数回**行ってから、**Generatorの更新を1回**行う（毎回ペアではない）。
- **理想状態**：
  - 生成データと真データが区別できないほど類似している。
  - このとき価値関数は最適化され、Generatorの損失は**最小値**になる。

# Faster-RCNN, YOLO
([目次に戻る](#目次))

- ディープラーニングによる物体検出・セグメンテーション技術
物体検出は特徴を抽出し、対象物の領域を切り出し、クラスを認識し.... とたくさんのことをやっている

- 物体の同定（identification）： 画像の中で物体がどこにあるのか？
- 物体認識/分類（classification）： 何の物体であるか？

- 物体検出の特徴

- 物体検出はバウンディングボックスを使用
- 長方形で関心領域を切り出す
- 4座標を予測する回帰問題

出典： Ross Girshick, "Fast R-CNN." in ICCV2015
## セマンティックセグメンテーション
- 物体領域を画素単位で切り出し、各画素にクラスを割り当てる手法
- 工業検査や医療画像解析など、精密な領域分割に応用される
- 重要な学習データセットに、VOC2012とMSCOCOがある

- 主要手法：完全畳み込みネットワーク（FCN; Fully Convolutional Network）
- 全ての層が畳み込み層、全結合層を有しない
- 画素ごとにラベル付した教師データを与えて学習する → 出力ノードが多数
- 未知画像も画素単位でカテゴリを予測する
- 入力画像のサイズは可変で良い

- 応用例
バイクと乗車している人間のそれぞれの境界線・輪郭線を描くためには、画素単位の高密度な予測が必要

## R-CNN（Regional CNN）
物体検出+物体認識のアルゴリズムの原形はR-CNN（Regional CNN）
物体検出タスクと物体認識タスクを順次に行う

### 処理の流れ

- 関心領域（ROI; Region of Interest）を切り出す
- 類似する領域をグルーピング
- 候補領域の画像の大きさを揃える
- CNNにより特徴量を求める
- CNNで求めた特徴量をSVMで学習
- 未知画像も特徴量をCNNで求め、学習済みSVMで分類

> 出典： Ross Girshick, "Fast R-CNN." in ICCV2015

R-CNNの発展版
R-CNNは多数の物体領域に対し複雑な処理を行うため、処理が重くて遅いのが課題
## 高速R-CNN（Fast R-CNN）
関心領域ごとに畳み込み層に通すのではなく、画像ごとに一回の畳み込み操作を行う
計算量を大幅に減少

## Faster R-CNN
関心領域の切り出しもCNNで行う
ほぼリアルタイムで動作し、動画認識への応用も

## その他の発展版
YOLO（You Only Look Once）
SSD（Single Shot Detector）


入力画像をグリッド領域に分割
各グリッドでクラス分類を行う
バウンディングボックスで候補領域を抽出すると同時に、物体なのか、背景なのかの確率を表す信頼度スコアを算出
これらの情報を組み合わせて物体認識を行う

### メリット

処理が速い（アルゴリズムは1つのCNNで完結し、領域推定と分類を同時に行う）
画像全体を見て予測することができるため、誤検出が「Fast R-CNN」の半分以下

# FCOS
([目次に戻る](#目次))
R-CNNの基本構造
R-CNNは以下の2つのパートで構成される：

- パート1： 物体候補領域の提案
- パート2： 提案された候補領域における物体のクラス分類

## R-CNNの課題
パート1の処理にSelective Searchを使用しているため、処理速度が遅い
Fast-R-CNNではパート2の処理は改良されたものの、パート1にはSelective Searchを使用
画像1枚の処理時間：

パート1：1秒
パート2：0.22秒

- 課題解決への方向性
高速な処理を行う手法を提案する → 動画などのリアルタイム処理が可能になる
> 出典： Girshick, Ross B. et al. "Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation." 2014 IEEE Conference on Computer Vision and Pattern Recognition (2014): 580-587.

## Faster-RCNN
物体候補領域の提案の処理にCNNを使用するRPN（Region Proposal Network）を提案

- End-to-Endな処理が可能になった
  一括処理：物体候補領域の選定 → 物体のクラス分類 → パラメータの更新

### RPN（Region Proposal Network）の仕組み
- 入力された画像は、VGG16により特徴マップに変換される
- この特徴マップを後の処理で使用する
- 特徴マップにAnchor Pointsを仮想し、PointごとにAnchor Boxesを作成する

- RPNの出力
  各Anchor BoxesをGrand TruthのBoxesと比較し、含まれているものが背景か物体か、どれくらいズレているかを出力：


- PASCAL VOCデータで検証
  R-CNNやFast R-CNNよりmAPスコアが高い
  R-CNN、Fast R-CNNより高速

> 出典： Ren, Shaoqing et al. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." IEEE Transactions on Pattern Analysis and Machine Intelligence 39 (2015): 1137-1149.

## YOLO (V1)
- 物体候補領域の提案とクラス分類を一つのネットワークで処理
- YOLO = You Only Look Once（一回だけ見れば良い）
- Faster R-CNNとは異なり、物体候補領域の提案とクラス分類を異なるネットワークで処理しない

## 利点と欠点
- 利点
  高速な処理
  画像全体を一度に見るから、背景を物体と間違えることがない
  汎化性が高い

- 欠点
  精度はFaster-RCNNに劣る

- 工夫：Grid cell
  Grid cellの概念

S×SのGridsに分割

- 候補領域の提案： 各Gridにおいて、そのGridの真ん中を中心とするBB
B個のBounding Boxを生成

- クラス分類： 各Gridごとに、含む物体のクラスを分類


### ネットワークの出力
各Gridにおける各バウンディングボックスの以下の要素を同時に出力：

中心、高さ、横(x,y,w,h)(x, y, w, h)
(x,y,w,h)

- 信頼度スコア
各クラスに対応する特徴マップ

例： S=7S = 7
S=7、B=2B = 2
B=2、クラス数 = 20の場合

- YOLOが最速
  APスコアは従来手法より高いわけではない

> 出典： Redmon, Joseph et al. "You Only Look Once: Unified, Real-Time Object Detection." 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016): 779-788.

YOLO (V1) の損失関数（Loss）

- YOLOは以下の損失関数を基に学習を行う：
損失関数の構成要素

- 予測・真値Bounding Boxの中心位置(x,y)(x, y)
(x,y)の2乗誤差

- 位置ずれが小さいほど損失も小さくなる


- 予測・真値Bounding Boxの幅・高さ(w,h)(w, h)
(w,h)の2乗誤差

- 大きさが近いほど損失も小さくなる


## Grid cellに物体が存在する場合の予測・真値Bounding Boxの信頼度スコアの2乗誤差

真値の信頼度スコアは、予測・真値Bounding BoxのIoUとなる
ii
i番目のGrid cellに物体があり、かつjj
j番目の予測・真値Bounding BoxのIoUが最も高い時に1となる。それ以外は0

- Grid cellに物体が存在しない場合の予測・真値Bounding Boxの信頼度スコアの2乗誤差

- 真値の信頼度スコアは0となる
Grid cellに物体が存在する場合の、物体のクラスの予測確率の2乗誤差

> 注： IoU（Intersection over Union）は2つの領域間の重なり度合いを表す指標。0-1の値をとり、重なり度合いが大きくなるほど1に近づく。

 YOLO (V1) と SSD の損失関数の比較

- 基本的な思想
YOLO（V1）とSSDの基本的な思想は同じ
両者とも、予測・真値Bounding boxの位置と大きさのずれと、予測クラスの確率の誤差を損失としている

### 主な違い

- Bounding boxの形状：
YOLO： Bounding boxは形状が固定されておらず、学習の過程で形状が最適化される
SSD： 形状が固定された複数のBounding boxを使用

- スケール：
SSD： マルチスケールな特徴マップを考慮
YOLO： シングルスケール

- 損失関数：
YOLO： クラスの予測確率の2乗誤差を損失
SSD： クロスエントロピーかつ背景クラスに対する予測確率の損失も考慮

## まとめ
Faster R-CNN

CNNベースのRPNを提案し、End-To-Endなネットワークを可能にした
従来手法のSelective Searchによる処理を改良し、処理を高速化

YOLO

候補領域の提案とクラス分類を同時に行うネットワークの提案
リアルタイム処理に適した高速な物体検出手法
# Transformer
([目次に戻る](#目次))

## Seq2seqとは？
系列(Sequence)を入力として、系列を出力するもの

### Encoder-Decoderモデルとも呼ばれる
入力系列がEncode(内部状態に変換)され、内部状態からDecode(系列に変換)する
実応用上も、入力・出力共に系列情報なものは多い

- 翻訳 (英語→日本語)
- 音声認識 (波形→テキスト)
- チャットボット (テキスト→テキスト)

- RNNとは
系列データを読み込むために再帰的に動作するNN

- 再帰的とは？
RNN
xt → [RNN] → ht
入力     出力

- 時間軸方向への展開
  再帰処理は時間軸方向に展開できる
  x0 → [RNN] → h0 → x1 → [RNN] → h1 → x2 → [RNN] → h2 → x3 → [RNN] → h3
  前の時刻の出力を現在の時刻の入力にする

- 系列情報の処理
  系列情報を舐めて内部状態に変換できる

  今日 → [RNN] → h1
  は  → [RNN] → h2  
  良い → [RNN] → h3
  天気 → [RNN] → h

- Encoder RNN
翻訳元の文を読み込み、実数値ベクトルに変換

- Decoder RNN
実数値ベクトルから、翻訳先の言語の文を生成

出力が再び入力になるRNN x 言語モデル
- RNNは系列情報を内部状態に変換することができる
- 文章の各単語が現れる際の同時確率は、事後確率で分解で
きる
- したがって、事後確率を求めることがRNNの目標にな
る
- 言語モデルを再現するようにRNNの重みが学習されてい
れば、ある時点の次の単語を予測することができる
- 先頭単語を与えれば文章を生成することも可能

---

## Transformer

### Attention

* 注目すべき部分とそうでない部分を学習して決定していく機構です。
  - query(検索クエリ)に一致するkeyを索引し、対応するvalueを取り出す操作であると見做すことができます。これは「辞書オブジェクト」の機能と同じです。

### Transformer

* 2017年6月に登場しました。
* Attention機構のみを利用したモデルです。RNNを使わず、当時のSOTAをはるかに少ない計算量で実現しました。
* **構造**：EncoderとDecoderに分かれており、どちらもSelf Attention機構を取り入れています。
* RNNを利用していないため、はじめに位置情報を追加する処理を入れています。

### Attension例

* **Source Target Attention（ソース・ターゲット注意機構）**
    * 情報がソースとターゲットに分かれています。
    * 受け取った情報に対して近い情報のものをベクトルとして取り出します。
* **Self-Attention（自己注意機構）**
    * Query, Key, Value全てをソースとして受け取っています。
    * 入力を全て同じにして学習的に注意箇所を決めていきます。

### 計算機構

* **Position-Wise Feed-Forward Networks**：位置情報を保持したまま順伝播させます。
* **Scaled dot product attention**：全単語に関するAttentionをまとめて計算します。
    $$
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$
* **Multi-Head attention**：重みパラメータの異なる8個のヘッドを使用します。
    * 8個のScaled Dot-Product Attentionの出力をConcatします。
    * それぞれのヘッドが異なる種類の情報を収集します。
* **Add & Norm**
    * **Add (Residual Connection)**
        * 入出力の差分を学習させます。
        * 実装上は出力に入力をそのまま加算するだけです。
        * 効果: 学習・テストエラーの低減。
    * **Norm (Layer Normalization)**
        * 各層においてバイアスを除く活性化関数への入力を平均0、分散1に正則化します。
        * 効果: 学習の高速化。
* **Position Encoding**
    * RNNを用いないので単語列の語順情報を追加します。

# BERT
([目次に戻る](#目次))

- Bidirectional Transformerをユニットにフルモデルで構成したモデル
- 事前学習タスクとして、マスク単語予測タスク、隣接文判定タスクを与える
- BERTからTransfer Learningを行った結果、8つのタスクでSOTA達成
- Googleが事前学習済みモデルを公開済み（TensorFlow / PyTorch）

## 事前学習のアプローチ

* **Feature-based**
    * 特徴量抽出器として活用するためのものです。
    * 様々なNLPタスクの素性として利用され、N-gramモデルやWord2Vecなど文や段落レベルの分散表現に拡張されたものもあります。最近ではElMoが話題になりました。
* **Fine-tuning**
    * 言語モデルの目的関数で事前学習します。
    * 事前学習の後に、使いたいタスクの元で教師あり学習を行います (すなわち事前学習はパラメータの初期値として利用されます)。

## BERTのアプローチ

* **双方向Transformer**
    * tensorを入力としtensorを出力します。
    * モデルの中に未来情報のリークを防ぐためのマスクが存在しません。
        * 従来のような言語モデル型の目的関数は採用できません (カンニングになるため)。
        * 事前学習タスクにおいて工夫する必要があります。
* **事前学習 (Pre-training) タスク**
    * **空欄語予測 (Masked Language Model)**
        * 文章中の単語のうちランダムにマスクされます。マスクされた単語が何かを予測します。
    * **隣接文予測 (Next Sentence Prediction)**
        * 二つの文章を入力として、隣接文であるかのT/Fを出力します。
* **事前学習 (Pre-training) 手続き**
    * データセット：BooksCorpus(800MB) + English Wikipedia(2500MB)
    * 入力文章の合計系列長が512以下になるように2つの文章をサンプリングします。
    * バッチサイズ：256 (= 256x512系列 = 128,000単語/バッチ)
        * 1,000,000ステップ = 33億の単語を40エポック学習します。
    * Adam：LR=$1e^{-4}$、L2weight\_decay=0.01
    * Dropout：0.1
    * 活性化関数：GeLu（ReLUに似た関数）
* **Fine-tuning (転移学習)**
    * **系列レベルの分類問題**
        * 固定長の分散表現は最初の`[CLS]`トークンの内部表現から得られます。
        * 新しく追加する層は分類全結合層+ソフトマックス層のみです。
    * Fine-tuning自体は高速にできるので、ハイパーパラメータ探索も可能です。



# GPT
([目次に戻る](#目次))

- 巨大な文章のデータセット（コーパス）を用いて事前学習（pre-trained）
- 汎用的な特徴量を習得済みで、転移学習（transfer learning）に使用可能
- 転移学習を活用すれば、手元にある新しいタスク（翻訳や質問応答など）に特化したデータセットの規模が小さくても、高精度な予測モデルを実現できる
- 転用する際にはネットワーク（主に下流）のファインチューニングを行う
- 代表的な事前学習モデルはBERTやGPT-nのモデルであり、事前学習と転移学習では全く同じモデルを使うことが特徴的
- 汎用的な学習済み自然言語モデルは、オープンソースとして利用可能なものもある

## GPT-3原論文： 「Language Models are Few-Shot Learners」
https://arxiv.org/abs/2005.14165
GPT-nモデル - GPTの特徴
GPT（Generative Pre-Training）

2019年にOpenAIが開発した有名な事前学習モデル
その後、GPT-2、GPT-3が相次いで発表
パラメータ数が桁違いに増加

- パラメータ数の進化
特に、GPT-3のパラメータ数は1750億個にもなり、約45TBのコーパスで事前学習を行う

### 参考文献
引用： OpenAIのブログ, https://towardsdatascience.com/gpt-3-the-new-mighty-language-model-from-openai-a74ff35346fc

## GPTの仕組み
  - GPTの構造はトランスフォーマーを基本とし、「ある単語の次に来る単語」を予測し、自動的に文章を完成できるように、教師なし学習を行う
  - 出力値は「その単語が次に来る確率」

具体例
例えば、単語系列 "After"、"running"、"I"、"am"、の次に来る単語の確率が：

"tired": 40%
"hot": 30%
"thirsty": 20%
"angry": 5%
"empty": 5%

になったと仮定すると、"tired"や"hot"が可能性の高い、"angry"や"empty"は低い

## 学習プロセス

- 学習前の1750億のパラメーターはランダムな値に設定され、学習を実行後に更新される
- 学習の途中で誤って予測をした場合、誤りと正解の間の誤差を計算しその誤差を学習する

## GPT-3について報告されている問題点
社会の安全に関する課題

「人間らしい」文章を生成する能力を持つため、フェイクニュースなどの悪用のリスクがある
現在は完全オープンソースとして提供されておらず、OpenAIへAPIの利用申請が必要
参考： https://openai.com/blog/openai-api/

### 学習や運用のコストの制約

膨大な数のパラメータを使用したGPT-3の事前学習を現実的な時間で実現するためには、非常に高性能なGPUを必要とする

### 機能の限界（人間社会の慣習や常識を認識できないことに起因）
生成した文章の文法が正しくても、違和感や矛盾を感じることがある
「物理現象に関する推論」が苦手（例：「アイスを冷凍庫に入れると硬くなりますか」には答えにくい）

## GPT-3のモデルサイズ・アーキテクチャー・計算量
- 計算量の単位
petaflop/s-days = 1秒に1ペタ（101510^{15}
1015）回の演算操作を1日分実施した計算量

- 学習時間

毎秒1ペタ回の演算を行う場合、GPT-3 175Bの学習には数年以上を要する
一般的に"GPT-3"と呼ばれているGPT-3 175Bのパラメータ数は1750億
訓練に要する計算量がモデルサイズとともに増加

## GPTの事前学習
- 最適化目標
事前学習では、以下の式を最大化するように学習する：
$$
U = \sum_i \log P(w_i \mid w_{i-k}, \ldots, w_{i-1}; \Theta)
$$

**記号の説明**

- $U$：言語データセット（ラベルなし）で、$\{w_1, w_2, \ldots, w_n\}$ の中はそのデータセットの一つ一つの単語を示している
- $k$：コンテキストウィンドウ。単語 $w_i$ を予測するためにその前の単語を何個使うかを示している
- $\Theta$：ニューラルネットワークのパラメーター

**学習プロセス**

文に出てくる単語 $w_i$ を、その前の単語 $w_{i-k}, \ldots, w_{i-1}$ を使って予測し、その単語 $w_i$ と予測する確率を最大化する

**モデル構造**

$$
h_0 = U W_e + W_p
$$

- $W_e$：単語の埋め込み表現
- $W_p$：位置エンコーディングベクトル
- $h_0$：単語の埋め込み表現に位置エンコーディングを足したもの

$$
h_l = \text{transformer\_block}(h_{l-1}) \quad \text{for } l = 1, \ldots, n
$$

- $\text{transformer\_block}$ は transformer の decoder を使う
- $n$：transformer のレイヤーの数
- $h_0$ を入力として入れ、その出力の $h_1$ を次のレイヤーの入力として入れ、次のレイヤーに $h_2$ を...という操作を $n$ 回行う

$$
P(u) = \text{softmax}(h_n W_e^T)
$$

- transformer の出力と埋め込み表現の転置行列をかけたものを softmax 関数に入れ、最終的な出力とする

## GPTとベースTransformerの比較

GPTはデコーダーのみを使うのでデコーダーを比較する

- GPT-1のfine-tuning
  転移学習では、以下の記号を使う：
  - 始まりを表す記号
  - 文と文を区切る記号
  - 終わりを表す記号

# 音声認識
([目次に戻る](#目次))

## 機械学習のための音声データの扱い

### 音声データとAI (1/2)

音声データを処理する能力を持つAIの研究・開発が近年多くなされている背景には、以下の理由があります。

* 利便性の向上
* 業務の生産性の向上
* 他の技術と組み合わさることができる

**活用事例:**

* スマートスピーカー
* 音声アシスタント
* 会議などで使われる自動議事録AI

### 音声データとAI (2/2)

音声認識をタスクとしたデータ分析コンペも多数開催されています。

* **Kaggle Freesound Audio Tagging 2019:**
    短い音声データからギターや犬の鳴き声などタグ付けするタスク。
* **Kaggle BirdCLEF2021: Processing audio data:**
    鳥の鳴き声の音声データから、各鳴き声に対応する鳥の種類を推測するタスク。

$\rightarrow$ 音声データをどうやって扱うか？

### そもそも、音声データとは?

#### 音が聞こえる仕組み

音はどのように耳に伝わるか？

* 物体の振動による空気の振動
* 空気のない宇宙では音は聞こえない

**音が聞こえるイメージ:**

音源 $\rightarrow$ 空気中（振動が伝わる）$\rightarrow$ 空気中 = 音が聞こえる

#### 音波

空気の振動による音の波。ある地点における波形。

**音波の特性:**

* **振幅:** 音の大きさ
* **波長:** 音の高さ

**音の特徴:**

* 振幅が大きい $\rightarrow$ 大きい音
* 波長が大きい $\rightarrow$ 低い音
* 1波長分の時間 = 1周期

#### 周波数と角周波数

**定義:**

* **周波数:** 一秒あたりの振動数（周期数）
* **角周波数:** 周波数を回転する角度で表現

**例:**

1秒間で3回振動 = 周波数は3

**ラジアンの確認:**

* $\pi = 180^\circ$
* $\frac{\pi}{2} = 90^\circ$
* $\frac{\pi}{4} = 45^\circ$

**角周波数の段階的表現:**

* $\frac{\pi}{4}$ ラジアン
* $\frac{\pi}{2}$ ラジアン
* $\pi$ ラジアン

### フーリエ変換の概算(補足)

* 全ての波形は正弦波・余弦波で表せる

$$
f(x) = a + \sum_{n=1}^{\infty} a_n \cos nx + b_n \sin nx
$$

（マクローリン展開より）

* オイラーの公式：$e^{i\theta} = \cos\theta + i\sin\theta$などを用いて、

## フーリエ変換の公式

$$
f(x) = \int_{-\infty}^{\infty} \left[ \int_{-\infty}^{\infty} f(t)e^{-i\omega t} dt \right] e^{i\omega x} d\omega
$$

* $f(x)$：波形
* $\int_{-\infty}^{\infty} f(t)e^{-i\omega t} dt$：ある波の振幅
* $e^{i\omega x} d\omega$：ある周波数の波

（※ $f(t)$はある条件を満たす）

ある波の振幅：
$$
F(\omega) = \int_{-\infty}^{\infty} f(t)e^{-i\omega t} dt
$$

## スペクトログラムの特徴
横軸：時間
縦軸：周波数
輝度：振幅
スペクトル → スペクトログラム（時間軸を追加）

参考： https://ja.wikipedia.org/wiki/%E3%82%B9%E3%83%9A%E3%82%AF%E3%83%88%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%A0

# CTC
([目次に戻る](#目次))
CTC (Connectionist Temporal Classification)

## 音声認識の概要
音声認識 (ASR: Automatic Speech Recognition) とは、

- 入力信号を音声特徴ベクトルに変換、その音声特徴ベクトルの系列から対応する単語列を推定するタスク
- 近年、深層学習技術の進展に伴い音声認識モデルの精度は向上しており、様々な製品に応用

## 基本的な音声認識処理の流れ
音声信号 → [特徴量抽出] → 音声特徴量 → [音声認識モデル] → 認識結果

### 特徴量抽出： 音声信号を認識しやすい表現に変換する

主にフーリエ変換を用いて音声信号を周波数ごとの音の強度に分解
人間の聴覚器官・発声器官の生物学的知見に基づく様々な処理

音声認識モデル： 音声認識結果の候補の確率を計算

「この音声は"天気"と言っている確率が0.8である」というように計算

### 従来の音声認識モデルの構造
音声特徴量 → [音響モデル] → [発音辞書] → [言語モデル] → 認識結果

### 3つの構成要素

#### 音響モデル
音声特徴量と音素列の間の確率を計算するモデル

> 音素：/a/や/i/といった母音、/k/や/s/といった子音から構成される音の最小単位

音を聴いて「これは"あ(/a/)"かな？それとも"い(/i/)"かな？」といった音素ごとの確率を求める

### 発音辞書

音素列と単語との対応を扱うモデル
「おはよう：/o/h/a/y/o」のような単語とその発音（音素列）が記述されたリスト

### 言語モデル

ある単語に対して、その単語が発話される確率を計算
文脈的にその単語が表れやすいかの確率を計算

### 従来手法の問題点

- 実装の複雑さ： 
  3つのモジュールの出力をうまく統合して音声認識結果を出力する「デコーダ」の実装が困難

- モジュール間の統合： 
  複数のモジュールの統合処理が複雑

### End-to-Endモデルの登場
2015年ごろから音響モデル、発音辞書、言語モデルを**1つのDNNで表現するEnd-to-Endモデル（E2Eモデル）**の研究が活発化

利点： 構成がシンプルで比較的簡単に実装可能
CTCはこのようなEnd-to-Endモデルの1つ

### CTC (Connectionist Temporal Classification)
CTCはEnd-to-Endモデルの中でも比較的初期に提案されたモデルで、従来手法のように隠れマルコフモデル（HMM）を使用せずにディープニューラルネットワーク（DNN）だけで音響モデルを構築する手法として提案 [Graves+,2006]。

### CTCにおける重要な発明
ブランク（blank）と呼ばれるラベルの導入
前向き・後ろ向きアルゴリズム（forward-backward algorithm）を用いたDNNの学習

### 前提
CTCでは基本的には音声のような時系列情報を扱うため、RNN（recurrent neural network）やLSTM（Long Short Term Memory）のような時系列を考慮したDNNを用います

### ブランクの導入
RNNに音声系列を入力すると、フレーム数だけ出力が得られます。
例： 8フレームの音声系列をRNNに入力し、出力値（確率）が最も高いラベル（音素）をフレーム毎に出力した結果：
[a, −, −, b, b, −, c, c]
> ここで「−」はブランクを表しています。

### 縮約（Contraction）の手順
- CTCでは、次の手順でフレーム単位のラベル系列を最終的なテキスト系列に変換します：
- 連続して出現している同一ラベルを1つのラベルにまとめる
- ブランク「−」を削除する

例：
[a, −, −, b, b, −, c, c] 
→ [a, −, b, −, c]  (連続ラベルをまとめる)
→ [a, b, c]        (ブランクを削除)
この縮約を関数BB
Bで表すと：

B(a,−,−,b,b,−,c,c)=[a,b,c] B(a, −, −, b, b, −, c, c) = [a, b, c]B(a,−,−,b,b,−,c,c)=[a,b,c]
ブランクを導入する理由
1. 同一ラベルが連続するテキスト系列を表現するため

ブランクが存在しない場合：[a,a,a,a,b,b,b]→[a,b][a, a, a, a, b, b, b] [a, b] [a,a,a,a,b,b,b]→[a,b]に縮約されてしまう

[a,a,b][a, a, b]
[a,a,b]を表現できない

解決策：[a,a,−,a,a,b,b][a, a, −, a, a, b, b]
[a,a,−,a,a,b,b]のようにブランクを挿入


2. 厳密にアライメントを決定させないため

音素の境界は曖昧なことが多い
単語の間にはポーズ（間）などの非音声区間も存在
ブランクの出力を許可することで、モデルが無理なアライメント推定を行わず音声認識結果が正解することのみを目的とした学習可能

- 前向き・後ろ向きアルゴリズムを用いたRNNの学習
問題設定
8フレームの音声系列で、最終的なテキスト列が[a,b,c][a, b, c]
[a,b,c]となるようなRNNの出力は複数存在：

[a,−,b,b,b,c,−,−][a, −, b, b, b, c, −, −]
[a,−,b,b,b,c,−,−]
[−,−,a,−,b,b,−,c][−, −, a, −, b, b, −, c]
[−,−,a,−,b,b,−,c]
など

つまり：
B(a,−,b,b,b,c,−,−)=[a,b,c]B(a, −, b, b, b, c, −, −) = [a, b, c]B(a,−,b,b,b,c,−,−)=[a,b,c]
B(−,−,a,−,b,b,−,c)=[a,b,c]B(−, −, a, −, b, b, −, c) = [a, b, c]B(−,−,a,−,b,b,−,c)=[a,b,c]

- 事後確率の計算
入力音声系列xx
xに対して縮約後の出力テキスト系列がl=[a,b,c]l = [a, b, c]
$$
l = [a, b, c] \text{ となる事後確率は：}
$$

$$
P(l|x) = P([a, -, b, b, b, c, -, -]|x) + P([-, -, a, -, b, b, -, c]|x) + \cdots
$$

$$
= \sum_{\pi \in B^{-1}(l)} P(\pi|x) \tag{1}
$$
ここで：

P([a,−,b,b,b,c,−,−]∣x)P([a, −, b, b, b, c, −, −]|x)
P([a,−,b,b,b,c,−,−]∣x)：入力xx
xに対してRNNの（縮約前の）出力が[a,−,b,b,b,c,−,−][a, −, b, b, b, c, −, −]
[a,−,b,b,b,c,−,−]となる確率

B−1(l)B^{-1}(l)
B−1(l)：「縮約するとテキスト系列になるような縮約前のラベル系列の集合」


### 損失関数
CTCにおいて最小化すべき損失関数は：
LCTC=−log⁡P(l∗∣x)(2)L_{CTC} = -\log P(l^*|x) \tag{2}LCTC​=−logP(l∗∣x)(2)
ここでl∗l^*
l∗は正解テキスト系列です。

### 確率計算の詳細
P(l∗∣x)P(l^*|x)
P(l∗∣x)を計算する際には、以下のようなグラフを考えます：

フレーム: 1  2  3  4  5  6  7  8
ラベル:   a  −  −  b  b  −  c  c  (赤パス)
         −  −  a  −  b  b  −  c  (青パス)
各パスの確率は、パス上で各ラベルが出力される確率の積として計算：
$$
P([a,−,b,b,b,c,−,−]\,|\,x) = y_1^a \times y_2^{-} \times y_3^b \times y_4^b \times y_5^b \times y_6^c \times y_7^{-} \times y_8^{-} \tag{3}
$$
ここで $y_t^k$ は、フレーム $t$ でラベル $k$ が出力される確率を表します。

一般的な計算式は次の通りです：

$$
P(l^*\,|\,x) = \sum_{\pi \in B^{-1}(l^*)} P(\pi\,|\,x) = \sum_{\pi \in B^{-1}(l^*)} \prod_{t=1}^T y_t^{\pi_t} \tag{4}
$$

ここで：

TT
T：フレーム数

πt\pi_t
πt​：パスπ\pi
πのフレームtt
tにおけるラベル

### 効率的な計算：前向き・後ろ向きアルゴリズム
縮約してl∗=[a,b,c]l^* = [a, b, c]
l∗=[a,b,c]となるパスは大量にあるため、全てを愚直に計算するのは非効率です。そのため実際のCTCでは、**前向き・後ろ向きアルゴリズム（forward-backward algorithm）**が用いられます。

### パスの分類
π∈B−1(l∗)\pi \in B^{-1}(l^*)
π∈B−1(l∗)なるパスを、あるフレームtt
tにおいてどの頂点を通るかに注目して分類します。

### 確率の和の法則により：
P(l∗∣x)=∑π∈B−1(l∗)P(π∣x)=∑s=1∣l′∣∑π∈B−1(l∗),πt=4=ls′P(π∣x)(5)P(l^*|x) = \sum_{\pi \in B^{-1}(l^*)} P(\pi|x) = \sum_{s=1}^{|l'|} \sum_{\pi \in B^{-1}(l^*), \pi_{t=4} = l'_s} P(\pi|x) \tag{5}P(l∗∣x)=π∈B−1(l∗)∑​P(π∣x)=s=1∑∣l′∣​π∈B−1(l∗),πt=4​=ls′​∑​P(π∣x)(5)
ここでl′l'
l′は
拡張ラベルと呼ばれ、正解ラベル系列にブランクを挿入した系列を表します。


# DCGAN
([目次に戻る](#目次))

### GANについて

- GANの構造
  ミニマックスゲームと価値関数

- GANの最適化方法
  本物のようなデータを生成できる理由

- DCGANについて
  具体的なネットワーク構造

###  応用技術の紹介

- GAN(Generative Adversarial Nets)とは
  生成器と識別器を競わせて学習する生成&識別モデル

> Generator: 乱数からデータを生成
> Discriminator: 入力データが真データ（学習データ）であるかを識別
## 2プレイヤーのミニマックスゲームとは？

* 1人が自分の勝利する確率を最大化する作戦を取る。
* もう一人は相手が勝利する確率を最小化する作戦を取る。

* GANでは価値関数$V(D,G)$に対し、$D$が最大化、$G$が最小化を行う。
$$
\min_G \max_D V(D,G)
$$
$$
V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

* バイナリークロスエントロピーと似ていますか？
$$
L = -\sum y \log \hat{y} + (1-y)\log(1-\hat{y})
$$

---

## GANの価値関数はバイナリークロスエントロピー

* 単一データのバイナリークロスエントロピー
$$
L = -y \log \hat{y} + (1-y)\log(1-\hat{y})
$$
    * $y$: 真値（ラベル）
    * $\hat{y}$: 予測値（確率）

* 真データを扱う時: $y=1, \hat{y}=D(x) \implies L = -\log D(x)$
* 生成データを扱う時: $y=0, \hat{y}=D(G(z)) \implies L = -\log(1 - D(G(z)))$

* 2つを足し合わせる
$$
L = (-\log D(x)) + [\log(1-D(G(z)))]
$$
$$
V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

* 複数データを扱うために期待値を取る
* 期待値: 何度も試行する際の平均的な結果値 $\sum xP(x)$

## 最適化方法

* **Generatorのパラメータ $\theta_g$ を固定**
    * 真データと生成データを$m$個ずつサンプル
    * $\theta_d$を勾配上昇法(Gradient Ascent)で更新
    $$
    \theta_d \leftarrow \theta_d + \eta \frac{1}{m} \sum_{i=1}^m [\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))]
    $$
    （$\eta$は勾配更新を$k$回更新）

* **Discriminatorのパラメータ $\theta_d$ を固定**
    * 生成データを$m$個ずつサンプル
    * $\theta_g$を勾配降下法(Gradient Descent)で更新
    $$
    \theta_g \leftarrow \theta_g - \eta \frac{1}{m} \sum_{i=1}^m [\log(1 - D(G(z^{(i)})))]
    $$
    （$\eta$を1回更新）
## なぜGeneratorは本物のようなデータを生成するのか？

* 生成データが本物のような状況とは
    * $p_g = p_{data}$ であるはず

* 価値関数が $p_g = p_{data}$ の時に最適化されていることを示せばよい
$$
\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

* 二つのステップにより確認する
    1. $G$を固定し、価値関数が最大値を取るときの$D(x)$を算出
    2. 上記の$D(x)$を価値関数に代入し、$G$が価値関数を最小化する条件を算出

---

## ステップ1: 価値関数を最大化する$D(x)$の値は？

* Generatorを固定
$$
V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$
$$
= \int_x p_{data}(x) \log D(x) dx + \int_z p_z(z) \log(1 - D(G(z))) dz
$$
$$
= \int_x p_{data}(x) \log D(x) + p_g(x) \log(1 - D(x)) dx
$$
$y=D(x), a=p_{data}(x), b=p_g(x)$と置けば
$$
a \log(y) + b \log(1-y)
$$

$a \log(y) + b \log(1-y)$の極値を求めよう。

---

## ステップ1: 価値関数を最大化する$D(x)$の値は？

$a \log(y) + b \log(1-y)$ を $y$ で微分
$$
\frac{a}{y} - \frac{b}{1-y} = 0
$$
$$
\frac{a}{y} = \frac{b}{1-y}
$$
$$
a(1-y) = by
$$
$$
a - ay = by
$$
$$
a = (a+b)y
$$
$$
y = \frac{a}{a+b}
$$

$y = D(x), a = p_{data}(x), b = p_g(x)$ なので
$$
D(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}
$$

* 価値関数が最大値を取るときの$D(x)$が判明

## ステップ2: 価値関数はいつ最小化するか？

* 価値関数の $D(x)$ を $\frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$ で置き換え
$$
V = \mathbb{E}_{x \sim p_{data}(x)} \left[ \log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} \right] + \mathbb{E}_{x \sim p_g(x)} \left[ \log \left( 1 - \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} \right) \right]
$$
$$
= \mathbb{E}_{x \sim p_{data}(x)} \left[ \log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} \right] + \mathbb{E}_{x \sim p_g(x)} \left[ \log \frac{p_g(x)}{p_{data}(x) + p_g(x)} \right]
$$

* 二つの確率分布がどれぐらい近いのか調べる必要がある
* 有名な指標としてJSダイバージェンスがある
$$
JS(p_1 || p_2) = \frac{1}{2} (\mathbb{E}_{x \sim p_1} [\log \frac{p_1}{p_2}] + \mathbb{E}_{x \sim p_2} [\log \frac{p_2}{p_1}])
$$
* JSダイバージェンスは非負で、分布が一致する時のみ0の値を取る

---

## ステップ2: 価値関数はいつ最小化するか？

* 価値関数を変形
$$
V = \mathbb{E}_{x \sim p_{data}(x)} \left[ \log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} \right] + \mathbb{E}_{x \sim p_g(x)} \left[ \log \frac{p_g(x)}{p_{data}(x) + p_g(x)} \right]
$$
$$
= \mathbb{E}_{x \sim p_{data}(x)} \left[ \log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} \right] + \mathbb{E}_{x \sim p_g(x)} \left[ \log \frac{p_g(x)}{p_{data}(x) + p_g(x)} \right] - 2 \log 2 + 2 \log 2
$$
$$
= 2 JS(p_{data} \ || \ p_g) - 2 \log 2
$$

* $\min_G V$ は $p_{data} = p_g$ のときに最小値となる ($-2 \log 2 \approx -1.386$)
* GANの学習により$G$は本物のようなデータを生成できる
## GANの問題点

* **モード崩壊 (mode collapse)**
    * 識別器をだましやすい特定の画像形式（モード）を学習
    * 生成画像の多様性が消滅


* **学習が不安定**
    * 勾配消失問題

* **意味のない損失**
    * 識別器と生成器は学習ごとに更新される
    * 学習ごとの損失関数と画像品質に相関なし

---

## Wasserstein GAN (WGAN)

Wasserstein GAN (WGAN) は、分布間の距離を定義するWasserstein DistanceをGANの損失関数に導入して安定な学習を可能にします。

* **GANの目的**


* **従来のGANの損失関数**
    * Jensen-Shannon (JS) divergence
    * 勾配消失問題

* **WGANの損失関数**
    * Wasserstein Distance (Earth-Mover's distance)
    * 有意味な勾配

Wasserstein GAN (2017)  
[https://arxiv.org/abs/1701.07875](https://arxiv.org/abs/1701.07875)

## Wasserstein GAN (WGAN)

Wasserstein GAN (WGAN) の損失関数は、学習に応じて生成画像の質の向上とともに安定して減少します。

* **WGANの損失関数**

    * Wasserstein Distance
        $$
        L_D^{WGAN} = -\mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]
        $$
        $$
        L_G^{WGAN} = -\mathbb{E}_{z \sim p_z}[\log D(G(z))]
        $$

    * K-Lipschitz continuous ($K=1$)
        $$
        \frac{|D(x_1) - D(x_2)|}{||x_1 - x_2||} \leq K
        $$

    * 勾配クリッピング (WGAN)
        $-c \leq W \leq c \quad (c=0.01)$

    * 勾配ペナルティ (WGAN-GP)
        $\nabla D \sim 1$

Wasserstein GAN (2017)  
[https://arxiv.org/abs/1701.07875](https://arxiv.org/abs/1701.07875)

Improved Training of Wasserstein GANs (2017)  
[https://arxiv.org/abs/1704.00028](https://arxiv.org/abs/1704.00028)

## CycleGAN

CycleGANやPix2Pixは、GANを用いて画像様式を変換する生成モデルです。

* **画像変換**: 画像を入力として異なる画像を生成

* **Pix2Pix [1]**
    * ペア画像必要
    * 教師あり学習
    * 一方向変換
    
* **CycleGAN [2]**
    * ペア画像不要
    * 教師なし学習
    * 双方向変換

## CycleGAN

CycleGANは、2つの画像集合間を双方向に変換でき、4つのニューラルネットワーク構造を持ちます。

* **データセット**: 異なる特徴を持つ画像集合 $\{A\}$, $\{B\}$


* **2つの識別器** ($D_A$, $D_B$)

    * **敵対的損失** (adversarial loss)
    $$
    L_{adv}^{D_A} = \mathbb{E}_{a \sim P_{data}[A]} [(D_A(a) - 1)^2] + \mathbb{E}_{b \sim P_{data}[B]} [(D_A(G_B(b)))^2]
    $$

* **2つの生成器** ($G_A$, $G_B$)

    * **敵対的損失** (adversarial loss)
    $$
    L_{adv}^{G_B} = \mathbb{E}_{a \sim P_{data}[A]} [(D_B(G_A(a)) - 1)^2]
    $$

    * **復元損失** (cycle-consistency loss)
    $$
    L_{cyc}^{G_B, G_A} = \mathbb{E}_{a \sim P_{data}[A]} [||G_A(G_B(a)) - a||]
    $$

    * **同一性損失** (identity loss)
    $$
    L_{id}^{G_A} = \mathbb{E}_{a \sim P_{data}[A]} [||G_A(a) - a||]
    $$

# Conditional GAN
([目次に戻る](#目次))

## 問題1
- 敵対的生成ネットワーク（GAN）の一種である条件付き敵対的生成ネットワーク（Conditional GAN，CGAN）について正しいものを選べ。

(a) Conditional GANは，画像生成時に与えられる潜在変数に制約条件を加えることで，従来のGANよりも生成したい画像の質を向上することができる
(b) Conditional GANは，画像生成時に条件パラメータを与え，生成したい画像のクラスを指定できる一方で，従来のGANでは生成する画像のクラスは指定できない
(c) Conditional GANも従来のGANと同様に，Discriminatorが処理するタスクはGeneratorにより生成された画像かそうではないかを識別する分類問題である
(d) Conditional GANは，画像生成時に条件パラメータを与えることで，従来のGANよりも生成したい画像の解像度を向上することができる

- 解答
(b) が正解

- 解説
- Conditional GANの主な特徴は、条件パラメータを与えることで生成したい画像のクラスを指定できること- 従来のGANでは潜在変数ZZZのみから画像を生成するため、どのようなクラスの画像が生成されるかを制御不可
CGANでは条件パラメータYYYを追加することで、特定のクラスの画像を意図的に生成することが可能になります。

## 問題2
- Conditional GANのネットワークとして正しいものを選べ。
ただし，ネットワークはDiscriminator学習時のものとする。
G: Generator
D: Discriminator
X: 真の画像
Y: 条件パラメータ
Z: 潜在変数

- 解答
(a) が正解 ( Discriminatorの前に条件Yを受け取ってる)

- 解説
従来のGANのネットワーク
Z → [G] → G(Z) → [D] → 真偽判定
X ────────────────→ [D] → 真偽判定
従来のGANでは：

## *Generator*: G(Z)G(Z)
G(Z) - 生成分布を近似

生成例（MNIST）: ランダムに生成

## CGANのネットワーク
Z, Y → [G] → G(Z|Y) → [D] ← Y → 真偽判定
X, Y ─────────────────→ [D] ←─── → 真偽判定
CGANでは：

## Generator: G(Z∣Y)G(Z|Y)
G(Z∣Y) -
条件付き生成分布を近似
Discriminatorは画像とその条件パラメータの両方を入力として受け取る
生成例（MNIST）: 条件パラメータで数字（0-9）を指定可能

## 重要なポイント
Generatorは潜在変数ZZ
Zと条件パラメータYY
Yの両方を入力として受け取る

Discriminatorは画像（真の画像XXXまたは生成画像G(Z,Y)G(Z,Y)
G(Z,Y)）と条件パラメータYYYを入力として受け取る

これにより、Discriminatorは「その画像が指定された条件に対して真の画像か生成画像か」を判定する

### 具体例（MNIST）
条件パラメータとして数字のクラス（0-9）を指定することで：

条件パラメータ = 0 → 「0」の画像を生成
条件パラメータ = 1 → 「1」の画像を生成
...
条件パラメータ = 9 → 「9」の画像を生成

> Reference
> GANの提案論文
> Goodfellow, Ian J. et al. "Generative Adversarial Nets." NIPS (2014).
> Conditional GANの提案論文
> Mirza, Mehdi and Simon Osindero. "Conditional Generative Adversarial Nets." ArXiv abs/1411.1784 (2014): n. pag.

# Pix2Pix
([目次に戻る](#目次))
## Pix2pix の概要

（GANの学習後にご視聴ください）

### 簡単なおさらい：GAN（1/2）

* **生成器（Generator）と識別器（Discriminator）を競わせて学習する生成&識別モデル**
    * Generator：乱数からデータを生成
    * Discriminator：入力データが真データであるかどうか識別
* **GeneratorとDiscriminatorのミニマックスゲーム**
    * Generatorは自分の勝利する確率を最大化する
    * DiscriminatorはGeneratorが勝利する確率を最小化する
    * 上記を交互に繰り返す

### 簡単なおさらい：GAN（2/2）

[Image of GANのネットワーク]

* **GANのネットワーク**
    * $Z$: 乱数
    * $G(z)$: 生成されたデータ
    * $x$: 真のデータ
    * $True\ or\ False$: 識別器の出力
    * $\theta_g$: Generatorのパラメータ
    * $\theta_d$: Discriminatorのパラメータ

* **Conditional GAN（条件付きGAN）**
    * GANの生成したいデータに**条件**をつける
    * 条件はラベルで指定 $\rightarrow$ 「犬」という条件で、犬の画像を生成する
    * 基本的なネットワークはGANと同様
* **各プレイヤーの役割（条件ラベル$x$の場合）**
    * Generator：$x$の画像を生成
    * Discriminator：以下のように識別
        * （Gが生成した犬の画像 $G(x, z)$，$x$ラベル）$\rightarrow$ False
        * （Gが生成した犬の画像 $G(x, z)$，$x$以外のラベル）$\rightarrow$ False
        * （真のラベル$x$の画像 $y$，$x$ラベル）$\rightarrow$ True
        * （真のラベル$x$の画像 $y$, $x$以外のラベル）$\rightarrow$ False

### Pix2pix：概要

* **役割**
    * CGANと同様の考え方
    * 条件としてラベルではなく**画像**を用いる
    * 条件画像が入力され、何らかの**変換を施した**画像を出力する
    * 画像の**変換方法**を学習
* **各プレイヤーの役割（条件画像$x$）**
    * Generator：条件画像$x$をもとにある画像 $G(x, z)$を生成
    * Discriminator：
        * (条件画像$x \rightarrow$ Generatorが生成した画像 $G(z|x)$)の変換と
        * (条件画像$x \rightarrow$ 真の変換が施された画像 $y$)の変換が正しい変換かどうか識別する

### Pix2pix：学習データ

条件画像 $x$ と真の何らかの変換が施された画像 $y$ のペアが学習データ

* **例:**
    * 着色
    * RGB化
    * 建物のエッジ抽出

### Pix2pix：ネットワーク

* **Pix2pixのネットワーク**
    * $Z$: 乱数
    * $G(z|x)$: 生成された画像
    * $y$: 真の画像
    * $True\ or\ False$: 識別器の出力
    * $x$: 条件画像
    * $\theta_g$: Generatorのパラメータ
    * $\theta_d$: Discriminatorのパラメータ

### Pix2pix：工夫

**U-Net**

* Generatorに使用
* 物体の位置を抽出
    * 入力 $\rightarrow$ 出力
    * Encoder (ダウンサンプリング)
    * Decoder (アップサンプリング)
    * 画像の特徴 $\rightarrow$ セマンティックセグメンテーション
    * スキップ接続: 物体の位置情報伝達
    * セマンティックセグメンテーション
    * 物体の位置が抽出される
    * ピクセル単位の分類が可能
    * 入力画像と出力画像でサイズは一致

# A3C
([目次に戻る](#目次))
Asynchronous Advantage Actor-Critic（A3C）

## A3Cとは
強化学習の学習法の一つ；DeepMindのVolodymyr Mnih（ムニ）のチームが提案

複数のエージェントが同一の環境で非同期に学習する
"A3C"の名前の由来：3つの"A"

- Asynchronous  
  複数のエージェントによる非同期な並列学習
- Advantage  
  複数ステップ先を考慮して更新する手法
- Actor  
  方策によって行動を選択
- Critic
  状態価値関数に応じて方策を修正

- Actor-Criticとは
  行動を決めるActor（行動器）を直接改善しながら、方策を評価するCritic（評価器）を同時に学習させるアプローチ
> 引用： Sutton, Berto, "Reinforcement Learning – an introduction." 1998

### A3Cによる非同期（Asynchronous）学習の詳細

複数のエージェントが並列に自律的に、rollout（ゲームプレイ）を実行し、勾配計算を行う
その勾配情報をもって、好き勝手なタイミングで共有ネットワークを更新する
各エージェントは定期的に自分のネットワーク（local network）の重みをglobal networkの重みと同期する
共有ネットワーク = パラメータサーバ

> 引用： https://pylessons.com/A3C-reinforcement-learning/
> 参考論文： https://arxiv.org/abs/1602.01783

### 並列分散エージェントで学習を行うA3Cのメリット
① 学習が高速化
② 学習を安定化

#### 安定化について
経験の自己相関が引き起こす学習の不安定化は、強化学習の長年の課題
DQNはExperience Replay（経験再生）機構を用いてこの課題を解消

バッファに蓄積した経験をランダムに取り出すことで経験の自己相関を低減
しかし、経験再生は基本的にはオフポリシー手法でしか使えない


A3Cはオンポリシー手法であり、サンプルを集めるエージェントを並列化することで自己相関を低減することに成功した

### A3Cの難しいところとA2Cについて
- A3Cの課題
  Python言語の特性上、非同期並列処理を行うのが面倒
  パフォーマンスを最大化するためには、大規模なリソースを持つ環境が必要

- A2Cについて
  A3Cの後にA2Cという手法が発表された
  A2Cは同期処理を行い、Pythonでも実装しやすい
  各エージェントが中央指令部から行動の指示を受けて、一斉に1ステップ進行し、中央指令部は各エージェントから遷移先状態の報告を受けて次の行動を指示する
  性能がA3Cに劣らないことがわかったので、その後よく使われるようになった

### A3Cのロス関数
ロス関数の構成

- アドバンテージ方策勾配
- 価値関数ロス
- 方策エントロピー

$$
Total loss=−アドバンテージ方策勾配+α⋅価値関数ロス−β⋅方策エントロピー\text{Total loss}
$$
$$
= -\text{アドバンテージ方策勾配} + \alpha \cdot \text{価値関数ロス} - \beta \cdot \text{方策エントロピー}Total loss=−アドバンテージ方策勾配+α⋅価値関数ロス−β⋅方策エントロピー
$$ 
βはハイパーパラメータ
βは探索の度合いを調整するハイパーパラメータ

### 分岐型Actor-Criticネットワーク
- 一般的なActor-Critic
  方策ネットワークと価値ネットワークを別々に定義し、別々のロス関数（方策勾配ロス/価値ロス）でネットワークを更新

- A3C（パラメータ共有型のActor-Critic）
  1つの分岐型のネットワークが、方策と価値の両方を出力し、たった1つの「トータルロス関数」でネットワークを更新
- A3Cアルゴリズムのアドバンテージ方策勾配項

### 方策勾配法の基本
- 方策勾配法では、θ
- θをパラメータに持つ方策πθ
- πθ​に従ったときの期待収益ρθ
- ρθ​が最大になるように、θ

### θを勾配法で最適化する

方策勾配定理により、パラメータの更新に用いられる勾配∇θρθ\nabla_\theta \rho_\theta
∇θ​ρθ​は、以下の式で表される


- ベースラインの効果
b(s)b(s)
b(s)は価値の「ベースライン」：
これを引くことで、推定量の分散が小さくなり学習の安定化が期待できる

## A3Cの特徴
式1の中のQπθ(s,a)−b(s)Q^{\pi_\theta}(s, a) - b(s)
Qπθ​(s,a)−b(s)にアドバンテージ関数を設定する

### A3Cのアルゴリズムの特徴としては、勾配を推定する際に：

b(s)b(s)
b(s)の推定には価値関数Vπθ(s)V^{\pi_\theta}(s)
Vπθ​(s)を使用

Q(s,a)Q(s,a)
Q(s,a)の指定には、kk
kステップ先読みした収益（式2）を用いる
つまり、式3の期待値が式1の勾配と等価

### A3CのAtari 2600における性能検証

- 実験設定

Atari 2600において、人間のスコアに対して規格化した深層強化学習の結果
A3C：16 CPUコアのみ使用、GPU使用なし（1~4日間の訓練）
他のエージェント：Nvidia K40 GPUを使用（8~10日間の訓練）

- 結論：より短い訓練時間でGPUなしでも、A3Cのスコアが顕著に高い
手法平均中央値訓練時間A3C高い性能高い性能1~4日
1日間の訓練で、CPUのみでも、Dueling D-DQNの精度に到達

参考： https://arxiv.org/pdf/1602.01783

- A3Cアルゴリズムの方策エントロピー項
  - 方策のランダム性の高い（= エントロピーが大きい）方策にボーナスを与えることで、方策の収束が早すぎて局所解に停滞する事態を防ぐ効果がある
  - 方策エントロピー項の追加は、方策関数の正則化効果が期待できる

- 方策のランダムさの指標
  例： ある状態ss
  sの入力について出力される行動の採用確率が


(a1,a2,a3,a4)=[0.25,0.25,0.25,0.25](a_1, a_2, a_3, a_4) = [0.25, 0.25, 0.25, 0.25]
(a1​,a2​,a3​,a4​)=[0.25,0.25,0.25,0.25]の場合

(a1,a2,a3,a4)=[0.85,0.05,0.05,0.05](a_1, a_2, a_3, a_4) = [0.85, 0.05, 0.05, 0.05]
(a1​,a2​,a3​,a4​)=[0.85,0.05,0.05,0.05]の場合

**前者のほうが方策のエントロピーが大きい状態である**

A3Cの実装の参考（TensorFlow Blogより）
https://blog.tensorflow.org/2018/07/deep-reinforcement-learning-keras再試行Claudeは間違えることがあります。回答内容を必ずご確認ください。リサーチbeta Sonnet 4

## 距離学習 (Metric learning)
([目次に戻る](#目次))

ディープラーニング技術を用いた距離学習は、人物同定 (Person Re-Identification) をはじめ、顔認識、画像分類、画像検索、異常検知など幅広いタスクに利用される技術

この解説プリントでは、距離学習の基本的なアイディア、そして深層距離学習の代表的な手法である Siamese Network [Hadsell+, “Dimensionality Reduction by Learning an Invariant Mapping”, CVPR (2006)]、および、Triplet Network [Hoffer+,“Deep metric learning using Triplet network”, arXiv:1412.6622 (2014)] について説明していきます。

### 距離学習とは

- 距離学習ではデータ間の **metric**、すなわち「データ間の距離」を学習
-データ間の距離を適切に測ることができれば、距離が近いデータ同士をまとめてクラスタリング$^1$ができた り、他のデータ要素から距離が遠いデータを異常と判定することで異常検知したりと様々な応用が可能となります。距離学習自体は古くからある手法ですが、近年のディープラーニングの発展とともに、ディープラーニング技術を利用した距離学習の手法が数多く提案されています。このような手法は、特に**深層距離学習** (deep metric learning) と呼ばれています。

一般に、画像や音声などの多次元データはニューラルネットワークを用いることにより、**次元削減** (データ圧縮) することが出来ます。例えば、畳み込みニューラルネットワーク (Convolutional Neural Network: CNN) を用いた画像分類の場合、$28 \times 28$ サイズの画像をニューラルネットワーク (CNN) に入力することで $10$ 次元のベクトルに圧縮することができます。つまり、元々 $28 \times 28 = 784$ 個の数値で表現されていた画像の情報を $10$ 個の変数に圧縮したと解釈できます。

CNN から出力されるベクトルは、入力データの重要な情報を抽出したベクトルと考えられるため、一般に**特徴ベクトル** (feature vector) と呼ばれます。

入力      出力
入力データ -> CNN -> 特徴ベクトル
次元    10次元


ただし、未学習のネットワークにデータを入力しても出力されるのは無意味なベクトルであり、特徴ベクトルに意味を持たせるには何らかの手法で CNN を学習する必要があります。深層距離学習はその1つの方法であり、2つの特徴ベクトル間の距離がデータの類似度を反映するようにネットワークを学習します。具体的には、

* 同じクラスに属する (= 類似) サンプルから得られる特徴ベクトルの距離は小さくする
* 異なるクラスに属する (= 非類似) サンプルから得られる特徴ベクトルの距離は大きくする

といった具合です。特徴ベクトルの属する空間は**埋め込み空間** (embedding space)$^2$と呼ばれますが、この埋め込み空間内で類似サンプルの特徴ベクトルは近くに、非類似サンプルの特徴ベクトルは遠くに配置されるように学習を行うことになります。つまり、CNN を深層距離学習の手法を用いて学習していくと、類似サンプル (から得られる特徴ベクトル) は埋め込み空間内で密集していき、逆に非類似サンプルは離れる

入力       学習
入力データ -> CNN -> 埋め込み空間 (学習前)
|
V
損失関数
|
V
埋め込み空間 (学習後)

- この図では、特徴ベクトルが $2$ 次元のベクトル $(x_1, x_2)$ であるとして、特徴ベクトルを埋め込み空間上の点として可視化
- 学習後のような埋め込み空間を構成することができれば、画像分類やクラスタリングが容易になるのは一目瞭然
- 画像分類などのタスクに対する精度の向上は、ニューラルネットワークモデル自体の構造を工夫 (複雑化) することでも達成可
- ネットワークの構造自体は工夫しなくても、類似度を反映した埋め込み空間を構成できるように学習を行うだけで精度向上が可能になる、というのが深層距離学習のアプローチとなります。

では、深層距離学習ではどのようにしてサンプル間の類似度を反映した埋め込み空間を構成するのでしょうか？ 以下では、深層距離学習の代表的な手法である Siamese network と Triplet network について説明してきます。
## Deep Metric Learning

## Siamese Network
- 入力：2つのデータ
- 同一クラスなら距離を縮め、異なるクラスなら距離を離す
- **Contrastive Loss** を使用

## Triplet Network
- 入力：Anchor + Positive + Negative
- **Triplet Loss**：
  - $D(a, p) + \text{margin} < D(a, n)$ を満たすように学習
  - より安定した**埋め込み空間**の学習可能

# MAML(メタ学習)
([目次に戻る](#目次))

## MAMLが解決したい課題

* 訓練に必要なデータ量が多い
    * 人手のアノテーションコスト
    * データ自体を準備できるかどうか
* 少ないデータの問題点
    * 過学習が発生しやすい
    $\rightarrow$ 少ないデータで学習させたい

深層学習モデルの開発に必要なデータ量を削減したい

### 深層学習モデル開発に必要なデータ量

| データセット名        | 画像の枚数 |
| :-------------------- | :--------- |
| MNIST                 | 7万枚      |
| ImageNet (ILSVRC2012) | 約120万枚  |
| Open Image Dataset V6 | 約900万枚  |
| MegaFace              | 約570万枚  |

## MAMLのコンセプト

タスクに共通する重みを学習し、新しいモデルの学習に活用

* **転移学習 (ファインチューニング)**
    * **事前学習**
        * オートエンコーダー (教師なし学習)
        * ImageNetを使った事前学習モデル (教師あり学習)
    * **モデルAの重みを初期値として学習**
        * タスクAのためのモデル
        * タスクAの重みを使用 (この部分のみ学習)

* **MAML (メタ学習)**
    * モデル全体の重みを学習

共通重みからファインチューニング
+-----------------+
| タスクAのためのモデル |
+-----------------+
共通重みからファインチューニング
+-----------------+
| タスクBのためのモデル |
+-----------------+
共通重みからファインチューニング
+-----------------+
| タスクCのためのモデル |
+-----------------+
共通重みからファインチューニング
+-----------------+
| タスクDのためのモデル |
+-----------------+

    * タスク共通の重みを学習

## MAMLの学習手順

タスクごとの学習を行った結果を共通重みに反映させ学習

1.  **共通重み $\theta$ をランダムに初期化**

    **Outer Loop:** $\theta$ が収束するまで繰り返し
    2.  タスク集合 $T$ からタスク $T_i$ を取り出し
        **Inner Loop:** タスクの個数分繰り返し
        3.  タスク $T_i$ に重み $\theta$ を最適化し $\theta'_i$ を得る
    4.  ($\theta'_1, \theta'_2, ...$) を集める
    5.  集めた重みで共通重み $\theta$ を更新 (SGD)

## MAMLの効果

Few-Shot learningで既存手法を上回る精度を実現

Omniglotデータセット(50種類の文字)を使ったクラス分類

|             | 5クラス  | 20クラス |
| :---------- | :------- | :------- |
| サンプル1枚 | (データ) | (データ) |
| サンプル5枚 | (データ) | (データ) |

この他にも、回帰問題、強化学習などでも効果が確認された

C. Finn+, arXiv:1703.03400

## MAMLの課題と対処

* MAMLは計算量が多い
    * タスクごとの勾配計算と共通パラメータの勾配計算の2回が必要
    * 実用的にはInner loopのステップ数を大きくできない
* 計算コストを削減する改良案 (近似方法)
    * First-order MAML: 2次以上の勾配を無視し計算コストを大幅低減
    * Reptile: Inner loopの逆伝搬を行わず、学習前後のパラメータの差を利用

タスクごとの学習と共通パラメータの学習で計算量が多い

# グラフ畳み込み(GCN)
([目次に戻る](#目次))

* **何のために？**
    $\rightarrow$ 用途は様々！（だから概要をつかみにくい...）
    今回は特徴をはっきりさせるため
    **もとの関数にフィルターをかける！**
    $y_t = g_t * x_t$

* **特徴をはっきりさせる？**
    * これを2次元画像に対して使う $\rightarrow$ CNN
    * これをグラフ（ネットワーク）に対して使う $\rightarrow$ GCN

畳み込みで関数の特徴を際立たせている！

### 教科書の数式とは違うように見えるのは？

私たちは...
な式を読めるようになって、活用したい！

しかし、畳み込みについて、書籍を開いても、web上の情報を検索しても
という式しか出てこない...なぜ？

「因果的な時不変システム」を取り扱う際に便利な形式だから！

「過去の出来事が、現在にどれだけ影響しているのか」といったことをモデル化する際に便利！

一般的（より広い意味）に考えると...
$\rightarrow$ 元の関数に「重み」をかけて、強調するところと、捨象するところの区別をするということがキモ！

$t-\tau$ という部分が気になるところだが、より重要なことは『元の関数に「重み」をかけて、強調したり、捨象したりしている』という構造！

ここが「重み」（のようなもの）

**畳み込みの一般的な形**

連続的な場合:
$$
y_t = f * g_t = \int f(\tau) g(t, \tau) d\tau
$$

離散的な場合:
$$
y_m = f * g_m = \sum_n f(n) g(m, n)
$$
重み

**具体例** （$n$は0から4まで変化させる）

$f_n = (0, 2, 4, 6, 8)$

$g_{m, n} = \begin{cases} 1, & m-n = 1 \\ 3, & m-n = 0 \\ 5, & m-n = -1 \\ 0, & \text{上記以外} \end{cases}$

|  $n$  |   0   |   1   |   2   |   3   |   4   |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  $f$  |   0   |   2   |   4   |   6   |   8   |

| $m-n$ |  -1   |   0   |   1   |
| :---: | :---: | :---: | :---: |
|  $g$  |   5   |   3   |   1   |

$m=0: 0 \times 1 + 1 \times 3 + 4 \times 5 = 23$
$m=1: 2 \times 1 + 4 \times 3 + 6 \times 5 = 44$
$m=2: 4 \times 1 + 6 \times 3 + 8 \times 5 = 62$

$y_m = (-, 23, 44, 62, -)$

## Spatialな場合

ノイズが目立たなくなり、形がはっきりする！
空間的の意

### Spatialな場合（どんな手順で？）

$93.8 \times (1/3) + 102.4 \times (1/3) + 96.7 \times (1/3) = 97.6$
このような空間的な操作を繰り返していく

- 重みの係数は用途に合わせて変更可能
- スペクトルに分解することで、特徴的な成分が明らかに！

# Grad-CAM, LIME, SHAP
([目次に戻る](#目次))
## ディープラーニングモデルの解釈性

* ディープラーニング活用の難しいことの1つは「ブラックボックス性」
* 判断の根拠を説明できないという問題があります。
* 実社会に実装する際に「なぜ予測が当たっているのか」を説明できないことが問題となります。
* モデルの解釈性に注目し、「ブラックボックス性」の解消を目指した研究が進められています。

### CAM
（Class Activation Mapping）

* 2016年のCVPRにて発表されたCNNの判断根拠可視化の手法です。
* GAP(Global Average Pooling)を使用するモデルに適用できる手法です。
* GAPは学習の過学習を防ぐ、正則化の役割として使われてきましたが、GAPがCNNが潜在的に注目している部分を可視化できるようにする性質を持っていることが分かりました。

### Grad-CAM
（Gradient-weighted Class Activation Mapping）

* CNNモデルに判断根拠を持たせ、モデルの予測根拠を可視化する手法です。
* 名称の由来は”Gradient” = 「勾配情報」です。
* 最後の畳み込み層の予測クラスの出力値に対する勾配を使用します。
* 勾配が大きいピクセルに重みを増やすことで、予測クラスの出力に大きく影響する重要な場所を特定します。
* CAMはモデルのアーキテクチャにGAPがないと可視化できなかったのに対し、Grad-CAMはGAPがなくても可視化できます。また、出力層が画像分類でなくてもよく、様々なタスクで使えます。

### LIME
(Local Interpretable Model-agnostic Explanations)

* 特定の入力データに対する予測について、その判断根拠を解釈・可視化するツールです。
    * 表形式データ：「どの変数が予測に効いたのか」を可視化します。
    * 画像データ：「画像のどの部分が予測に効いたのか」を可視化します。
* 単純で解釈しやすいモデルを用いて、複雑なモデルを近似することで解釈を行います。
    * 複雑なモデル：人間による解釈の困難なアルゴリズムで作った予測モデル（例：決定木のアンサンブル学習器、ニューラルネットワークなど）。
* LIMEへの入力は1つの個別の予測結果です。
    * 画像データ：1枚の画像の分類結果。
    * 表形式データ：1行分の推論結果。
* 対象サンプルの周辺のデータ空間からサンプリングして集めたデータセットを教師データとして、データ空間の対象範囲内でのみ有効な近似用モデルを作成します。近似用モデルから予測に寄与した特徴量を選び、解釈を行うことで、本来の難解なモデルの方を解釈したことと見なします。
* スーパーピクセル: 色や質感が似ている領域をグループ化すること。

### 参考文献から補足
• 参考：https://github.com/marcotcr/lime
• Pythonを用いたLIMEの実装ライブラリは、データ形式（表形式、テキスト、画像）によって
アルゴリズムが異なる ※コンセプトは同じ
(例) 表データから作成されたモデルは “LimeTabularExplainer”を使用して解釈する
LIMEを用いて、Google のインセプションモジュールによる画像認識を解釈している様子
（画像は論文より引用 ：https://arxiv.org/abs/1602.04938）


### SHAP
（SHapley Additive exPlanations）

* 機械学習モデルの予測結果を解釈するための手法です。
* 各特徴量がモデルの出力にどの程度寄与しているかを定量的に評価します。
* 協力ゲーム理論の概念であるshapley value(シャープレイ値)を機械学習に応用したものです。

# Docker
([目次に戻る](#目次))

## コンテナ型仮想化

仮想環境はハードウェア上で独立した複数の環境を実行する技術で、ホスト型、ハイパーバイザー型、コンテナ型の3種類が存在します。

| カテゴリ             | ホスト型                                                     | ハイパーバイザー型                                                                                                                             | コンテナ型                                                                                     |
| :------------------- | :----------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- |
| **概要**             | 物理的なホスト OS 上で動作                                   | ハードウェア上で直接実行                                                                                                                       | 単一のOSカーネルで複数のコンテナ実行                                                           |
| **利点**             | ⚫ インストールの簡単さ<br>⚫ ホスト OS のデバイスドライバ利用 | ⚫ 異なる OS の同時実行<br>⚫ 高いパフォーマンスと安定性<br>⚫ ハードウェアと直接通信するためのオーバーヘッドが少ない<br>⚫ 完全なゲスト OS の隔離 | ⚫ ハードウェアの効率的な利用<br>⚫ 起動の速さ<br>⚫ リソースのオーバーヘッドの少なさ<br>⚫ 移植性 |
| **欠点**             | ⚫ パフォーマンスのオーバーヘッド<br>⚫ セキュリティ問題の影響 | ⚫ セットアップと管理が複雑<br>⚫ ハードウェアのリソースに直接依存<br>⚫ Type 2 の場合 ホスト OS のオーバーヘッドが存在                           | ⚫ 完全な隔離の不足<br>⚫ 特定の OS カーネルへの依存                                             |
| **ソフトウェアの例** | ⚫ VMware Workstation<br>⚫ Oracle VirtualBox                  | ⚫ VMware vSphere<br>⚫ Microsoft Hyper-V                                                                                                        | ⚫ Docker<br>⚫ LXC                                                                              |
| **備考**             | なし                                                         | **Type 1とType 2の違い:**<br>Type 1:<br>⚫ 高いパフォーマンスと安定性<br>Type 2<br>⚫ ホスト OS 上で動作<br>⚫ 設定の簡単さ                       | なし                                                                                           |

## Dockerとは

Dockerはコンテナ型仮想化ソリューションで、事実上の標準として利用されています。

* **Docker の定義と背景**
    * アプリケーションと依存関係をコンテナとしてパッケージ化
    * どの環境でも一貫した動作を保証
    * アプリケーションの依存関係をインフラから分離
* **Docker のアーキテクチャ**
    * **Docker エンジン**: コア技術で、Linuxの`cgroups`、`namespaces`を利用
    * **Docker イメージ**: アプリケーションと依存関係のスナップショット
    * **Docker コンテナ**: イメージから生成されるアプリケーションの実行環境
* **Docker の歴史と普及の背景**
    * 2013年: Docker Engineとして公開
    * Microsoftとの協力でWindows Serverに導入
    * 現在: データセンター、クラウドプロバイダーでの採用

Dockerイメージに封入された実行環境で作成したアプリが実行される

## Dockerの用途

Dockerは、アプリケーションの開発からデプロイメントまでの一貫性と効率性を向上させるために広く利用されています。

* **アプリケーションの開発とテスト**
    * 現代の開発の複雑さへの対応
    * Dockerによるワークフローの簡素化と加速
* **本番環境でのデプロイ**
    * コンテナの概念を導入することによる環境の隔離
    * 2013年のDocker導入以降の業界標準化
* **マイクロサービスアーキテクチャの実装**
    * 各マイクロサービスの独立したデプロイ
    * システムの柔軟性と耐障害性の向上
* **環境の統一と再現性の確保**
    * アプリケーションと依存関係のコンテナ化
    * 開発、テスト、本番環境の動作の一貫性

[Image of Dockerの用途における開発から本番環境までのワークフロー]
※ 実際には、ステージング環境・本番環境のDockerイメージはクラウド上でビルドすることがあります

## 基本的な Docker コマンド

Dockerの基本コマンドは、コンテナの作成、運用、管理を効率的に行うための不可欠です。

* **主要なDockerコマンド**
    * `docker build`: Dockerfileからイメージの作成（ソースコードやアプリケーションの依存関係を含むイメージ作成）
    * `docker run`: コンテナの起動（指定されたイメージから新しいコンテナを作成・実行、オプションや引数でカスタマイズ可能）
    * `docker ps`: 実行中のコンテナの一覧表示（`-a`オプションで停止中のコンテナも表示）
    * `docker images`: ローカルのイメージ一覧表示（ID、リポジトリ名、タグ、作成日時などの情報表示）
    * `docker pull`: Docker Hubや他のレジストリからイメージ取得（ローカルに存在しないイメージのダウンロード）
    * `docker push`: ローカルのイメージをレジストリにアップロード（自作イメージの公開・共有）
    * `docker start`: 停止しているコンテナの起動（既存の停止コンテナを再起動可能）
    * `docker stop`: 実行中のコンテナの停止（`docker start`で再起動可能）
    * `docker rm`: コンテナの削除（停止しているコンテナをシステムから完全削除、一度削除したコンテナは復元不可）

## Dockerfile の作成方法

Dockerfileは、コンテナイメージの作成手順を定義するための指示書であり、その命令とベストプラクティスを理解します。

* **主要な命令**
    * `FROM`: ベースイメージの指定（例: `FROM ubuntu:20.04`）
    * `RUN`: シェルコマンドの実行（例: `RUN apt-get update && apt-get install -y curl`）
    * `CMD`: デフォルトのコマンド指定（例: `CMD ["echo", "Hello, World!"]`）
    * `ENTRYPOINT`: デフォルトのアプリケーション指定（例: `ENTRYPOINT ["echo"]`）
    * `COPY`: ホストからコンテナへのファイルコピー（例: `COPY ./app /app`）
    * `ADD`: ファイルの追加 (tar解凍やURLからのダウンロードも可)（例: `ADD https://example.com/app.tar.gz /app`）
    * `WORKDIR`: 作業ディレクトリの設定（例: `WORKDIR /app`）
    * `ENV`: 環境変数の設定（例: `ENV MY_ENV_VARIABLE=value`）

* **Dockerfile とは**
    * Dockerコンテナイメージを作成するためのスクリプトファイル
    * ベースイメージの選択、アプリケーションのコードの追加、依存関係のインストール、環境変数の設定などが記述されます。
* **基本構文**
    * 命令は大文字で始まり、引数が続く
    * 例: `INSTRUCTION arguments`
* **ベストプラクティス**
    * `.dockerignore`ファイルを使用して不要なファイルを除外
    * キャッシュを効率的に使用するための配置
    * 不要なパッケージや一時ファイルの削除
    * 複数の`RUN`命令を連鎖させる
    * 公式イメージの使用を推奨

## GPU 環境におけるDockerの使用

Dockerを使用することで、GPUの計算リソースを効率的に活用したアプリケーションの開発とデプロイが簡単に行えます。

* **背景**
    * GPUはディープラーニングや高性能計算に広く利用
    * DockerがGPUサポートを強化
* **Docker での GPU サポート**
    * NVIDIA Container Toolkitを提供
    * コンテナランタイムライブラリやユーティリティを含む
    * Docker、LXC、Podmanなどのサポート
* **GPU アプリケーションのコンテナ化**
    * GPUを活用したアプリケーションをDockerで実行可能
    * 効率的なGPUリソースの利用
* **NVIDIA Container Toolkit の特徴**
    * GPU アクセラレーション: コンテナ内でのNVIDIA GPU直接利用
    * エコシステムのサポート: DockerやPodman等のサポート
    * 自動設定: NVIDIA GPU利用のための自動設定
* **深層学習モデルの開発とDocker**
    * 一貫した環境設定の簡易化
    * モデルの開発やテストの一貫性

※ 容量の大きなデータセットはイメージに含めないことも多い

## コンテナオーケストレーション

コンテナオーケストレーションは、大規模なアプリケーションなどで複数のコンテナ管理を実現するソリューションです。

* **コンテナオーケストレーション**
    * **定義**: コンテナのデプロイ、スケーリング、運用の自動化プロセス
    * **必要性**: 大規模アプリケーション・マイクロサービスの効果的な管理、コンテナダウン時の自動復旧
    * **代表的ツール**: Kubernetes（オーケストレーションニーズ対応）
* **主な機能:**
    * サービスディスカバリーとロードバランシング
    * ストレージオーケストレーション
    * 自動ロールアウト・ロールバック
    * 自動ビンパッキング
    * セルフヒーリング
    * シークレット・設定管理
* **代表的オーケストレーションツール:**
    * **Kubernetes**: オープンソース、大規模運用向け
    * `docker-compose`: Docker公式、開発環境・小規模デプロイ向け

### 参考文献補足
https://qiita.com/caunu-s/items/4fa0e0465ea83fcc06e4
類似のプログラムにpodmanがある
- PodmanとDockerの主な違いは以下の通りです：
デーモンの有無: Podmanはデーモンレスで動作し、Dockerは常駐デーモン（docker daemon）が必要です。 
- セキュリティ: Podmanは「rootless」モードをサポートしており、ユーザー権限でコンテナを実行できますが、Dockerは通常root権限で動作します。 
- コマンドラインインターフェース: PodmanはDockerと互換性のあるCLIを提供しており、ほとんどのDockerコマンドをそのまま使用できます。 

コンテナの管理: Podmanは複数のコンテナを一つのユニットとして管理する「pod」概念を持っていますが、Dockerは個別のコンテナを管理します。 

- 使用ケース: Podmanは特にセキュリティが重視される環境での使用が推奨され、Dockerは広く普及しているため、コミュニティやサポートが充しています
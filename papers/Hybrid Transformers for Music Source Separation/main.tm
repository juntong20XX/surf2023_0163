<TeXmacs|2.1.2>

<style|<tuple|article|std-latex>>

<\body>
  <\hide-preamble>
    <assign|sfdefault|<macro|phv>>

    <assign|rmdefault|<macro|ptm>>

    <assign|ttdefault|<macro|pcr>>

    <assign|ninept|<macro|<assign|baselinestretch|<macro|.95>><let>>>

    <assign|maketitle|<\macro>
      \ \ <assign|the-footnote|<macro|>> <assign|@|<macro|<rsup|<math|<@thefnmark>>><hss>>>
      <if@twocolumn> <with|par-columns|2|[<@maketitle>]
      <else><new-page><global><@topnum>0pt<@maketitle><fi><@thanks>
      \ <reset-counter|footnote> <let><maketitle> <let><@maketitle>
      <assign|the-footnote|<macro|<number|<footnote-nr>|arabic>>><assign|@|<macro|>><assign|@|<macro|>><assign|@|<macro|>><assign|@|<macro|>><let><thanks|>>
    </macro>>

    <assign|@|<\macro>
      <surround|<new-page><vspace|2em>||<\padded-center>
        \;

        <\with|font-size|1.19|font-series|bold>
          <@title>
        </with>

        \ <vspace|1.5em>

        <\with|font-size|1.19>
          <lineskip>.5em <tabular*|<tformat|<cwith|1|-1|1|1|cell-halign|c>|<cwith|1|-1|1|1|cell-lborder|0ln>|<cwith|1|-1|1|1|cell-rborder|0ln>|<cwith|1|-1|1|-1|cell-valign|t>|<table|<row|<cell|<@name>>>|<row|<cell|<@address>>>>>>
        </with>

        \ 
      </padded-center>>

      \ <vspace|1.5em>
    </macro>>

    <assign|title|<macro|1|<assign|@|<macro|<change-case|<arg|1>|UPCASE>>>>>

    <assign|name|<macro|1|<assign|@|<macro|<with|font-shape|italic|<arg|1>><next-line>>>>>

    <assign|address|<macro|1|<assign|@|<macro|<arg|1>>>>>

    <assign|@|<macro|<change-case|title of paper|UPCASE>>>

    <assign|@|<macro|<with|font-shape|italic|Name of author><next-line>>>

    <assign|@|<macro|Address - Line 1<next-line>Address - Line
    2<next-line>Address - Line 3>>

    <assign|sthanks|<macro|1|<assign|the-footnote|<macro|<number|<footnote-nr>|fnsymbol>>><@@savethanks|<arg|1>>>>

    <assign|twoauthors|<macro|1|2|3|4|<assign|@|<macro|>>
    <assign|@|<macro|<tabular*|<tformat|<cwith|1|-1|1|-1|cell-valign|c>|<table|<row|<cell|<with|font-shape|italic|<arg|1>>>>|<row|<cell|>>|<row|<cell|<arg|2>>>>>><space|1in><tabular*|<tformat|<cwith|1|-1|1|-1|cell-valign|c>|<table|<row|<cell|<with|font-shape|italic|<arg|3>>>>|<row|<cell|>>|<row|<cell|<arg|4>>>>>>>>>>

    <assign|@|<\macro|1|2|3|4|5|6|7|8>
      <next-counter|><assign|@|<macro|<csname>the<arg|1><endcsname>.<space|0.6em>>>
      <ifnum><arg|2>=1

      <\with|font-series|bold|par-mode|center>
        <interlinepenalty><@M> <@svsec><change-case|<arg|8>|UPCASE>

        <else><ifnum><arg|2>=2<no-indent><interlinepenalty><@M><@svsec><arg|8>

        <else>

        <\with|font-shape|italic>
          <no-indent><interlinepenalty><@M> <@svsec><arg|8>

          <fi><fi> <csname><arg|1>mark<endcsname|<arg|7>><addcontentsline|toc|<arg|1>|<numberline|<csname>the<arg|1><endcsname>>
          <arg|7>> <@tempskipa><arg|5> <@xsect|<@tempskipa>>
        </with>
      </with>
    </macro>>

    <assign|abstract|<macro|<padded-center|
    <with|font-series|bold|ABSTRACT<vspace|-.5em><vspace|0pt>> >>>

    <assign|endabstract|<macro|<new-line>>>

    <assign|keywords|<macro|<vspace|.5em>
    <with|font-series|bold|<with|font-shape|italic|Index
    Terms><emdash><space|0.17em>>>>

    <assign|endkeywords|<macro|<new-line>>>

    <assign|copyrightnotice|<macro|1|<assign|@|<macro|<arg|1>>>>>

    <assign|toappear|<macro|1|<assign|@|<macro|<arg|1>>>>>

    <assign|ps|<macro|<assign|mypage|<macro|>><let><@mkboth><@gobbletwo><assign|@|<macro|>>
    <assign|@|<macro|<rlap|<@toappear>><htab|0pt><mypage><htab|0pt>
    <llap|<@copyrightnotice>> <assign|mypage|<macro|<page-the-page>>><assign|@|<macro|>><assign|@|<macro|>>>>>>

    <assign|thebibliography|<macro|1|<section|References><list|[<number|<enumi-nr>|arabic>]|<settowidth><labelwidth>[<arg|1>]<leftmargin><labelwidth>
    <advance><leftmargin><labelsep> <usecounter|enumi>>
    <assign|newblock|<macro|<space|<tex-len|.11em|.33em|.07em>>>>
    <clubpenalty>4000<widowpenalty>4000 <sfcode>`<math|<wide|<text|=>|\<dot\>>>1000>>

    <assign|@|<\macro|1|2>
      <vspace*|10pt><setbox><@tempboxa><arg|1>. <arg|2>
      <ifdim><wd><@tempboxa>\<gtr\><hsize><arg|1>. <arg|2>

      <else>to<hsize><htab|0pt>\<box\><@tempboxa><htab|0pt> <fi>
    </macro>>

    <assign|fnum|<macro|<with|font-series|bold|Fig. <the-figure>>>>

    <assign|fnum|<macro|<with|font-series|bold|Table <the-table>>>>

    <assign|chmark|<macro|<ding|51>>>

    <assign|crmark|<macro|<ding|55>>>

    <assign|source|<macro|1|<with|font-family|tt|<arg|1>>>>

    <assign|alex|<macro|1|<with|color|blue|A: <arg|1>>>>

    <assign|simon|<macro|1|<with|color|red|S: <arg|1>>>>

    <assign|x|<macro|<math-bf|x>>>

    <assign|L|<macro|\<cal-L\>>>
  </hide-preamble>

  <doc-data|<doc-title|hybrid transformers for music source
  separation>|<doc-date|<date|>>>

  <abstract-data|<\abstract>
    A natural question arising in Music Source Separation (MSS) is whether
    long range contextual information is useful, or whether local acoustic
    features are sufficient. In other fields, attention based
    Transformers<nbsp><cite|transformer> have shown their ability to
    integrate information over long sequences. In this work, we introduce
    Hybrid Transformer Demucs (HT Demucs), an hybrid temporal/spectral
    bi-U-Net based on Hybrid Demucs<nbsp><cite|defossez2021hybrid>, where the
    innermost layers are replaced by a cross-domain Transformer Encoder,
    using self-attention within one domain, and cross-attention across
    domains. While it performs poorly when trained only on
    MUSDB<nbsp><cite|musdb>, we show that it outperforms Hybrid Demucs
    (trained on the same data) by 0.45 dB of SDR when using 800 extra
    training songs. Using sparse attention kernels to extend its receptive
    field, and per source fine-tuning, we achieve state-of-the-art results on
    MUSDB with extra training data, with 9.20 dB of SDR.
  </abstract>>

  <section|Introduction><label|sec:intro>

  Since the 2015 Signal Separation Evaluation Campaign (SiSEC)
  <cite|sisec15>, the community of MSS has mostly focused on the task of
  training supervised models to separate songs into 4 stems: drums, bass,
  vocals and other (all the other instruments). The reference dataset that is
  used to benchmark MSS is MUSDB18<nbsp><cite|musdb|musdb18-hq> which is made
  of 150 songs in two versions (HQ and non-HQ). Its training set is composed
  of 87 songs, a relatively small corpus compared with other deep learning
  based tasks, where Transformer<nbsp><cite|transformer> based architectures
  have seen widespread success and adoption, such as
  vision<nbsp><cite|layerscale|Rombach_2022_CVPR> or natural language
  tasks<nbsp><cite|brown2020language>. Source separation is a task where
  having a short context or a long context as input both make sense.
  Conv-Tasnet<nbsp><cite|convtasnet> uses about one second of context to
  perform the separation, using only local acoustic features. On the other
  hand, Demucs<nbsp><cite|demucsv2> can use up to 10 seconds of context,
  which can help to resolve ambiguities in the input. In the present work, we
  aim at studying how Transformer architectures can help leverage this
  context, and what amount of data is required to train them.

  We first present in Section<nbsp><reference|sec:architecture> a novel
  architecture, <em|Hybrid Transformer Demucs> (HT Demucs), which replaces
  the innermost layers of the original Hybrid Demucs
  architecture<nbsp><cite|defossez2021hybrid> with Transformer layers,
  applied both in the time and spectral representation, using self-attention
  within one domain, and cross-attention across domains. As Transformers are
  usually data hungry, we leverage an internal dataset composed of 800 songs
  on top of the MUSDB dataset, described in
  Section<nbsp><reference|sec:dataset>.

  Our second contribution is to evaluate extensively this new architecture in
  Section<nbsp><reference|sec:results>, with various settings (depth, number
  of channels, context length, augmentations etc.). We show in particular
  that it improves over the baseline Hybrid Demucs architecture (retrained on
  the same data) by 0.35<nbsp>dB.

  Finally, we experiment with increasing the context duration using sparse
  kernels based with Locally Sensitive Hashing to overcome memory issues
  during training, and fine-tuning procedure, thus achieving a final SDR of
  9.20<nbsp>dB on the test set of MUSDB.

  We release the training code, pre-trained models, and samples on our
  github<nbsp><hlink|facebookresearch/demucs.|https://github.com/facebookresearch/demucs>

  <section|Related Work><label|sec:related>

  A traditional split for MSS methods is between spectrogram based and
  waveform based models. The former includes models like
  Open-Unmix<nbsp><cite|umx>, a biLSTM with fully connected that predicts a
  mask on the input spectrogram or D3Net<nbsp><cite|d3net> which uses dilated
  convolutional blocks with dense connections. More recently, using
  complex-spectrogram as input and output was favored<nbsp><cite|lasaft> as
  it provides a richer representation and removes the topline given by the
  Ideal-Ratio-Mask. The latest spectrogram model, Band-Split
  RNN<nbsp><cite|bsrnn>, combines this idea, along with multiple dual-path
  RNNs<nbsp><cite|luo2020dual>, each acting in carefully crafted frequency
  band. It currently achieves the state-of-the-art on MUSDB with 8.9 dB.
  Waveform based models started with Wave-U-Net<nbsp><cite|waveunet>, which
  served as the basis for Demucs<nbsp><cite|demucsv2>, a time domain U-Net
  with a bi-LSTM between the encoder and decoder. Around the same time,
  Conv-TasNet showed competitive results<nbsp><cite|convtasnet|demucsv2>
  using residual dilated convolution blocks to predict a mask over a learnt
  representation. Finally, a recent trend has been to use both temporal and
  spectral domains, either through model blending, like
  KUIELAB-MDX-Net<nbsp><cite|kuielab>, or using a bi-U-Net structure with a
  shared backbone as Hybrid Demucs<nbsp><cite|defossez2021hybrid>. Hybrid
  Demucs was the first ranked architecture at the latest MDX MSS
  Competition<nbsp><cite|mdx2021>, although it is now surpassed by Band-Split
  RNN.

  Using large datasets has been shown to be beneficial to the task of MSS.
  Spleeter <cite|spleeter> is a spectrogram masking U-Net architecture
  trained on 25,000 songs extracts of 30 seconds, and was at the time of its
  release, the best model available. Both D3Net and Demucs highly benefited
  from using extra training data, while still offering strong performance on
  MUSDB only. Band-Split RNN introduced a novel unsupervised augmentation
  technique requiring only mixes to improve its performance by 0.7 dB of SDR.

  Transformers have been used for speech source separation with
  SepFormer<nbsp><cite|subakan2021attention>, which is similar to Dual-Path
  RNN: short range attention layers are interleaved with long range ones.
  However, its requires almost 11GB of memory for the forward pass for 5
  seconds of audio at 8 kHz, and thus is not adequate for studying longer
  inputs at 44.1 kHz.

  <label|tab:baselines>

  <tabular*|<tformat|<cwith|1|-1|1|1|cell-halign|l>|<cwith|1|-1|1|1|cell-lborder|0ln>|<cwith|1|-1|2|2|cell-halign|c>|<cwith|1|-1|3|3|cell-halign|r>|<cwith|1|-1|4|4|cell-halign|r>|<cwith|1|-1|5|5|cell-halign|r>|<cwith|1|-1|6|6|cell-halign|r>|<cwith|1|-1|7|7|cell-halign|r>|<cwith|1|-1|7|7|cell-rborder|0ln>|<cwith|1|-1|1|-1|cell-valign|c>|<cwith|1|1|1|-1|cell-tborder|1ln>|<cwith|1|1|4|4|cell-col-span|4>|<cwith|1|1|4|4|cell-halign|c>|<cwith|1|1|4|4|cell-rborder|0ln>|<cwith|1|1|3|7|cell-bborder|1ln>|<cwith|2|2|1|-1|cell-bborder|1ln>|<cwith|3|3|1|-1|cell-bborder|1ln>|<cwith|6|6|1|-1|cell-bborder|1ln>|<cwith|7|7|1|-1|cell-bborder|1ln>|<cwith|12|12|1|-1|cell-bborder|1ln>|<cwith|16|16|1|-1|cell-bborder|1ln>|<table|<row|<cell|>|<cell|>|<cell|>|<cell|Test
  SDR in dB>|<cell|>|<cell|>|<cell|>>|<row|<cell|<with|font-series|bold|Architecture>>|<cell|<with|font-series|bold|Extra?>>|<cell|<source|All>>|<cell|<source|Drums>>|<cell|<source|Bass>>|<cell|<source|Other>>|<cell|<source|Vocals>>>|<row|<cell|IRM
  oracle>|<cell|N/A>|<cell|8.22>|<cell|8.45>|<cell|7.12>|<cell|7.85>|<cell|9.43>>|<row|<cell|KUIELAB-MDX-Net
  <cite|kuielab>>|<cell|<crmark>>|<cell|7.54>|<cell|7.33>|<cell|7.86>|<cell|5.95>|<cell|9.00>>|<row|<cell|Hybrid
  Demucs <cite|defossez2021hybrid>>|<cell|<crmark>>|<cell|7.64>|<cell|8.12>|<cell|8.43>|<cell|5.65>|<cell|8.35>>|<row|<cell|Band-Split
  RNN <cite|bsrnn>>|<cell|<crmark>>|<cell|<with|font-series|bold|8.24>>|<cell|9.01>|<cell|7.22>|<cell|6.70>|<cell|10.01>>|<row|<cell|HT
  Demucs>|<cell|<crmark>>|<cell|7.52>|<cell|7.94>|<cell|8.48>|<cell|5.72>|<cell|7.93>>|<row|<cell|Spleeter<rsup|<math|\<ast\>>>
  <cite|spleeter>>|<cell|<math|25<math-up|k>>>|<cell|5.91>|<cell|6.71>|<cell|5.51>|<cell|4.55>|<cell|6.86>>|<row|<cell|D3Net<rsup|<math|\<ast\>>>
  <cite|d3net>>|<cell|1.5k>|<cell|6.68>|<cell|7.36>|<cell|6.20>|<cell|5.37>|<cell|7.80>>|<row|<cell|Demucs
  v2<rsup|<math|\<ast\>>> <cite|demucsv2>>|<cell|150>|<cell|6.79>|<cell|7.58>|<cell|7.60>|<cell|4.69>|<cell|7.29>>|<row|<cell|Hybrid
  Demucs <cite|defossez2021hybrid>>|<cell|800>|<cell|8.34>|<cell|9.31>|<cell|9.13>|<cell|6.18>|<cell|8.75>>|<row|<cell|Band-Split
  RNN <cite|bsrnn>>|<cell|1750<rsup|<math|\<dagger\>>>>|<cell|8.97>|<cell|10.15>|<cell|8.16>|<cell|<with|font-series|bold|7.08>>|<cell|<with|font-series|bold|10.47>>>|<row|<cell|HT
  Demucs>|<cell|150>|<cell|8.49>|<cell|9.51>|<cell|9.76>|<cell|6.13>|<cell|8.56>>|<row|<cell|HT
  Demucs>|<cell|800>|<cell|8.80>|<cell|10.05>|<cell|9.78>|<cell|6.42>|<cell|8.93>>|<row|<cell|HT
  Demucs (fine tuned)>|<cell|800>|<cell|9.00>|<cell|10.08>|<cell|10.39>|<cell|6.32>|<cell|9.20>>|<row|<cell|Sparse
  HT Demucs (fine tuned)>|<cell|800>|<cell|<with|font-series|bold|9.20>>|<cell|<with|font-series|bold|10.83>>|<cell|<with|font-series|bold|10.47>>|<cell|6.41>|<cell|9.37>>>>>
  <vspace|-0.1cm>

  <with|font-size|0.71| >

  <section|Architecture><label|sec:architecture>

  \;

  We introduce the Hybrid Transformer Demucs model, based on Hybrid
  Demucs<nbsp><cite|defossez2021hybrid>. The original Hybrid Demucs model is
  made of two U-Nets, one in the time domain (with temporal convolutions) and
  one in the spectrogram domain (with convolutions over the frequency axis).
  Each U-Net is made of 5 encoder layers, and 5 decoder layers. After the
  5-th encoder layer, both representation have the same shape, and they are
  summed before going into a shared 6-th layer. Similarly, the first decoder
  layer is shared, and its output is sent both the temporal and spectral
  branch. The output of the spectral branch is transformed to a waveform
  using the iSTFT, before being summed with the output of the temporal
  branch, giving the actual prediction of the model.

  Hybrid Transformer Demucs keeps the outermost 4 layers as is from the
  original architecture, and replaces the 2 innermost layers in the encoder
  and the decoder, including local attention and bi-LSTM, with a cross-domain
  Transformer Encoder. It treats in parallel the 2D signal from the spectral
  branch and the 1D signal from the waveform branch. Unlike the original
  Hybrid Demucs which required careful tuning of the model parameters (STFT
  window and hop length, stride, paddding, etc.) to align the time and
  spectral representation, the cross-domain Transformer Encoder can work with
  heterogeneous data shape, making it a more flexible architecture.

  The architecture of Hybrid Transformer Demucs is depicted on
  Fig.<reference|fig:architecture>. On the left, we show a single
  self-attention Encoder layer of the Transformer<nbsp><cite|transformer>
  with normalizations before the Self-Attention and Feed-Forward operations,
  it is combined with Layer Scale <cite|layerscale> initialized to
  <math|\<epsilon\>=10<rsup|-4>> in order to stabilize the training. The two
  first normalizations are layer normalizations (each token is independently
  normalized) and the third one is a time layer normalization (all the tokens
  are normalized together). The input/output dimension of the Transformer is
  <math|384>, and linear layers are used to convert to the internal dimension
  of the Transformer when required. The attention mechanism has 8 heads and
  the hidden state size of the feed forward network is equal to 4 times the
  dimension of the transformer. The cross-attention Encoder layer is the same
  but using cross-attention with the other domain representation. In the
  middle, a cross-domain Transformer Encoder of depth 5 is depicted. It is
  the interleaving of self-attention Encoder layers and cross-attention
  Encoder layers in the spectral and waveform domain. 1D <cite|transformer>
  and 2D <cite|2Dpe> sinusoidal encodings are added to the scaled inputs and
  reshaping is applied to the spectral representation in order to treat it as
  a sequence. On Fig.<nbsp><reference|fig:architecture> (c), we give a
  representation of the entire architecture, along with the double U-Net
  encoder/decoder structure.

  Memory consumption and the speed of attention quickly deteriorates with an
  increase of the sequence lengths. To further scale, we leverage sparse
  attention kernels introduced in the <with|font-family|tt|xformer>
  package<nbsp><cite|xFormers2021>, along with a Locally Sensitive Hashing
  (LSH) scheme to determine dynamically the sparsity pattern. We use a
  sparsity level of 90% (defined as the proportion of elements removed in the
  softmax), which is determined by performing 32 rounds of LSH with 4 buckets
  each. We select the elements that match at least <math|k> times over all 32
  rounds of LSH, with <math|k> such that the sparsity level is 90%. We refer
  to this variant as Sparse HT Demucs.

  <section|Dataset><label|sec:dataset>

  We curated an internal dataset composed of 3500 songs with the stems from
  200 artists with diverse music genres. Each stem is assigned to one of the
  4 sources according to the name given by the music producer (for instance
  "vocals2", "fx", "sub" etc...). This labeling is noisy because these names
  are subjective and sometime ambiguous. For 150 of those tracks, we manually
  verified that the automated labeling was correct, and discarded ambiguous
  stems. We trained a first Hybrid Demucs model on MUSDB and those 150
  tracks. We preprocess the dataset according to several rules. First, we
  keep only the stems for which all four sources are non silent at least 30%
  of the time. For each 1 second segment, we define it as silent if its
  volume is less than -40dB. Second, for a song <math|x> of our dataset,
  noting <math|x<rsub|i>>, with <math|i\<in\><around|{|<with|math-font-family|rm|drums,
  bass, other, vocals>|}>>, each stem and <math|f> the Hybrid Demucs model
  previously mentioned, we define <math|y<rsub|i,j>=f<around|(|x<rsub|i>|)><rsub|j>>
  i.e. the output <math|j> when separating the stem <math|i>. Theoretically,
  if all the stems were perfectly labeled and if <math|f> were a perfect
  source separation model we would have <math|y<rsub|i,j>=x<rsub|i>*\<delta\><rsub|i,j>>
  with <math|\<delta\><rsub|i,j>> being the Kronecker delta. For a waveform
  <math|z>, let us define the volume in dB measured over 1 second segments:

  <\equation>
    V<around|(|z|)>=10\<cdot\>log<rsub|10><around*|(|<math-up|AveragePool><around|(|z<rsup|2>,1<text|sec>|)>|)>.
  </equation>

  For each pair of sources <math|i,j>, we take the segments of 1 second where
  the stem is present (regarding the first criteria) and define
  <math|P<rsub|i,j>> as the proportion of these segments where
  <math|V<around|(|y<rsub|i,j>|)>-V<around|(|x<rsub|i>|)>\<gtr\>-10<space|0.17em><math-up|dB>>.
  We obtain a square matrix <math|P\<in\><around|[|0,1|]><rsup|4\<times\>4>>,
  and we notice that in perfect condition, we should have
  <math|P=<math-up|Id>>. We thus keep only the songs for which for all
  sources <math|i>, <math|P<rsub|i,i>\<gtr\>70%>, and pairs of sources
  <math|i\<neq\>j>, <math|P<rsub|i,j>\<less\>30%>. This procedure selects 800
  songs.

  <section|Experiments and Results><label|sec:results>

  <subsection|Experimental Setup>

  All our experiments are done on 8 Nvidia V100 GPUs with 32GB of memory,
  using fp32 precision. We use the L1 loss on the waveforms, optimized with
  Adam <cite|adam> without weight decay, a learning rate of
  <math|3\<cdot\>10<rsup|-4>>, <math|\<beta\><rsub|1>=0.9>,
  <math|\<beta\><rsub|2>=0.999> and a batch size of 32 unless stated
  otherwise. We train for 1200 epochs of 800 batches each over the MUSDB18-HQ
  dataset, completed with the 800 curated songs dataset presented in
  Section<nbsp><reference|sec:dataset>, sampled at 44.1kHz and stereophonic.
  We use exponential moving average as described
  in<nbsp><cite|defossez2021hybrid>, and select the best model over the valid
  set, composed of the valid set of MUSDB18-HQ and another 8 songs. We use
  the same data augmentation as described in<nbsp><cite|defossez2021hybrid>,
  including repitching/tempo stretch and remixing of the stems within one
  batch.

  Using one model per source can be beneficial<nbsp><cite|bsrnn>, although
  its adds overhead both at train time and evaluation. In order to limit the
  impact at train time, we propose a procedure where one copy of the
  multi-target model is fine-tuned on a single target task for 50 epochs,
  with a learning rate of <math|10<rsup|-4>>, no remixing, repitching, nor
  rescaling applied as data augmentation. Having noticed some instability
  towards the end of the training of the main model, we use for the fine
  tuning a gradient clipping (maximum L2 norm of <math|5>. for the gradient),
  and a weight decay of 0.05.

  At test time, we split the audio into chunks having the same duration as
  the one used for training, with an overlap of 25% and a linear transition
  from one chunk to the next. We report is the Signal-to-Distortion-Ratio
  (SDR) as defined by the SiSEC18 <cite|sisec18> which is the median across
  the median SDR over all 1 second chunks in each song. On
  Tab.<nbsp><reference|tab:preliminary> we also report the Real Time Factor
  (RTF) computed on a single core of an Intel Xeon CPU at 2.20GHz. It is
  defined as the time to process some fixed audio input (we use 40 seconds of
  gaussian noise) divided by the input duration.

  <subsection|Comparison with the baselines>

  On Tab.<nbsp><reference|tab:baselines>, we compare to several
  state-of-the-art baselines. For reference, we also provide baselines that
  are trained without any extra training data. Comparing to the original
  Hybrid Demucs architecture, we notice that the improved sequence modeling
  capabilities of transformers increased the SDR by 0.45 dB for the simple
  version, and up to almost 0.9 dB when using the sparse variant of our model
  along with fine tuning. The most competitive baseline is Band-Split
  RNN<nbsp><cite|bsrnn>, which achieves a better SDR on both the other and
  vocals sources, despite using only MUSDB18HQ as a supervised training set,
  and using 1750 unsupervised tracks as extra training data.

  <subsection|Impact of the architecture hyper-parameters><label|sec:hyperparams>

  We first study the influence of three architectural hyper-parameters in
  Tab.<nbsp><reference|tab:preliminary>: the duration in seconds of the
  excerpts used to train the model, the depth of the transformer encoder and
  its dimension. For short training excerpts (3.4 seconds), we notice that
  augmenting the depth from 5 to 7 increases the test SDR and that augmenting
  the transformer dimension from 384 to 512 lowers it slightly by 0.05 dB.
  With longer segments (7.8 seconds), we observe an increase of almost 0.6 dB
  when using a depth of 5 and 384 dimensions. Given this observation, we also
  tried to increase the duration or the depth but this led to Out Of Memory
  (OOM) errors. Finally, augmenting the dimension to 512 when training over
  7.8 seconds led to an improvement of 0.1 dB.

  <subsection|Impact of the data augmentation><label|sec:data_augment>

  On Tab.<nbsp><reference|tab:mixing>, we study the impact of disabling some
  of the data augmentation, as we were hoping that using more training data
  would reduce the need for such augmentations. However, we observe a
  constant deterioration of the final SDR as we disable those. While the
  repitching augmentation has a limited impact, the remixing augmentation
  remain highly important to train our model, with a loss of 0.7 dB without
  it.

  <subsection|Impact of using sparse kernels and fine
  tuning><label|sec:finetune>

  We test the sparse kernels described in
  Section<nbsp><reference|sec:architecture> to increase the depth to 7 and
  the train segment duration to 12.2 seconds, with a dimension of 512. This
  simple change yields an extra 0.14 dB of SDR (8.94 dB). The fine-tuning per
  source improves the SDR by 0.25 dB, to 9.20 dB, despite requiring only 50
  epochs to train. We tried further extending the receptive field of the
  Transformer Encoder to 15 seconds during the fine tuning stage by reducing
  the batch size, however, this led to the same SDR of 9.20 dB, although
  training from scratch with such a context might lead to a different result.

  <section*|Conclusion>

  We introduced Hybrid Transformer Demucs, a Transformer based variant of
  Hybrid Demucs that replaces the innermost convolutional layers by a
  Cross-domain Transformer Encoder, using self-attention and cross-attention
  to process spectral and temporal informations. This architecture benefits
  from our large training dataset and outperforms Hybrid Demucs by 0.45 dB.
  Thanks to sparse attention techniques, we scaled our model to an input
  length up to 12.2 seconds during training which led to a supplementary gain
  of 0.4 dB. Finally, we could explore splitting the spectrogram into
  subbands in order to process them differently as it is done in
  <cite|bsrnn>.

  <\comment>
    <\big-table>
      <label|tab:preliminary>

      <padded-center|<resizebox|0.4tex-text-width|!|<tabular*|<tformat|<cwith|1|-1|1|1|cell-halign|c>|<cwith|1|-1|1|1|cell-lborder|0ln>|<cwith|1|-1|2|2|cell-halign|c>|<cwith|1|-1|3|3|cell-halign|c>|<cwith|1|-1|4|4|cell-halign|c>|<cwith|1|-1|4|4|cell-rborder|0ln>|<cwith|1|-1|1|-1|cell-valign|c>|<cwith|1|1|1|-1|cell-tborder|1ln>|<cwith|1|1|1|-1|cell-bborder|1ln>|<cwith|2|2|1|-1|cell-bborder|1ln>|<cwith|6|6|1|-1|cell-bborder|1ln>|<cwith|8|8|1|-1|cell-bborder|1ln>|<table|<row|<cell|<with|font-series|bold|duration>>|<cell|<with|font-series|bold|depth>>|<cell|<with|font-series|bold|dimension>>|<cell|<source|Test
      SDR (All)>>>|<row|<cell|7.8>|<cell|5>|<cell|384>|<cell|8.51>>|<row|<cell|3.4>|<cell|5>|<cell|384>|<cell|7.76>>|<row|<cell|12.2>|<cell|5>|<cell|384>|<cell|8.56>>|<row|<cell|7.8>|<cell|7>|<cell|384>|<cell|8.68>>|<row|<cell|7.8>|<cell|5>|<cell|512>|<cell|7.68>>|<row|<cell|12.2>|<cell|5>|<cell|512>|<cell|8.69>>|<row|<cell|12.2>|<cell|7>|<cell|384>|<cell|Out
      of Memory>>>>>> >
    </big-table|Study of duration, Transformer depth and Transformer
    dimension <alex|change all results to BS=32 only!! and add poulain
    955717e8, and add as OOM 955717e8 + depth=7 + segment =12.2>
    <simon|est-ce qu'on rajouterait pas les real time factor CPU dans les
    tableaux ?>>
  </comment>

  <\big-table>
    <label|tab:preliminary>

    <padded-center|<resizebox|0.46tex-text-width|!|<tabular*|<tformat|<cwith|1|-1|1|1|cell-halign|r>|<cwith|1|-1|1|1|cell-lborder|0ln>|<cwith|1|-1|2|2|cell-halign|c>|<cwith|1|-1|3|3|cell-halign|c>|<cwith|1|-1|4|4|cell-halign|r>|<cwith|1|-1|5|5|cell-halign|c>|<cwith|1|-1|6|6|cell-halign|r>|<cwith|1|-1|6|6|cell-rborder|0ln>|<cwith|1|-1|1|-1|cell-valign|c>|<cwith|1|1|1|-1|cell-tborder|1ln>|<cwith|1|1|1|-1|cell-bborder|1ln>|<cwith|4|4|1|-1|cell-bborder|1ln>|<cwith|7|7|1|-1|cell-bborder|1ln>|<cwith|8|8|1|-1|cell-bborder|1ln>|<table|<row|<cell|<with|font-series|bold|dur.
    (sec)>>|<cell|<with|font-series|bold|depth>>|<cell|<with|font-series|bold|dim.>>|<cell|<with|font-series|bold|nb.
    param.>>|<cell|<with|font-series|bold|RTF (cpu)>>|<cell|<source|SDR
    (All)>>>|<row|<cell|3.4>|<cell|5>|<cell|384>|<cell|26.9M>|<cell|1.02>|<cell|8.17>>|<row|<cell|3.4>|<cell|7>|<cell|384>|<cell|34.0M>|<cell|1.23>|<cell|8.26>>|<row|<cell|3.4>|<cell|5>|<cell|512>|<cell|41.4M>|<cell|1.30>|<cell|8.12>>|<row|<cell|7.8>|<cell|5>|<cell|384>|<cell|26.9M>|<cell|1.49>|<cell|8.70>>|<row|<cell|7.8>|<cell|7>|<cell|384>|<cell|34.0M>|<cell|1.68>|<cell|OOM>>|<row|<cell|7.8>|<cell|5>|<cell|512>|<cell|41.4M>|<cell|1.77>|<cell|<with|font-series|bold|8.80>>>|<row|<cell|12.2>|<cell|5>|<cell|384>|<cell|26.9M>|<cell|2.04>|<cell|OOM>>>>>>
    >
  </big-table|Impact of segment duration, transformer depth and transformer
  dimension. OOM means Out of Memory. Results are commented in
  Sec.<nbsp><reference|sec:hyperparams>.>

  <label|tab:mixing>

  <resizebox|0.42tex-text-width|!|<tabular*|<tformat|<cwith|1|-1|1|1|cell-halign|c>|<cwith|1|-1|1|1|cell-lborder|0ln>|<cwith|1|-1|2|2|cell-halign|c>|<cwith|1|-1|3|3|cell-halign|c>|<cwith|1|-1|4|4|cell-halign|c>|<cwith|1|-1|5|5|cell-halign|c>|<cwith|1|-1|5|5|cell-rborder|0ln>|<cwith|1|-1|1|-1|cell-valign|c>|<cwith|1|1|1|-1|cell-tborder|1ln>|<cwith|1|1|1|-1|cell-bborder|1ln>|<cwith|4|4|1|-1|cell-bborder|1ln>|<table|<row|<cell|<with|font-series|bold|dur.
  (sec)>>|<cell|<with|font-series|bold|depth>>|<cell|<with|font-series|bold|remixing>>|<cell|<with|font-series|bold|repitching>>|<cell|<source|SDR
  (All)>>>|<row|<cell|7.8>|<cell|5>|<cell|<chmark>>|<cell|<chmark>>|<cell|8.70>>|<row|<cell|7.8>|<cell|5>|<cell|<chmark>>|<cell|<crmark>>|<cell|8.65>>|<row|<cell|7.8>|<cell|5>|<cell|<crmark>>|<cell|<chmark>>|<cell|8.00>>>>>>\ 

  <label|tab:sparse_ft>

  <tabular*|<tformat|<cwith|1|-1|1|1|cell-halign|c>|<cwith|1|-1|1|1|cell-lborder|0ln>|<cwith|1|-1|2|2|cell-halign|c>|<cwith|1|-1|3|3|cell-halign|c>|<cwith|1|-1|4|4|cell-halign|c>|<cwith|1|-1|5|5|cell-halign|c>|<cwith|1|-1|6|6|cell-halign|c>|<cwith|1|-1|6|6|cell-rborder|0ln>|<cwith|1|-1|1|-1|cell-valign|c>|<cwith|1|1|1|-1|cell-tborder|1ln>|<cwith|1|1|1|-1|cell-bborder|1ln>|<cwith|5|5|1|-1|cell-bborder|1ln>|<table|<row|<cell|<with|font-series|bold|depth>>|<cell|<with|font-series|bold|sparsity>>|<cell|<with|font-series|bold|train
  dur. (sec)>>|<cell|<with|font-series|bold|fine-tune dur.
  (sec)>>|<cell|<with|font-series|bold|per source?>>|<cell|<source|SDR
  (All)>>>|<row|<cell|5>|<cell|<crmark>>|<cell|7.8>|<cell|<crmark>>|<cell|<crmark>>|<cell|8.80>>|<row|<cell|7>|<cell|90%>|<cell|12.2>|<cell|<crmark>>|<cell|<crmark>>|<cell|8.94>>|<row|<cell|7>|<cell|90%>|<cell|12.2>|<cell|12.2>|<cell|<chmark>>|<cell|<with|font-series|bold|9.20>>>|<row|<cell|7>|<cell|90%>|<cell|12.2>|<cell|18>|<cell|<chmark>>|<cell|?>>>>>\ 

  \;

  <\bibliography|bib|IEEEbib|refs>
    <\bib-list|10>
      <bibitem*|1><label|bib-transformer>Ashish Vaswani, Noam Shazeer, Niki
      Parmar, Jakob Uszkoreit, Llion Jones, Aidan<nbsp>N. Gomez, Lukasz
      Kaiser, and Illia Polosukhin, <newblock>\PAttention is all you need,\Q
      <newblock><with|font-shape|italic|CoRR>, vol. abs/1706.03762, 2017.

      <bibitem*|2><label|bib-defossez2021hybrid>Alexandre Défossez,
      <newblock>\PHybrid spectrogram and waveform source separation,\Q
      <newblock>in <with|font-shape|italic|Proceedings of the ISMIR 2021
      Workshop on Music Source Separation>, 2021.

      <bibitem*|3><label|bib-musdb>Zafar Rafii, Antoine Liutkus,
      Fabian-Robert Stöter, Stylianos<nbsp>Ioannis Mimilakis, and Rachel
      Bittner, <newblock>\PThe musdb18 corpus for music separation,\Q 2017.

      <bibitem*|4><label|bib-sisec15>Nobutaka Ono, Zafar Rafii, Daichi
      Kitamura, Nobutaka Ito, and Antoine Liutkus, <newblock>\PThe 2015
      Signal Separation Evaluation Campaign,\Q <newblock>in
      <with|font-shape|italic|International Conference on Latent Variable
      Analysis and Signal Separation (LVA/ICA)>, Aug. 2015.

      <bibitem*|5><label|bib-musdb18-hq>Zafar Rafii, Antoine Liutkus,
      Fabian-Robert Stöter, Stylianos<nbsp>Ioannis Mimilakis, and Rachel
      Bittner, <newblock>\PMusdb18-hq - an uncompressed version of musdb18,\Q
      Aug. 2019.

      <bibitem*|6><label|bib-layerscale>Hugo Touvron, Matthieu Cord,
      Alexandre Sablayrolles, Gabriel Synnaeve, and Hervé Jégou,
      <newblock>\PGoing deeper with image transformers,\Q <newblock>in
      <with|font-shape|italic|Proceedings of the IEEE/CVF International
      Conference on Computer Vision>, 2021.

      <bibitem*|7><label|bib-Rombach_2022_CVPR>Robin Rombach, Andreas
      Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer,
      <newblock>\PHigh-resolution image synthesis with latent diffusion
      models,\Q <newblock>in <with|font-shape|italic|Proceedings of the
      IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)>,
      June 2022, pp. 10684\U10695.

      <bibitem*|8><label|bib-brown2020language>Tom B.<nbsp>Brown et<nbsp>al.,
      <newblock>\PLanguage models are few-shot learners,\Q <newblock>2020.

      <bibitem*|9><label|bib-convtasnet>Yi<nbsp>Luo and Nima Mesgarani,
      <newblock>\PConv-tasnet: Surpassing ideal time\Ufrequency magnitude
      masking for speech separation,\Q <newblock><with|font-shape|italic|IEEE/ACM
      Transactions on Audio, Speech, and Language Processing>, 2019.

      <bibitem*|10><label|bib-demucsv2>Alexandre Défossez, Nicolas Usunier,
      Léon Bottou, and Francis Bach, <newblock>\PMusic source separation in
      the waveform domain,\Q 2019.

      <bibitem*|11><label|bib-umx>F.-R. Stöter, S.<nbsp>Uhlich,
      A.<nbsp>Liutkus, and Y.<nbsp>Mitsufuji, <newblock>\POpen-unmix - a
      reference implementation for music source separation,\Q
      <newblock><with|font-shape|italic|Journal of Open Source Software>,
      2019.

      <bibitem*|12><label|bib-d3net>Naoya Takahashi and Yuki Mitsufuji,
      <newblock>\PD3net: Densely connected multidilated densenet for music
      source separation,\Q 2020.

      <bibitem*|13><label|bib-lasaft>Woosung Choi, Minseok Kim, Jaehwa Chung,
      and Soonyoung Jung, <newblock>\PLasaft: Latent source attentive
      frequency transformation for conditioned source separation,\Q
      <newblock>in <with|font-shape|italic|IEEE International Conference on
      Acoustics, Speech and Signal Processing (ICASSP)>, 2021.

      <bibitem*|14><label|bib-bsrnn>Yi<nbsp>Luo and Jianwei Yu,
      <newblock>\PMusic source separation with band-split rnn,\Q 2022.

      <bibitem*|15><label|bib-luo2020dual>Yi<nbsp>Luo, Zhuo Chen, and Takuya
      Yoshioka, <newblock>\PDual-path rnn: efficient long sequence modeling
      for time-domain single-channel speech separation,\Q <newblock>in
      <with|font-shape|italic|ICASSP 2020-2020 IEEE International Conference
      on Acoustics, Speech and Signal Processing (ICASSP)>. IEEE, 2020, pp.
      46\U50.

      <bibitem*|16><label|bib-waveunet>Daniel Stoller, Sebastian Ewert, and
      Simon Dixon, <newblock>\PWave-u-net: A multi-scale neural network for
      end-to-end audio source separation,\Q
      <newblock><with|font-shape|italic|arXiv preprint arXiv:1806.03185>,
      2018.

      <bibitem*|17><label|bib-kuielab>Minseok Kim, Woosung Choi, Jaehwa
      Chung, Daewon Lee, and Soonyoung Jung, <newblock>\PKuielab-mdx-net: A
      two-stream neural network for music demixing,\Q 2021.

      <bibitem*|18><label|bib-mdx2021>Yuki Mitsufuji, Giorgio Fabbro, Stefan
      Uhlich, Fabian-Robert Stöter, Alexandre Défossez, Minseok Kim, Woosung
      Choi, Chin-Yun Yu, and Kin-Wai Cheuk, <newblock>\PMusic demixing
      challenge 2021,\Q <newblock><with|font-shape|italic|Frontiers in Signal
      Processing>, vol. 1, jan 2022.

      <bibitem*|19><label|bib-spleeter>Romain Hennequin, Anis Khlif, Felix
      Voituret, and Manuel Moussallam, <newblock>\PSpleeter: a fast and
      efficient music source separation tool with pre-trained models,\Q
      <newblock><with|font-shape|italic|Journal of Open Source Software>,
      2020.

      <bibitem*|20><label|bib-subakan2021attention>Cem Subakan, Mirco
      Ravanelli, Samuele Cornell, Mirko Bronzi, and Jianyuan Zhong,
      <newblock>\PAttention is all you need in speech separation,\Q
      <newblock>in <with|font-shape|italic|ICASSP 2021-2021 IEEE
      International Conference on Acoustics, Speech and Signal Processing
      (ICASSP)>. IEEE, 2021, pp. 21\U25.

      <bibitem*|21><label|bib-2Dpe>Zelun Wang and Jyh-Charn Liu,
      <newblock>\PTranslating math formula images to latex sequences using
      deep neural networks with sequence-level training,\Q 2019.

      <bibitem*|22><label|bib-xFormers2021>Benjamin Lefaudeux, Francisco
      Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren,
      Min Xu, Jieru Hu, Marta Tintore, and Susan Zhang, <newblock>\Pxformers:
      A modular and hackable transformer modelling library,\Q
      <slink|https://github.com/facebookresearch/xformers>, 2021.

      <bibitem*|23><label|bib-adam>Diederik<nbsp>P. Kingma and Jimmy Ba,
      <newblock>\PAdam: A method for stochastic optimization,\Q 2014.

      <bibitem*|24><label|bib-sisec18>Fabian-Robert Stöter, Antoine Liutkus,
      and Nobutaka Ito, <newblock>\PThe 2018 signal separation evaluation
      campaign,\Q 2018.
    </bib-list>
  </bibliography>
</body>

<\initial>
  <\collection>
    <associate|mode|prog>
    <associate|preamble|false>
    <associate|prog-language|latex>
    <associate|tex-column-sep|6mm>
    <associate|tex-even-side-margin|-6.2mm>
    <associate|tex-head-height|0pt>
    <associate|tex-head-sep|0pt>
    <associate|tex-odd-side-margin|-6.2mm>
    <associate|tex-text-height|229mm>
    <associate|tex-text-width|178mm>
    <associate|tex-top-margin|0pt>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?|LaTeX Source\\main.tex>>
    <associate|auto-10|<tuple|5.5|?|LaTeX Source\\main.tex>>
    <associate|auto-11|<tuple|5.5|?|LaTeX Source\\main.tex>>
    <associate|auto-12|<tuple|1|?|LaTeX Source\\main.tex>>
    <associate|auto-13|<tuple|2|?|LaTeX Source\\main.tex>>
    <associate|auto-14|<tuple|2|?|LaTeX Source\\main.tex>>
    <associate|auto-15|<tuple|3|?|LaTeX Source\\main.tex>>
    <associate|auto-16|<tuple|4|?|LaTeX Source\\main.tex>>
    <associate|auto-2|<tuple|2|?|LaTeX Source\\main.tex>>
    <associate|auto-3|<tuple|3|?|LaTeX Source\\main.tex>>
    <associate|auto-4|<tuple|4|?|LaTeX Source\\main.tex>>
    <associate|auto-5|<tuple|5|?|LaTeX Source\\main.tex>>
    <associate|auto-6|<tuple|5.1|?|LaTeX Source\\main.tex>>
    <associate|auto-7|<tuple|5.2|?|LaTeX Source\\main.tex>>
    <associate|auto-8|<tuple|5.3|?|LaTeX Source\\main.tex>>
    <associate|auto-9|<tuple|5.4|?|LaTeX Source\\main.tex>>
    <associate|bib-2Dpe|<tuple|21|?|LaTeX Source\\main.tex>>
    <associate|bib-Rombach_2022_CVPR|<tuple|7|?|LaTeX Source\\main.tex>>
    <associate|bib-adam|<tuple|23|?|LaTeX Source\\main.tex>>
    <associate|bib-brown2020language|<tuple|8|?|LaTeX Source\\main.tex>>
    <associate|bib-bsrnn|<tuple|14|?|LaTeX Source\\main.tex>>
    <associate|bib-convtasnet|<tuple|9|?|LaTeX Source\\main.tex>>
    <associate|bib-d3net|<tuple|12|?|LaTeX Source\\main.tex>>
    <associate|bib-defossez2021hybrid|<tuple|2|?|LaTeX Source\\main.tex>>
    <associate|bib-demucsv2|<tuple|10|?|LaTeX Source\\main.tex>>
    <associate|bib-kuielab|<tuple|17|?|LaTeX Source\\main.tex>>
    <associate|bib-lasaft|<tuple|13|?|LaTeX Source\\main.tex>>
    <associate|bib-layerscale|<tuple|6|?|LaTeX Source\\main.tex>>
    <associate|bib-luo2020dual|<tuple|15|?|LaTeX Source\\main.tex>>
    <associate|bib-mdx2021|<tuple|18|?|LaTeX Source\\main.tex>>
    <associate|bib-musdb|<tuple|3|?|LaTeX Source\\main.tex>>
    <associate|bib-musdb18-hq|<tuple|5|?|LaTeX Source\\main.tex>>
    <associate|bib-sisec15|<tuple|4|?|LaTeX Source\\main.tex>>
    <associate|bib-sisec18|<tuple|24|?|LaTeX Source\\main.tex>>
    <associate|bib-spleeter|<tuple|19|?|LaTeX Source\\main.tex>>
    <associate|bib-subakan2021attention|<tuple|20|?|LaTeX Source\\main.tex>>
    <associate|bib-transformer|<tuple|1|?|LaTeX Source\\main.tex>>
    <associate|bib-umx|<tuple|11|?|LaTeX Source\\main.tex>>
    <associate|bib-waveunet|<tuple|16|?|LaTeX Source\\main.tex>>
    <associate|bib-xFormers2021|<tuple|22|?|LaTeX Source\\main.tex>>
    <associate|sec:architecture|<tuple|3|?|LaTeX Source\\main.tex>>
    <associate|sec:data_augment|<tuple|5.4|?|LaTeX Source\\main.tex>>
    <associate|sec:dataset|<tuple|4|?|LaTeX Source\\main.tex>>
    <associate|sec:finetune|<tuple|5.5|?|LaTeX Source\\main.tex>>
    <associate|sec:hyperparams|<tuple|5.3|?|LaTeX Source\\main.tex>>
    <associate|sec:intro|<tuple|1|?|LaTeX Source\\main.tex>>
    <associate|sec:related|<tuple|2|?|LaTeX Source\\main.tex>>
    <associate|sec:results|<tuple|5|?|LaTeX Source\\main.tex>>
    <associate|tab:baselines|<tuple|2|?|LaTeX Source\\main.tex>>
    <associate|tab:mixing|<tuple|2|?|LaTeX Source\\main.tex>>
    <associate|tab:preliminary|<tuple|2|?|LaTeX Source\\main.tex>>
    <associate|tab:sparse_ft|<tuple|2|?|LaTeX Source\\main.tex>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      transformer

      defossez2021hybrid

      musdb

      sisec15

      musdb

      musdb18-hq

      transformer

      layerscale

      Rombach_2022_CVPR

      brown2020language

      convtasnet

      demucsv2

      defossez2021hybrid

      umx

      d3net

      lasaft

      bsrnn

      luo2020dual

      waveunet

      demucsv2

      convtasnet

      demucsv2

      kuielab

      defossez2021hybrid

      mdx2021

      spleeter

      subakan2021attention

      kuielab

      defossez2021hybrid

      bsrnn

      spleeter

      d3net

      demucsv2

      defossez2021hybrid

      bsrnn

      defossez2021hybrid

      transformer

      layerscale

      transformer

      2Dpe

      xFormers2021

      adam

      defossez2021hybrid

      defossez2021hybrid

      bsrnn

      sisec18

      bsrnn

      bsrnn
    </associate>
    <\associate|table>
      <tuple|normal|<surround|<hidden-binding|<tuple>|1>||Study of duration,
      Transformer depth and Transformer dimension <with|color|<quote|blue>|A:
      change all results to BS=32 only!! and add poulain 955717e8, and add as
      OOM 955717e8 + depth=7 + segment =12.2> <with|color|<quote|red>|S:
      est-ce qu'on rajouterait pas les real time factor CPU dans les tableaux
      ?>>|<pageref|auto-12>>

      <tuple|normal|<surround|<hidden-binding|<tuple>|2>||Impact of segment
      duration, transformer depth and transformer dimension. OOM means Out of
      Memory. Results are commented in Sec.
      <no-break><specific|screen|<resize|<move|<with|color|<quote|#A0A0FF>|->|-0.3em|>|0em||0em|>><reference|sec:hyperparams>.>|<pageref|auto-13>>

      <tuple|normal|<surround|<hidden-binding|<tuple>|3>||Impact of data
      augmentation. The model has a depth of 5, and a dimension of 384. See
      Sec. <no-break><specific|screen|<resize|<move|<with|color|<quote|#A0A0FF>|->|-0.3em|>|0em||0em|>><reference|sec:data_augment>
      for details.>|<pageref|auto-14>>

      <tuple|normal|<surround|<hidden-binding|<tuple>|4>||Effect of LSH based
      sparse attention and of the fine-tuning procedure described in Section
      <no-break><specific|screen|<resize|<move|<with|color|<quote|#A0A0FF>|->|-0.3em|>|0em||0em|>><reference|sec:finetune>>|<pageref|auto-15>>
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Introduction>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Related
      Work> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Architecture>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>Dataset>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|5<space|2spc>Experiments
      and Results> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5><vspace|0.5fn>

      <with|par-left|<quote|1tab>|5.1<space|2spc>Experimental Setup
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|1tab>|5.2<space|2spc>Comparison with the
      baselines <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      <with|par-left|<quote|1tab>|5.3<space|2spc>Impact of the architecture
      hyper-parameters <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8>>

      <with|par-left|<quote|1tab>|5.4<space|2spc>Impact of the data
      augmentation <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9>>

      <with|par-left|<quote|1tab>|5.5<space|2spc>Impact of using sparse
      kernels and fine tuning <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-10>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Conclusion>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-11><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-16><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>
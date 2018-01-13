#pragma once

#include "marian.h"

namespace marian {

class Transformer {
public:
  Expr TransposeTimeBatch(Expr input) { return transpose(input, {0, 2, 1, 3}); }

  Expr AddPositionalEmbeddings(Ptr<ExpressionGraph> graph,
                               Expr input,
                               int start = 0) {
    using namespace keywords;

    int dimEmb = input->shape()[-1];
    int dimWords = input->shape()[-3];

    float num_timescales = dimEmb / 2;
    float log_timescale_increment = std::log(10000.f) / (num_timescales - 1.f);

    std::vector<float> vPos(dimEmb * dimWords, 0);
    for(int p = start; p < dimWords + start; ++p) {
      for(int i = 0; i < num_timescales; ++i) {
        float v = p * std::exp(i * -log_timescale_increment);
        vPos[(p - start) * dimEmb + i] = std::sin(v);
        vPos[(p - start) * dimEmb + num_timescales + i] = std::cos(v);
      }
    }

    // shared across batch entries
    auto signal = graph->constant({dimWords, 1, dimEmb},
                                  init = inits::from_vector(vPos));
    return input + signal;
  }

  Expr TriangleMask(Ptr<ExpressionGraph> graph, int length) {
    using namespace keywords;

    // fill triangle mask
    std::vector<float> vMask(length * length, 0);
    for(int i = 0; i < length; ++i)
      for(int j = 0; j <= i; ++j)
        vMask[i * length + j] = 1.f;
    return graph->constant({1, length, length},
                           init = inits::from_vector(vMask));
  }

  Expr InverseMask(Expr mask) {
    // convert 0/1 mask to transformer style -inf mask
    auto ms = mask->shape();
    mask = (1 - mask) * -99999999.f;
    return reshape(mask, {ms[-3], 1, ms[-2], ms[-1]}) ;
  }

  Expr SplitHeads(Expr input, int dimHeads) {
    int dimModel = input->shape()[-1];
    int dimSteps = input->shape()[-2];
    int dimBatch = input->shape()[-3];
    int dimBeam = input->shape()[-4];

    int dimDepth = dimModel / dimHeads;

    auto output
        = reshape(input, {dimBatch * dimBeam, dimSteps, dimHeads, dimDepth});

    return transpose(output, {0, 2, 1, 3});
  }

  Expr JoinHeads(Expr input, int dimBeam = 1) {
    int dimDepth = input->shape()[-1];
    int dimSteps = input->shape()[-2];
    int dimHeads = input->shape()[-3];
    int dimBatchBeam = input->shape()[-4];

    int dimModel = dimHeads * dimDepth;
    int dimBatch = dimBatchBeam / dimBeam;

    auto output = transpose(input, {0, 2, 1, 3});

    return reshape(output, {dimBeam, dimBatch, dimSteps, dimModel});
  }

  Expr PreProcess(Ptr<ExpressionGraph> graph,
                  std::string prefix,
                  std::string ops,
                  Expr input,
                  float dropProb = 0.0f) {
    using namespace keywords;

    int dimModel = input->shape()[-1];
    auto output = input;
    for(auto op : ops) {
      // dropout
      if(op == 'd' && dropProb > 0.0f) {
        auto dropMask = graph->dropout(dropProb, output->shape());
        output = dropout(output, mask = dropMask);
      }
      // layer normalization
      if(op == 'n') {
        auto scale = graph->param(
            prefix + "_ln_scale_pre", {1, dimModel}, init = inits::ones);
        auto bias = graph->param(
            prefix + "_ln_bias_pre", {1, dimModel}, init = inits::zeros);
        output = layer_norm(output, scale, bias, 1e-6);
      }
    }
    return output;
  }

  Expr PostProcess(Ptr<ExpressionGraph> graph,
                   std::string prefix,
                   std::string ops,
                   Expr input,
                   Expr prevInput,
                   float dropProb = 0.0f) {
    using namespace keywords;

    int dimModel = input->shape()[-1];
    auto output = input;
    for(auto op : ops) {
      // dropout
      if(op == 'd' && dropProb > 0.0f) {
        auto dropMask = graph->dropout(dropProb, output->shape());
        output = dropout(output, mask = dropMask);
      }
      // skip connection, moved behind layer normalization
      if(op == 'a') {
        output = output + prevInput;
      }
      // highway connection
      if(op == 'h') {
        auto Wh = graph->param(
            prefix + "_Wh", {dimModel, dimModel}, init = inits::glorot_uniform);
        auto bh
            = graph->param(prefix + "_bh", {1, dimModel}, init = inits::zeros);

        auto t = affine(prevInput, Wh, bh);
        output = highway(output, prevInput, t);
      }
      // layer normalization
      if(op == 'n') {
        auto scale = graph->param(
            prefix + "_ln_scale", {1, dimModel}, init = inits::ones);
        auto bias = graph->param(
            prefix + "_ln_bias", {1, dimModel}, init = inits::zeros);
        output = layer_norm(output, scale, bias, 1e-6);
      }
    }
    return output;
  }

  Expr Attention(Ptr<ExpressionGraph> graph,
                 Ptr<Options> options,
                 std::string prefix,
                 Expr q,
                 Expr k,
                 Expr v,
                 Expr mask = nullptr,
                 bool inference = false) {
    using namespace keywords;

    float dk = k->shape()[-1];

    // scaling to avoid extreme values due to matrix multiplication
    float scale = 1.0 / std::sqrt(dk);

    // softmax over batched dot product of query and keys (applied over all
    // time steps and batch entries), also add mask for illegal connections

    // @TODO: do this better
    int dimBeamQ = q->shape()[-4];
    int dimBeamK = k->shape()[-4];
    int dimBeam = dimBeamQ / dimBeamK;
    if(dimBeam > 1) {
      k = repeat(k, dimBeam, axis = -4);
      v = repeat(v, dimBeam, axis = -4);
    }

    auto weights = softmax(bdot(q, k, false, true, scale) + mask);

    // optional dropout for attention weights
    float dropProb
        = inference ? 0 : options->get<float>("transformer-dropout-attention");
    if(dropProb) {
      auto dropMask = graph->dropout(dropProb, weights->shape());
      weights = dropout(weights, mask = dropMask);
    }

    // apply attention weights to values
    return bdot(weights, v);
  }

  Expr MultiHead(Ptr<ExpressionGraph> graph,
                 Ptr<Options> options,
                 std::string prefix,
                 int dimOut,
                 int dimHeads,
                 Expr q,
                 const std::vector<Expr> &keys,
                 const std::vector<Expr> &values,
                 const std::vector<Expr> &masks,
                 bool inference = false) {
    using namespace keywords;

    int dimModel = q->shape()[-1];

    auto Wq = graph->param(
        prefix + "_Wq", {dimModel, dimModel}, init = inits::glorot_uniform);
    auto bq = graph->param(prefix + "_bq", {1, dimModel}, init = inits::zeros);
    auto qh = affine(q, Wq, bq);
    qh = SplitHeads(qh, dimHeads);

    std::vector<Expr> outputs;
    for(int i = 0; i < keys.size(); ++i) {
      std::string prefixProj = prefix;
      if(i > 0)
        prefixProj += "_enc" + std::to_string(i + 1);

      auto Wk = graph->param(prefixProj + "_Wk",
                             {dimModel, dimModel},
                             init = inits::glorot_uniform);
      auto bk = graph->param(
          prefixProj + "_bk", {1, dimModel}, init = inits::zeros);

      auto Wv = graph->param(prefixProj + "_Wv",
                             {dimModel, dimModel},
                             init = inits::glorot_uniform);
      auto bv = graph->param(
          prefixProj + "_bv", {1, dimModel}, init = inits::zeros);

      auto kh = affine(keys[i], Wk, bk);
      auto vh = affine(values[i], Wv, bv);

      kh = SplitHeads(kh, dimHeads);
      vh = SplitHeads(vh, dimHeads);

      // apply multi-head attention to downscaled inputs
      auto output
          = Attention(graph, options, prefix, qh, kh, vh, masks[i], inference);

      output = JoinHeads(output, q->shape()[-4]);

      outputs.push_back(output);
    }

    Expr output;
    if(outputs.size() > 1)
      output = concatenate(outputs, axis = -1);
    else
      output = outputs.front();

    int dimAtt = output->shape()[-1];

    auto Wo = graph->param(
        prefix + "_Wo", {dimAtt, dimOut}, init = inits::glorot_uniform);
    auto bo = graph->param(prefix + "_bo", {1, dimOut}, init = inits::zeros);
    output = affine(output, Wo, bo);

    return output;
  }

  Expr LayerAttention(Ptr<ExpressionGraph> graph,
                      Ptr<Options> options,
                      std::string prefix,
                      Expr input,
                      Expr keys,
                      Expr values,
                      Expr mask,
                      bool inference = false) {
    return LayerAttention(graph,
                          options,
                          prefix,
                          input,
                          std::vector<Expr>{keys},
                          std::vector<Expr>{values},
                          std::vector<Expr>{mask},
                          inference);
  }

  Expr LayerAttention(Ptr<ExpressionGraph> graph,
                      Ptr<Options> options,
                      std::string prefix,
                      Expr input,
                      const std::vector<Expr> &keys,
                      const std::vector<Expr> &values,
                      const std::vector<Expr> &masks,
                      bool inference = false) {
    using namespace keywords;

    int dimModel = input->shape()[-1];

    float dropProb = inference ? 0 : options->get<float>("transformer-dropout");
    auto opsPre = options->get<std::string>("transformer-preprocess");
    auto output = PreProcess(graph, prefix + "_Wo", opsPre, input, dropProb);

    int heads = options->get<int>("transformer-heads");

    // multi-head self-attention over previous input
    output = MultiHead(graph,
                       options,
                       prefix,
                       dimModel,
                       heads,
                       output,
                       keys,
                       values,
                       masks,
                       inference);

    auto opsPost = options->get<std::string>("transformer-postprocess");
    output
        = PostProcess(graph, prefix + "_Wo", opsPost, output, input, dropProb);

    return output;
  }

  Expr LayerFFN(Ptr<ExpressionGraph> graph,
                Ptr<Options> options,
                std::string prefix,
                Expr input,
                bool inference = false) {
    using namespace keywords;

    int dimModel = input->shape()[-1];

    float dropProb = inference ? 0 : options->get<float>("transformer-dropout");
    auto opsPre = options->get<std::string>("transformer-preprocess");
    auto output = PreProcess(graph, prefix + "_ffn", opsPre, input, dropProb);

    int dimFfn = options->get<int>("transformer-dim-ffn");

    auto W1 = graph->param(
        prefix + "_W1", {dimModel, dimFfn}, init = inits::glorot_uniform);
    auto b1 = graph->param(prefix + "_b1", {1, dimFfn}, init = inits::zeros);

    auto W2 = graph->param(
        prefix + "_W2", {dimFfn, dimModel}, init = inits::glorot_uniform);
    auto b2 = graph->param(prefix + "_b2", {1, dimModel}, init = inits::zeros);

    output = affine(output, W1, b1);
    output = swish(output);
    output = affine(output, W2, b2);

    auto opsPost = options->get<std::string>("transformer-postprocess");
    output
        = PostProcess(graph, prefix + "_ffn", opsPost, output, input, dropProb);

    return output;
  }
};

class EncoderTransformer : public EncoderBase, public Transformer {
public:
  EncoderTransformer(Ptr<Options> options) : EncoderBase(options) {}

  Expr WordEmbeddings(Ptr<ExpressionGraph> graph,
                      Ptr<data::CorpusBatch> batch) {
    // standard encoder word embeddings

    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];
    int dimEmb = opt<int>("dim-emb");

    auto embFactory = embedding(graph)("dimVocab", dimVoc)("dimEmb", dimEmb);

    if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
      embFactory("prefix", "Wemb");
    else
      embFactory("prefix", prefix_ + "_Wemb");

    if(options_->has("embedding-fix-src"))
      embFactory("fixed", opt<bool>("embedding-fix-src"));

    if(options_->has("embedding-vectors")) {
      auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
      embFactory                              //
          ("embFile", embFiles[batchIndex_])  //
          ("normalization", opt<bool>("embedding-normalization"));
    }

    return embFactory.construct();
  }

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch) {
    using namespace keywords;

    int dimEmb = opt<int>("dim-emb");
    int dimBatch = batch->size();
    int dimSrcWords = (*batch)[batchIndex_]->batchWidth();

    auto embeddings = WordEmbeddings(graph, batch);

    // select embeddings that occur in the batch
    Expr batchEmbeddings, batchMask;
    std::tie(batchEmbeddings, batchMask)
        = EncoderBase::lookup(embeddings, batch);

    // apply dropout over source words
    float dropoutSrc = inference_ ? 0 : opt<float>("dropout-src");
    if(dropoutSrc) {
      int srcWords = batchEmbeddings->shape()[-3];
      auto dropMask = graph->dropout(dropoutSrc, {srcWords, 1, 1});
      batchEmbeddings = dropout(batchEmbeddings, mask = dropMask);
    }

    // according to paper embeddings are scaled by \sqrt(d_m)
    auto scaledEmbeddings = std::sqrt(dimEmb) * batchEmbeddings;
    scaledEmbeddings = AddPositionalEmbeddings(graph, scaledEmbeddings);

    // reorganize batch and timestep
    scaledEmbeddings = atleast_nd(scaledEmbeddings, 4);
    batchMask = atleast_nd(batchMask, 4);
    auto layer = TransposeTimeBatch(scaledEmbeddings);
    auto layerMask
        = reshape(TransposeTimeBatch(batchMask), {1, dimBatch, 1, dimSrcWords});

    auto opsEmb = opt<std::string>("transformer-postprocess-emb");

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    layer = PreProcess(graph, prefix_ + "_emb", opsEmb, layer, dropProb);

    layerMask = InverseMask(layerMask);

    // apply layers
    for(int i = 1; i <= opt<int>("enc-depth"); ++i) {
      layer = LayerAttention(graph,
                             options_,
                             prefix_ + "_l" + std::to_string(i) + "_self",
                             layer,
                             layer,
                             layer,
                             layerMask,
                             inference_);

      layer = LayerFFN(graph,
                       options_,
                       prefix_ + "_l" + std::to_string(i) + "_ffn",
                       layer,
                       inference_);
    }

    // restore organization of batch and time steps. This is currently required
    // to make RNN-based decoders and beam search work with this. We are looking
    // into makeing this more natural.
    auto context = TransposeTimeBatch(layer);

    return New<EncoderState>(context, batchMask, batch);
  }

  void clear() {}
};

class TransformerState : public DecoderState {
public:
  TransformerState(const rnn::States &states,
                   Expr probs,
                   std::vector<Ptr<EncoderState>> &encStates)
      : DecoderState(states, probs, encStates) {}

  virtual Ptr<DecoderState> select(const std::vector<size_t> &selIdx, int beamSize) {
    rnn::States selectedStates;

    int dimDepth = states_[0].output->shape()[-1];
    int dimTime  = states_[0].output->shape()[-2];
    int dimBatch = selIdx.size() / beamSize;

    std::vector<size_t> selIdx2;
    for(auto i : selIdx)
      for(int j = 0; j < dimTime; ++j)
        selIdx2.push_back(i * dimTime + j);

    for(auto state : states_) {
      auto sel = rows(flatten_2d(state.output), selIdx2);
      sel = reshape(sel, {beamSize, dimBatch, dimTime, dimDepth});
      selectedStates.push_back({sel, nullptr});
    }

    return New<TransformerState>(selectedStates, probs_, encStates_);
  }
};

class DecoderTransformer : public DecoderBase, public Transformer {
public:
  DecoderTransformer(Ptr<Options> options) : DecoderBase(options) {}

  virtual Ptr<DecoderState> startState(
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch,
      std::vector<Ptr<EncoderState>> &encStates) {
    rnn::States startStates;
    return New<TransformerState>(startStates, nullptr, encStates);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    using namespace keywords;

    auto embeddings = state->getTargetEmbeddings();
    auto decoderMask = state->getTargetMask();

    // dropout target words
    float dropoutTrg = inference_ ? 0 : opt<float>("dropout-trg");
    if(dropoutTrg) {
      int trgWords = embeddings->shape()[-3];
      auto trgWordDrop = graph->dropout(dropoutTrg, {trgWords, 1, 1});
      embeddings = dropout(embeddings, mask = trgWordDrop);
    }

    //************************************************************************//

    int dimEmb = embeddings->shape()[-1];
    int dimBeam = 1;
    if(embeddings->shape().size() > 3)
      dimBeam = embeddings->shape()[-4];

    // according to paper embeddings are scaled by \sqrt(d_m)
    auto scaledEmbeddings = std::sqrt(dimEmb) * embeddings;

    int startPos = 0;
    auto prevDecoderStates = state->getStates();
    if(prevDecoderStates.size() > 0)
      startPos = prevDecoderStates[0].output->shape()[-2];

    scaledEmbeddings
        = AddPositionalEmbeddings(graph, scaledEmbeddings, startPos);

    scaledEmbeddings = atleast_nd(scaledEmbeddings, 4);

    // reorganize batch and timestep
    auto query = TransposeTimeBatch(scaledEmbeddings);

    auto opsEmb = opt<std::string>("transformer-postprocess-emb");
    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");

    query = PreProcess(graph, prefix_ + "_emb", opsEmb, query, dropProb);

    rnn::States decoderStates;
    int dimTrgWords = query->shape()[-2];
    int dimBatch = query->shape()[-3];
    auto selfMask = TriangleMask(graph, dimTrgWords);
    if(decoderMask) {
      decoderMask = atleast_nd(decoderMask, 4);
      decoderMask = reshape(TransposeTimeBatch(decoderMask),
                            {1, dimBatch, 1, dimTrgWords});
      selfMask = selfMask * decoderMask;
      //if(dimBeam > 1)
      //  selfMask = repeat(selfMask, dimBeam, axis = -4);
    }

    selfMask = InverseMask(selfMask);

    std::vector<Expr> encoderContexts;
    std::vector<Expr> encoderMasks;

    for(auto encoderState : state->getEncoderStates()) {
      auto encoderContext = encoderState->getContext();
      auto encoderMask = encoderState->getMask();

      encoderContext = TransposeTimeBatch(encoderContext);

      int dimSrcWords = encoderContext->shape()[-2];

      int dims = encoderMask->shape().size();
      encoderMask = atleast_nd(encoderMask, 4);
      encoderMask = reshape(TransposeTimeBatch(encoderMask),
                            {1, dimBatch, 1, dimSrcWords});
      encoderMask = InverseMask(encoderMask);
      if(dimBeam > 1)
        encoderMask = repeat(encoderMask, dimBeam, axis = -4);

      encoderContexts.push_back(encoderContext);
      encoderMasks.push_back(encoderMask);
    }

    // apply layers
    for(int i = 1; i <= opt<int>("dec-depth"); ++i) {
      auto values = query;
      if(prevDecoderStates.size() > 0)
        values = concatenate({prevDecoderStates[i - 1].output, query}, axis = -2);

      decoderStates.push_back({values, nullptr});

      // TODO: do not recompute matrix multiplies
      query = LayerAttention(graph,
                             options_,
                             prefix_ + "_l" + std::to_string(i) + "_self",
                             query,
                             values,
                             values,
                             selfMask,
                             inference_);

      if(encoderContexts.size() > 0) {
        // auto comb = opt<std::string>("transformer-multi-encoder");
        std::string comb = "stack";
        if(comb == "concat") {
          query
              = LayerAttention(graph,
                               options_,
                               prefix_ + "_l" + std::to_string(i) + "_context",
                               query,
                               encoderContexts,
                               encoderContexts,
                               encoderMasks,
                               inference_);

        } else if(comb == "stack") {
          for(int j = 0; j < encoderContexts.size(); ++j) {
            std::string prefix
                = prefix_ + "_l" + std::to_string(i) + "_context";
            if(j > 0)
              prefix += "_enc" + std::to_string(j + 1);

            query = LayerAttention(graph,
                                   options_,
                                   prefix,
                                   query,
                                   encoderContexts[j],
                                   encoderContexts[j],
                                   encoderMasks[j],
                                   inference_);
          }
        } else {
          ABORT("Unknown value for transformer-multi-encoder: {}", comb);
        }
      }

      query = LayerFFN(graph,
                       options_,
                       prefix_ + "_l" + std::to_string(i) + "_ffn",
                       query,
                       inference_);
    }

    auto decoderContext = TransposeTimeBatch(query);

    //************************************************************************//

    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs").back();

    auto layerOut = mlp::dense(graph)          //
        ("prefix", prefix_ + "_ff_logit_out")  //
        ("dim", dimTrgVoc);

    if(opt<bool>("tied-embeddings") || opt<bool>("tied-embeddings-all")) {
      std::string tiedPrefix = prefix_ + "_Wemb";
      if(opt<bool>("tied-embeddings-all") || opt<bool>("tied-embeddings-src"))
        tiedPrefix = "Wemb";
      layerOut.tie_transposed("W", tiedPrefix);
    }

    // assemble layers into MLP and apply to embeddings, decoder context and
    // aligned source context
    auto output = mlp::mlp(graph).push_back(layerOut);

    Expr logits = output->apply(decoderContext);

    // return unormalized(!) probabilities
    return New<TransformerState>(
        decoderStates, logits, state->getEncoderStates());
  }

  // helper function for guided alignment
  virtual const std::vector<Expr> getAlignments(int i = 0) { return {}; }

  void clear() {}
};
}

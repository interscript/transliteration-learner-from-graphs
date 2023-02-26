
require_relative "encoders"


module Transliterator

  # A generic diacritizer to be overloaded by the respective languages.
  class Transformer

    def initialize(config)

      # load onnx models
      onnx_models_path = config["file_paths"]["onnx_models_path"]
      onnx_models_path = "/home/jair/WORK/Interscript/transliteration-learner-from-graphs/ressources/transformer_model.onnx"

      # load inference model from model_path
      @onnx_model = OnnxRuntime::Model.new(onnx_models_path)

    end


    def create_mask(n)

      mask = Array.new(n) { Array.new(n, false) }
      (0..n-1).map {|i| (i+1..n-1).map {|j| mask[i][j]=true }}
      mask

    end


    def predicts(src)

      d_src = {"idx": src}
      tokens = @onnx_model.predict(d_src)
      tokens["logits"]

    end


    def encode(src)

      d_src = {"idx": src}
      # output shape of src
      tokens = @onnx_model.predict(d_src)
      d_tokens = {"logits": tokens["logit"]}
      pos = @positional_embedding.predict(d_tokens)
      d_pos_src_mask = {src: pos["output"], src_mask: src_mask}
      @transformer_encoder.predict(d_pos_src_mask)["output"]

    end


    def decode(ys, memory, tgt_mask)

      d_tgt = {"tgt": ys}

      tokens = @token_tgt_embedding.predict(d_tgt)
      d_tokens = {"tokens": tokens["output"]}

      pos = @positional_embedding.predict(d_tokens)
      tgt = pos["output"]

      d_data = {tgt: tgt, memory: memory, tgt_mask: tgt_mask}
      out = @transformer_decoder.predict(d_data)
      out["output"]

    end


    def greedy_decode(src)

      num_tokens = src.length
      max_len = num_tokens + 5
      src_mask = Array.new(num_tokens) { Array.new(num_tokens, false) }
      memory = encode(src, src_mask)

      start_symbol=2 # BOS_IDX
      ys = [[start_symbol]]

      tgt_tokens = (0..max_len-2+1).map { |i|

          sz = ys.length
          tgt_mask = create_mask(sz)
          out = decode(ys, memory, tgt_mask)
          d_outs = {outs: [out.transpose[0][-1]]}

          probas = @transformer_generator.predict(d_outs)
          max = probas["output"][0].each_with_index.max[1]

          ys = ys + [[max]]
          if max == 3 # EOS_IDX
            break
          end
      }

      Array(ys).map {|x| x[0]}

    end

  end

end

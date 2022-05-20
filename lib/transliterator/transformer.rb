
require_relative "vocab"
require_relative "encoders"


module Transliterator

  # A generic diacritizer to be overloaded by the respective languages.
  class Transformer

    def initialize(config)

      @onnx_models_path = config["transliteration"]["ONNX_RUBY_DIR"]

      # load inference model from model_path
      token_src_embbedding_path = @onnx_models_path+"token_src_embbedding.onnx"
      @token_src_embedding = OnnxRuntime::Model.new(token_src_embbedding_path)

      token_tgt_embbedding_path = @onnx_models_path+"token_tgt_embbedding.onnx"
      @token_tgt_embedding = OnnxRuntime::Model.new(token_tgt_embbedding_path)

      positional_embbedding_path = @onnx_models_path+"positional_embbedding.onnx"
      @positional_embedding = OnnxRuntime::Model.new(positional_embbedding_path)

      transformer_generator_path = @onnx_models_path+"transformer_generator.onnx"
      @transformer_generator = OnnxRuntime::Model.new(transformer_generator_path)

      transformer_encoder_path = @onnx_models_path+"transformer_encoder.onnx"
      @transformer_encoder = OnnxRuntime::Model.new(transformer_encoder_path)

      transformer_decoder_path = @onnx_models_path+"transformer_decoder.onnx"
      @transformer_decoder = OnnxRuntime::Model.new(transformer_decoder_path)

    end


    def create_mask(n)

      mask = Array.new(n) { Array.new(n, false) }
      (0..n-1).map {|i| (i+1..n-1).map {|j| mask[i][j]=true }}
      mask

    end


    def encode(src, src_mask)

      d_src = {"src": src}
      tokens = @token_src_embedding.predict(d_src)
      d_tokens = {"tokens": tokens["output"]}
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

# ported from https://github.com/pytorch/pytorch/blob/626e410e1dedcdb9d5a410a8827cc7a8a9fbcce1/torch/nn/modules/transformer.py
require_relative "vocab"
require_relative "encoders"
#require_relative "titi"

require "torch"

module Transliterator

  # A generic diacritizer to be overloaded by the respective languages.
  class Transformer

    attr_accessor :onnx_session

    def initialize(onnx_model_path, config)

      # load inference model from model_path
      @onnx_session = OnnxRuntime::Model.new(onnx_model_path)
      # OnnxRuntime::InferenceSession.new(onnx_model_path)

      # load config
      @config = config
      @d_model= @config[:d_model]
      @dropout = @config[:dropout]
      @batch_size = @config[:batch_size]
      @max_len = @config[:max_len]

    end

  end

end
